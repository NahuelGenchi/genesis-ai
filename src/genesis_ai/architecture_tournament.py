from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer

TOURNAMENT_VERSION = "m6-architecture-tournament-v1"
CPU_THREADS = 1


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _candidate_config(values: dict[str, Any]) -> ModelConfig:
    config_values = dict(values)
    config_values.setdefault("vocab_size", 512)
    config = ModelConfig(**config_values)
    config.validate()
    if config.vocab_size != 512:
        raise ValueError("M6 architecture tournament is frozen to vocab_size=512")
    return config


def plan_candidate(config: ModelConfig, *, flop_budget: int, target_tokens_per_step: int) -> dict[str, int]:
    model = GenesisLM(config)
    active_flops_per_token = int(model.estimated_active_flops_per_token())
    if active_flops_per_token <= 0:
        raise ValueError("active FLOPs/token must be positive")
    max_tokens = flop_budget // active_flops_per_token
    steps = max_tokens // target_tokens_per_step
    if steps <= 0:
        raise ValueError("FLOP budget is too small for one full training step")
    processed_tokens = steps * target_tokens_per_step
    estimated_training_flops = processed_tokens * active_flops_per_token
    if estimated_training_flops > flop_budget:
        raise AssertionError("planned candidate exceeds FLOP budget")
    return {
        "active_parameters": int(model.estimated_active_parameter_count()),
        "total_parameters": int(model.parameter_count()),
        "active_flops_per_token": active_flops_per_token,
        "steps": int(steps),
        "processed_tokens": int(processed_tokens),
        "estimated_training_flops": int(estimated_training_flops),
        "unused_flop_budget": int(flop_budget - estimated_training_flops),
    }


def _evaluate(
    model: GenesisLM,
    dataset: TokenDataset,
    *,
    batches: int,
    batch_size: int,
    seed: int,
    device: str,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(dataset, batch_size, generator)
            _, loss = model(x.to(device), y.to(device))
            assert loss is not None
            total += float(loss.detach().cpu())
    return total / batches


def run_candidate(
    *,
    name: str,
    config: ModelConfig,
    public_data: str | Path,
    tokenizer: ByteBPETokenizer,
    flop_budget: int,
    target_tokens_per_step: int,
    learning_rate: float,
    validation_batches: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("M6 architecture tournament is frozen to deterministic CPU")
    if target_tokens_per_step % config.context_length != 0:
        raise ValueError("target_tokens_per_step must divide evenly by context_length")
    batch_size = target_tokens_per_step // config.context_length
    plan = plan_candidate(config, flop_budget=flop_budget, target_tokens_per_step=target_tokens_per_step)

    random.seed(seed)
    torch.manual_seed(seed)
    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False, fused=False)
    train_dataset = TokenDataset(public_data, tokenizer, config.context_length, split="train")
    validation_dataset = TokenDataset(public_data, tokenizer, config.context_length, split="validation")
    train_generator = torch.Generator(device="cpu").manual_seed(seed + 1)

    initial_validation_loss = _evaluate(
        model,
        validation_dataset,
        batches=validation_batches,
        batch_size=batch_size,
        seed=seed + 2,
        device=device,
    )
    loss_sum = 0.0
    model.train()
    started = time.perf_counter()
    for _ in range(1, plan["steps"] + 1):
        x, y = sample_batch(train_dataset, batch_size, train_generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach().cpu())
    elapsed = time.perf_counter() - started
    final_validation_loss = _evaluate(
        model,
        validation_dataset,
        batches=validation_batches,
        batch_size=batch_size,
        seed=seed + 2,
        device=device,
    )
    quality_gain = initial_validation_loss - final_validation_loss
    return {
        "name": name,
        "config": config.to_dict(),
        "plan": plan,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "validation_loss_improvement": quality_gain,
        "mean_training_loss": loss_sum / plan["steps"],
        "wall_seconds": elapsed,
        "training_tokens_per_second": plan["processed_tokens"] / elapsed if elapsed > 0 else 0.0,
        "quality_gain_per_1e12_flops": quality_gain / (plan["estimated_training_flops"] / 1_000_000_000_000),
    }


def run_tournament(
    *,
    definition_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    definition_path = Path(definition_path)
    public_data = Path(public_data)
    tokenizer_path = Path(tokenizer_path)
    definition = _load_json(definition_path)
    if definition.get("format_version") != "1.0" or definition.get("name") != TOURNAMENT_VERSION:
        raise ValueError("unsupported architecture tournament definition")
    flop_budget = int(definition["training_flop_budget"])
    target_tokens_per_step = int(definition["target_tokens_per_step"])
    learning_rate = float(definition["learning_rate"])
    validation_batches = int(definition["validation_batches"])
    seed = int(definition["seed"])
    candidates = definition.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("architecture tournament requires at least two candidates")
    names = [candidate.get("name") for candidate in candidates if isinstance(candidate, dict)]
    if len(names) != len(candidates) or len(set(names)) != len(names):
        raise ValueError("architecture candidate names must be unique")

    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    results = []
    for candidate in candidates:
        config = _candidate_config(candidate["config"])
        results.append(
            run_candidate(
                name=str(candidate["name"]),
                config=config,
                public_data=public_data,
                tokenizer=tokenizer,
                flop_budget=flop_budget,
                target_tokens_per_step=target_tokens_per_step,
                learning_rate=learning_rate,
                validation_batches=validation_batches,
                seed=seed,
                device=device,
            )
        )

    winner = min(results, key=lambda result: (float(result["final_validation_loss"]), str(result["name"])))
    baseline = next((result for result in results if result["name"] == "baseline-learned-layernorm-gelu"), None)
    if baseline is None:
        raise ValueError("tournament definition must contain the frozen baseline")
    for result in results:
        result["validation_loss_delta_vs_baseline"] = float(result["final_validation_loss"]) - float(baseline["final_validation_loss"])
        result["validation_loss_relative_vs_baseline"] = float(result["final_validation_loss"]) / float(baseline["final_validation_loss"]) - 1.0

    return {
        "format_version": "1.0",
        "tournament_version": TOURNAMENT_VERSION,
        "definition_sha256": sha256_file(definition_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "cash_compute_cost_usd": 0.0,
        "determinism": {"device": "cpu", "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True},
        "selection_rule": "lowest_final_validation_loss_at_fixed_training_flop_budget",
        "winner": winner["name"],
        "baseline": baseline["name"],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-FLOP Genesis architecture tournament.")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_tournament(
        definition_path=args.definition,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
