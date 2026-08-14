from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer

FLOP_ESTIMATOR_VERSION = "dense-training-v1"


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    config: ModelConfig
    parameter_count: int
    training_flops_per_token: int
    batch_size: int
    tokens_per_step: int
    steps: int
    estimated_flops: int


def estimated_training_flops_per_token(model: GenesisLM) -> int:
    config = model.config
    # Coarse training estimator: 6 FLOPs/parameter/token plus explicit
    # quadratic attention work (QK + AV, forward/backward approximation).
    dense = 6 * model.parameter_count()
    attention = 12 * config.n_layers * config.context_length * config.d_model
    return dense + attention


def plan_candidate(
    name: str,
    config: ModelConfig,
    *,
    training_flop_budget: int,
    target_tokens_per_step: int,
) -> CandidatePlan:
    if training_flop_budget <= 0 or target_tokens_per_step <= 0:
        raise ValueError("compute budget and target tokens/step must be positive")
    model = GenesisLM(config)
    flops_per_token = estimated_training_flops_per_token(model)
    batch_size = max(1, target_tokens_per_step // config.context_length)
    tokens_per_step = batch_size * config.context_length
    steps = max(1, training_flop_budget // (flops_per_token * tokens_per_step))
    estimated_flops = steps * tokens_per_step * flops_per_token
    return CandidatePlan(
        name=name,
        config=config,
        parameter_count=model.parameter_count(),
        training_flops_per_token=flops_per_token,
        batch_size=batch_size,
        tokens_per_step=tokens_per_step,
        steps=steps,
        estimated_flops=estimated_flops,
    )


def _validation_loss(model: GenesisLM, dataset: TokenDataset, device: str, batches: int) -> tuple[float, int]:
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            if index >= batches:
                break
            _, loss = model(x.to(device), y.to(device))
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("no validation batches")
    return sum(losses) / len(losses), len(losses)


def run_candidate(
    plan: CandidatePlan,
    *,
    train_data: TokenDataset,
    validation_data: TokenDataset,
    seed: int,
    learning_rate: float,
    validation_batches: int,
    device: str,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    model = GenesisLM(plan.config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    initial_loss, initial_batches = _validation_loss(model, validation_data, device, validation_batches)
    started = time.perf_counter()
    model.train()
    last_loss = float("nan")
    for _ in range(plan.steps):
        x, y = sample_batch(train_data, plan.batch_size, generator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    elapsed = time.perf_counter() - started
    final_loss, final_batches = _validation_loss(model, validation_data, device, validation_batches)

    return {
        "name": plan.name,
        "seed": seed,
        "config": plan.config.to_dict(),
        "parameter_count": plan.parameter_count,
        "flop_estimator": FLOP_ESTIMATOR_VERSION,
        "training_flops_per_token": plan.training_flops_per_token,
        "estimated_training_flops": plan.estimated_flops,
        "steps": plan.steps,
        "batch_size": plan.batch_size,
        "tokens_per_step": plan.tokens_per_step,
        "tokens_seen": plan.steps * plan.tokens_per_step,
        "learning_rate": learning_rate,
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "validation_batches": min(initial_batches, final_batches),
        "last_training_loss": last_loss,
        "training_seconds": elapsed,
        "training_tokens_per_second": (plan.steps * plan.tokens_per_step) / elapsed,
        "loss_improvement": initial_loss - final_loss,
    }


def load_experiment(path: Path, tokenizer: ByteBPETokenizer) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format_version") != "1.0":
        raise ValueError("unsupported experiment definition")
    required = {"format_version", "name", "seed", "training_flop_budget", "target_tokens_per_step", "learning_rate", "validation_batches", "candidates"}
    if set(raw) != required:
        raise ValueError("experiment definition has unexpected fields")
    if not isinstance(raw["candidates"], list) or len(raw["candidates"]) < 2:
        raise ValueError("experiment needs at least two candidates")
    names: set[str] = set()
    for candidate in raw["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != {"name", "config"}:
            raise ValueError("candidate must contain name and config")
        name = candidate["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("candidate names must be unique non-empty strings")
        names.add(name)
        if not isinstance(candidate["config"], dict):
            raise ValueError("candidate config must be an object")
        values = dict(candidate["config"])
        values["vocab_size"] = tokenizer.vocab_size
        ModelConfig.from_dict(values)
    return raw


def run_experiment(
    definition_path: Path,
    data_dir: Path,
    tokenizer_path: Path,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    definition = load_experiment(definition_path, tokenizer)
    seed = int(definition["seed"])
    training_flop_budget = int(definition["training_flop_budget"])
    target_tokens_per_step = int(definition["target_tokens_per_step"])
    learning_rate = float(definition["learning_rate"])
    validation_batches = int(definition["validation_batches"])
    if learning_rate <= 0 or validation_batches <= 0:
        raise ValueError("learning_rate and validation_batches must be positive")

    results: list[dict[str, object]] = []
    plans: list[CandidatePlan] = []
    for candidate in definition["candidates"]:
        config_values = dict(candidate["config"])
        config_values["vocab_size"] = tokenizer.vocab_size
        config = ModelConfig.from_dict(config_values)
        plan = plan_candidate(
            str(candidate["name"]),
            config,
            training_flop_budget=training_flop_budget,
            target_tokens_per_step=target_tokens_per_step,
        )
        plans.append(plan)

    max_context = max(plan.config.context_length for plan in plans)
    train_data = TokenDataset(data_dir, tokenizer, max_context, split="train")
    validation_cache: dict[int, TokenDataset] = {}
    for plan in plans:
        train_for_context = TokenDataset(data_dir, tokenizer, plan.config.context_length, split="train")
        validation = validation_cache.setdefault(
            plan.config.context_length,
            TokenDataset(data_dir, tokenizer, plan.config.context_length, split="validation"),
        )
        results.append(
            run_candidate(
                plan,
                train_data=train_for_context,
                validation_data=validation,
                seed=seed,
                learning_rate=learning_rate,
                validation_batches=validation_batches,
                device=device,
            )
        )

    best = min(results, key=lambda result: float(result["final_validation_loss"]))
    budget_min = min(int(result["estimated_training_flops"]) for result in results)
    budget_max = max(int(result["estimated_training_flops"]) for result in results)
    return {
        "format_version": "1.0",
        "experiment": definition["name"],
        "definition_sha256": sha256_file(definition_path),
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "seed": seed,
        "requested_training_flop_budget": training_flop_budget,
        "flop_estimator": FLOP_ESTIMATOR_VERSION,
        "budget_spread_fraction": (budget_max - budget_min) / training_flop_budget,
        "results": results,
        "winner_by_validation_loss": best["name"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-compute Genesis AI architecture experiments.")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_experiment(args.definition, args.data, args.tokenizer, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
