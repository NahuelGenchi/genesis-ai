from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .ingest import sha256_file
from .model import GenesisLM
from .research import FLOP_ESTIMATOR_VERSION, estimated_training_flops_per_token
from .tokenizer import ByteBPETokenizer

SCALING_FORMAT_VERSION = "1.0"


def load_ladder(path: str | Path, *, vocab_size: int) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version",
        "name",
        "seed",
        "target_tokens_per_step",
        "warmup_steps",
        "measured_steps",
        "feasibility",
        "promotion",
        "stages",
    }
    if not isinstance(raw, dict) or set(raw) != required or raw.get("format_version") != SCALING_FORMAT_VERSION:
        raise ValueError("unsupported scaling ladder definition")
    if not isinstance(raw["stages"], list) or len(raw["stages"]) < 2:
        raise ValueError("scaling ladder requires a reference and candidate stage")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise ValueError("seed must be an integer")
    for name in ("target_tokens_per_step", "warmup_steps", "measured_steps"):
        if not isinstance(raw[name], int) or isinstance(raw[name], bool) or raw[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")

    feasibility = raw["feasibility"]
    if set(feasibility) != {"local_max_training_hours", "local_peak_rss_max_mb"}:
        raise ValueError("invalid feasibility policy")
    for name, value in feasibility.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid feasibility value: {name}")

    promotion = raw["promotion"]
    required_promotion = {
        "min_domain_accuracy_absolute_gain",
        "max_m3_validation_loss_regression_fraction",
        "require_zero_exact_contamination",
        "require_reproducible_checkpoint",
        "require_zero_cash_compute",
    }
    if set(promotion) != required_promotion:
        raise ValueError("invalid scale-promotion policy")
    for name in ("min_domain_accuracy_absolute_gain", "max_m3_validation_loss_regression_fraction"):
        value = promotion[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
            raise ValueError(f"invalid scale-promotion threshold: {name}")
    for name in ("require_zero_exact_contamination", "require_reproducible_checkpoint", "require_zero_cash_compute"):
        if not isinstance(promotion[name], bool):
            raise ValueError(f"scale-promotion flag must be boolean: {name}")

    seen: set[str] = set()
    reference_count = 0
    previous_training_tokens = 0
    for stage in raw["stages"]:
        if not isinstance(stage, dict) or set(stage) != {"name", "role", "training_tokens", "config"}:
            raise ValueError("invalid scaling stage")
        name = stage["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("stage names must be unique non-empty strings")
        seen.add(name)
        if stage["role"] not in {"reference", "candidate"}:
            raise ValueError("stage role must be reference or candidate")
        reference_count += int(stage["role"] == "reference")
        tokens = stage["training_tokens"]
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0 or tokens < previous_training_tokens:
            raise ValueError("stage training tokens must be positive and non-decreasing")
        previous_training_tokens = tokens
        config_values = dict(stage["config"])
        config_values["vocab_size"] = vocab_size
        ModelConfig.from_dict(config_values)
    if reference_count != 1 or raw["stages"][0]["role"] != "reference":
        raise ValueError("first and only one stage must be the reference")
    return raw


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark_stage(
    stage: dict[str, Any],
    *,
    vocab_size: int,
    seed: int,
    target_tokens_per_step: int,
    warmup_steps: int,
    measured_steps: int,
    feasibility: dict[str, float],
) -> dict[str, Any]:
    config_values = dict(stage["config"])
    config_values["vocab_size"] = vocab_size
    config = ModelConfig.from_dict(config_values)
    torch.manual_seed(seed)
    model = GenesisLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = max(1, target_tokens_per_step // config.context_length)
    tokens_per_step = batch_size * config.context_length
    x = torch.randint(0, config.vocab_size, (batch_size, config.context_length))
    y = torch.randint(0, config.vocab_size, (batch_size, config.context_length))

    model.train()
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()

    started = time.perf_counter()
    last_loss = 0.0
    for _ in range(measured_steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    elapsed = time.perf_counter() - started
    measured_tokens = tokens_per_step * measured_steps
    training_tps = measured_tokens / elapsed
    target_training_tokens = int(stage["training_tokens"])
    estimated_hours = target_training_tokens / training_tps / 3600.0
    flops_per_token = estimated_training_flops_per_token(model)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer_state_bytes = _tensor_bytes(optimizer.state_dict()["state"])
    peak_rss = _peak_rss_mb()
    feasible = (
        estimated_hours <= float(feasibility["local_max_training_hours"])
        and peak_rss <= float(feasibility["local_peak_rss_max_mb"])
    )
    return {
        "name": stage["name"],
        "role": stage["role"],
        "config": config.to_dict(),
        "parameter_count": model.parameter_count(),
        "active_parameter_count": model.estimated_active_parameter_count(),
        "flop_estimator": FLOP_ESTIMATOR_VERSION,
        "estimated_training_flops_per_token": flops_per_token,
        "target_training_tokens": target_training_tokens,
        "estimated_target_training_flops": target_training_tokens * flops_per_token,
        "batch_size": batch_size,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "measured_tokens": measured_tokens,
        "measured_seconds": elapsed,
        "training_tokens_per_second": training_tps,
        "last_random_training_loss": last_loss,
        "estimated_target_training_hours": estimated_hours,
        "model_parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_state_bytes": optimizer_state_bytes,
        "peak_process_rss_mb": peak_rss,
        "local_cpu_feasible": feasible,
    }


def run_ladder(definition_path: str | Path, tokenizer_path: str | Path) -> dict[str, Any]:
    definition_path = Path(definition_path)
    tokenizer_path = Path(tokenizer_path)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    ladder = load_ladder(definition_path, vocab_size=tokenizer.vocab_size)
    results = []
    for ordinal, stage in enumerate(ladder["stages"]):
        results.append(
            benchmark_stage(
                stage,
                vocab_size=tokenizer.vocab_size,
                seed=int(ladder["seed"]) + ordinal,
                target_tokens_per_step=int(ladder["target_tokens_per_step"]),
                warmup_steps=int(ladder["warmup_steps"]),
                measured_steps=int(ladder["measured_steps"]),
                feasibility=ladder["feasibility"],
            )
        )
    candidates = [result for result in results if result["role"] == "candidate"]
    next_stage = next((result["name"] for result in candidates if result["local_cpu_feasible"]), None)
    return {
        "format_version": SCALING_FORMAT_VERSION,
        "experiment": ladder["name"],
        "definition_sha256": sha256_file(definition_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "cash_compute_cost_usd": 0.0,
        "feasibility": ladder["feasibility"],
        "promotion": ladder["promotion"],
        "stages": results,
        "next_stage": next_stage,
        "ladder_rule": "advance sequentially; a larger stage is not authorized until the prior candidate passes the promotion criteria",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Genesis evidence-based model scaling ladder.")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_ladder(args.definition, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
