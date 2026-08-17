from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer

PREFLIGHT_VERSION = "m6-scale-5m-rope-preflight-v1"
FINALIST_VERSION = "m6-architecture-finalist-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_preflight_contract(
    definition_path: str | Path,
    finalist_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], ModelConfig]:
    definition = _load_json(definition_path)
    finalist = _load_json(finalist_path)
    if definition.get("format_version") != "1.0" or definition.get("name") != PREFLIGHT_VERSION:
        raise ValueError("unsupported scale preflight definition")
    if finalist.get("format_version") != "1.0" or finalist.get("finalist_version") != FINALIST_VERSION:
        raise ValueError("unsupported architecture finalist evidence")
    if finalist.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("architecture finalist violates zero-cash contract")
    decision = finalist.get("decision")
    if not isinstance(decision, dict) or decision.get("passed") is not True:
        raise ValueError("scale preflight requires a reproduced architecture winner")
    accepted = definition.get("accepted_architecture")
    if not isinstance(accepted, str) or accepted != finalist.get("accepted_architecture"):
        raise ValueError("scale preflight architecture does not match reproduced finalist")
    if accepted != "rope-only":
        raise ValueError("scale preflight v1 is frozen to the reproduced RoPE-only winner")

    raw_config = definition.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("scale preflight model config is missing")
    config = ModelConfig.from_dict(raw_config)
    if config.position_encoding != "rotary":
        raise ValueError("scale preflight v1 requires rotary position encoding")
    model = GenesisLM(config)
    expected_parameters = int(definition.get("expected_parameter_count", -1))
    if model.parameter_count() != expected_parameters:
        raise ValueError(
            f"scale preflight parameter count mismatch: expected {expected_parameters}, got {model.parameter_count()}"
        )

    target_tokens = int(definition.get("target_training_tokens_per_replica", 0))
    target_tokens_per_step = int(definition.get("target_tokens_per_step", 0))
    warmup_steps = int(definition.get("warmup_steps", 0))
    measured_steps = int(definition.get("measured_steps", 0))
    max_seconds = int(definition.get("maximum_projected_training_seconds_per_replica", 0))
    threads = int(definition.get("torch_threads", 0))
    if target_tokens != 20_000_000:
        raise ValueError("scale preflight v1 is frozen to 20M tokens per replica")
    if target_tokens_per_step <= 0 or target_tokens_per_step % config.context_length != 0:
        raise ValueError("target tokens per step must divide evenly by context length")
    if warmup_steps < 1 or measured_steps < 10:
        raise ValueError("scale preflight sample is too small")
    if max_seconds <= 0 or max_seconds >= 6 * 60 * 60:
        raise ValueError("scale preflight must leave margin below the hosted job ceiling")
    if not 1 <= threads <= 4:
        raise ValueError("scale preflight v1 allows 1-4 CPU threads")
    return definition, finalist, config


def _train_step(
    model: GenesisLM,
    optimizer: torch.optim.Optimizer,
    dataset: TokenDataset,
    generator: torch.Generator,
    *,
    batch_size: int,
) -> float:
    x, y = sample_batch(dataset, batch_size, generator)
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def run_preflight(
    *,
    definition_path: str | Path,
    finalist_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("scale preflight v1 is frozen to CPU")
    definition_path = Path(definition_path)
    finalist_path = Path(finalist_path)
    public_data = Path(public_data)
    tokenizer_path = Path(tokenizer_path)
    definition, finalist, config = load_preflight_contract(definition_path, finalist_path)

    threads = int(definition["torch_threads"])
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    seed = int(definition["seed"])
    torch.manual_seed(seed)

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    dataset = TokenDataset(public_data, tokenizer, config.context_length, split="train")
    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(definition["learning_rate"]),
        foreach=False,
        fused=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    tokens_per_step = int(definition["target_tokens_per_step"])
    batch_size = tokens_per_step // config.context_length
    warmup_steps = int(definition["warmup_steps"])
    measured_steps = int(definition["measured_steps"])

    model.train()
    warmup_losses = [
        _train_step(model, optimizer, dataset, generator, batch_size=batch_size)
        for _ in range(warmup_steps)
    ]
    measured_loss_sum = 0.0
    started = time.perf_counter()
    for _ in range(measured_steps):
        measured_loss_sum += _train_step(model, optimizer, dataset, generator, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        raise ValueError("invalid scale preflight elapsed time")

    measured_tokens = measured_steps * tokens_per_step
    tokens_per_second = measured_tokens / elapsed
    target_tokens = int(definition["target_training_tokens_per_replica"])
    projected_training_seconds = target_tokens / tokens_per_second
    maximum_seconds = int(definition["maximum_projected_training_seconds_per_replica"])
    passed = projected_training_seconds <= maximum_seconds
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    active_flops_per_token = int(model.estimated_active_flops_per_token())

    return {
        "format_version": "1.0",
        "preflight_version": PREFLIGHT_VERSION,
        "definition_sha256": sha256_file(definition_path),
        "finalist_sha256": sha256_file(finalist_path),
        "finalist_result_provenance": finalist.get("result_provenance"),
        "accepted_architecture": finalist["accepted_architecture"],
        "config": config.to_dict(),
        "parameter_count": model.parameter_count(),
        "active_flops_per_token": active_flops_per_token,
        "target_training_tokens_per_replica": target_tokens,
        "estimated_training_flops_per_replica": active_flops_per_token * target_tokens,
        "sample": {
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "batch_size": batch_size,
            "tokens_per_step": tokens_per_step,
            "measured_tokens": measured_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
            "mean_warmup_loss": sum(warmup_losses) / len(warmup_losses),
            "mean_measured_loss": measured_loss_sum / measured_steps,
            "peak_rss_kib": peak_rss_kib,
        },
        "projection": {
            "training_seconds_per_replica": projected_training_seconds,
            "training_hours_per_replica": projected_training_seconds / 3600.0,
            "maximum_training_seconds_per_replica": maximum_seconds,
            "maximum_training_hours_per_replica": maximum_seconds / 3600.0,
            "passes_remote_cpu_time_budget": passed,
            "failure_route": None if passed else "free-accelerator-required",
        },
        "runner_contract": "github-hosted-ubuntu-latest-cpu",
        "determinism": {
            "device": device,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "adamw_foreach": False,
            "adamw_fused": False,
            "seed": seed,
        },
        "cash_compute_cost_usd": 0.0,
        "promotion_authority": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the exact reproduced ~5M RoPE scale candidate on free CPU.")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--finalist", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_preflight(
        definition_path=args.definition,
        finalist_path=args.finalist,
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
