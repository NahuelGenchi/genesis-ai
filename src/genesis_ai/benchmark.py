from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch

from .checkpoint import load_model, tokenizer_from_payload
from .ingest import sha256_file


def cost_per_million_tokens(hourly_cost: float, tokens_per_second: float) -> float:
    if hourly_cost < 0:
        raise ValueError("hourly_cost must be non-negative")
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return hourly_cost * 1_000_000 / (tokens_per_second * 3600)


def _peak_rss_mb() -> float:
    # Linux ru_maxrss is KiB. The project runner is Linux-only.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark_checkpoint(
    checkpoint: Path,
    *,
    device: str = "cpu",
    hourly_cost: float = 0.0,
    batch_size: int = 8,
    training_steps: int = 50,
    decode_tokens: int = 128,
) -> dict[str, object]:
    if batch_size <= 0 or training_steps <= 0 or decode_tokens <= 0:
        raise ValueError("batch_size, training_steps, and decode_tokens must be positive")

    torch.manual_seed(2026)
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    context = model.config.context_length
    x = torch.randint(0, model.config.vocab_size, (batch_size, context), device=device)
    y = torch.randint(0, model.config.vocab_size, (batch_size, context), device=device)

    model.train()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    training_tokens = batch_size * context * training_steps
    training_tps = training_tokens / training_seconds

    torch.manual_seed(2026)
    decode_model, decode_payload = load_model(checkpoint, device)
    decode_tokenizer = tokenizer_from_payload(decode_payload)
    decode_model.eval()
    prompt_ids = decode_tokenizer.encode("The ")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        decode_model.generate(prompt, 8, temperature=0.8, top_k=40)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        decode_model.generate(prompt, decode_tokens, temperature=0.8, top_k=40)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - start
    decode_tps = decode_tokens / decode_seconds

    result: dict[str, object] = {
        "format_version": "1.0",
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "parameter_count": decode_model.parameter_count(),
        "estimated_active_flops_per_token": decode_model.estimated_active_flops_per_token(),
        "device": device,
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "training": {
            "batch_size": batch_size,
            "context_length": context,
            "measured_steps": training_steps,
            "tokens": training_tokens,
            "seconds": training_seconds,
            "tokens_per_second": training_tps,
            "estimated_cost_per_million_tokens": cost_per_million_tokens(hourly_cost, training_tps),
        },
        "decode": {
            "generated_tokens": decode_tokens,
            "seconds": decode_seconds,
            "tokens_per_second": decode_tps,
            "estimated_cost_per_million_tokens": cost_per_million_tokens(hourly_cost, decode_tps),
        },
        "peak_process_rss_mb": _peak_rss_mb(),
        "hourly_compute_cost_assumption": hourly_cost,
        "cost_formula": "hourly_cost * 1_000_000 / (tokens_per_second * 3600)",
        "cost_scope": "Cash compute only; excludes electricity, hardware ownership, networking, storage, and labor.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Genesis AI training and decode throughput.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hourly-cost", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=50)
    parser.add_argument("--decode-tokens", type=int, default=128)
    args = parser.parse_args()
    result = benchmark_checkpoint(
        args.checkpoint,
        device=args.device,
        hourly_cost=args.hourly_cost,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        decode_tokens=args.decode_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
