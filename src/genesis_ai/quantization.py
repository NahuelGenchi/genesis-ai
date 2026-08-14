from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_model, tokenizer_from_payload
from .data import TokenDataset
from .ingest import sha256_file
from .tokenizer import ByteBPETokenizer


def dynamic_int8(model: nn.Module) -> nn.Module:
    """Return a CPU dynamic-int8 copy with Linear layers quantized."""
    model = model.cpu().eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
    return quantized


def quantized_linear_count(model: nn.Module) -> int:
    return sum(
        module.__class__.__name__ == "Linear" and "quantized" in module.__class__.__module__
        for module in model.modules()
    )


def serialized_state_dict_bytes(model: nn.Module) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size


def validation_loss(model: nn.Module, dataset: TokenDataset, batches: int = 20) -> tuple[float, int]:
    if batches <= 0:
        raise ValueError("batches must be positive")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, (x, y) in enumerate(loader):
            if index >= batches:
                break
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("no validation batches")
    return sum(losses) / len(losses), len(losses)


def decode_benchmark(
    model: nn.Module,
    tokenizer: ByteBPETokenizer,
    *,
    generated_tokens: int = 256,
    repeats: int = 3,
    seed: int = 9001,
) -> dict[str, object]:
    if generated_tokens <= 0 or repeats <= 0:
        raise ValueError("generated_tokens and repeats must be positive")
    prompt_ids = tokenizer.encode("The ")
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    model.eval()
    throughputs: list[float] = []
    final_ids: list[int] = []

    with torch.no_grad():
        torch.manual_seed(seed)
        model.generate(prompt.clone(), 16, temperature=0.8, top_k=40)
        for repeat in range(repeats):
            torch.manual_seed(seed + repeat)
            started = time.perf_counter()
            output = model.generate(prompt.clone(), generated_tokens, temperature=0.8, top_k=40)
            elapsed = time.perf_counter() - started
            throughputs.append(generated_tokens / elapsed)
            if repeat == 0:
                final_ids = output[0].tolist()

    text = tokenizer.decode(final_ids, errors="replace")
    return {
        "generated_tokens": generated_tokens,
        "repeats": repeats,
        "tokens_per_second_samples": throughputs,
        "median_tokens_per_second": statistics.median(throughputs),
        "sample_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def benchmark_low_bit(
    checkpoint: Path,
    data_dir: Path,
    *,
    validation_batches: int = 20,
    decode_tokens: int = 256,
    decode_repeats: int = 3,
) -> dict[str, object]:
    fp32_model, payload = load_model(checkpoint, "cpu")
    tokenizer = tokenizer_from_payload(payload)
    dataset = TokenDataset(data_dir, tokenizer, fp32_model.config.context_length, split="validation")

    fp32_loss, evaluated_batches = validation_loss(fp32_model, dataset, validation_batches)
    fp32_state_bytes = serialized_state_dict_bytes(fp32_model)
    fp32_decode = decode_benchmark(
        fp32_model,
        tokenizer,
        generated_tokens=decode_tokens,
        repeats=decode_repeats,
    )

    int8_model = dynamic_int8(fp32_model)
    quantized_linears = quantized_linear_count(int8_model)
    if quantized_linears <= 0:
        raise RuntimeError("dynamic int8 conversion produced no quantized Linear modules")
    int8_loss, int8_batches = validation_loss(int8_model, dataset, validation_batches)
    int8_state_bytes = serialized_state_dict_bytes(int8_model)
    int8_decode = decode_benchmark(
        int8_model,
        tokenizer,
        generated_tokens=decode_tokens,
        repeats=decode_repeats,
    )

    fp32_tps = float(fp32_decode["median_tokens_per_second"])
    int8_tps = float(int8_decode["median_tokens_per_second"])
    result: dict[str, object] = {
        "format_version": "1.0",
        "method": "pytorch-dynamic-int8-linear",
        "torch_version": torch.__version__,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "validation_documents": dataset.document_count,
        "validation_tokens": dataset.token_count,
        "validation_batches": min(evaluated_batches, int8_batches),
        "fp32": {
            "validation_loss": fp32_loss,
            "perplexity": math.exp(min(fp32_loss, 20)),
            "serialized_state_dict_bytes": fp32_state_bytes,
            "decode": fp32_decode,
        },
        "int8": {
            "validation_loss": int8_loss,
            "perplexity": math.exp(min(int8_loss, 20)),
            "serialized_state_dict_bytes": int8_state_bytes,
            "quantized_linear_modules": quantized_linears,
            "decode": int8_decode,
        },
        "tradeoff": {
            "validation_loss_delta": int8_loss - fp32_loss,
            "validation_loss_regression_percent": ((int8_loss / fp32_loss) - 1.0) * 100.0,
            "state_dict_size_ratio": int8_state_bytes / fp32_state_bytes,
            "state_dict_size_reduction_percent": (1.0 - int8_state_bytes / fp32_state_bytes) * 100.0,
            "decode_speedup": int8_tps / fp32_tps,
            "decode_speed_change_percent": ((int8_tps / fp32_tps) - 1.0) * 100.0,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs dynamic INT8 inference on the frozen Genesis baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--decode-repeats", type=int, default=3)
    args = parser.parse_args()
    result = benchmark_low_bit(
        args.checkpoint,
        args.data,
        validation_batches=args.validation_batches,
        decode_tokens=args.decode_tokens,
        decode_repeats=args.decode_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
