from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_model, tokenizer_from_payload
from .data import TokenDataset
from .ingest import sha256_file


def evaluate_checkpoint(
    checkpoint: str,
    data: str,
    *,
    batch_size: int = 8,
    batches: int = 20,
    split: str = "validation",
    device: str = "cpu",
) -> dict[str, object]:
    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_size and batches must be positive")
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    dataset = TokenDataset(data, tokenizer, model.config.context_length, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= batches:
                break
            _, loss = model(x.to(device), y.to(device))
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("no evaluation batches")
    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "batches": len(losses),
        "split": split,
        "documents": dataset.document_count,
        "tokens": dataset.token_count,
        "checkpoint_step": int(payload["step"]),
        "parameter_count": model.parameter_count(),
        "data_manifest_sha256": sha256_file(Path(data) / "manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Genesis AI checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        batch_size=args.batch_size,
        batches=args.batches,
        split=args.split,
        device=args.device,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
