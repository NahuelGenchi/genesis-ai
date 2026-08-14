import argparse
import json
import math

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_model
from .data import ByteDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Genesis AI checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, payload = load_model(args.checkpoint, args.device)
    dataset = ByteDataset(args.data, model.config.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= args.batches:
                break
            _, loss = model(x.to(args.device), y.to(args.device))
            assert loss is not None
            losses.append(float(loss))
    if not losses:
        raise RuntimeError("no evaluation batches")
    mean_loss = sum(losses) / len(losses)
    result = {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "batches": len(losses),
        "checkpoint_step": payload["step"],
        "parameter_count": model.parameter_count(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
