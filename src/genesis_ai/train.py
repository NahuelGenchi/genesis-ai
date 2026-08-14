import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ModelConfig
from .data import ByteDataset
from .model import GenesisLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Genesis AI from random weights")
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--run-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = ModelConfig()
    model = GenesisLM(config).to(args.device)
    dataset = ByteDataset(args.data, config.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    started = time.time()
    model.train()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(args.device), y.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step} loss={last_loss:.4f}")

    elapsed = time.time() - started
    metadata = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
        "parameter_count": model.parameter_count(),
        "elapsed_seconds": elapsed,
        "final_loss": last_loss,
    }
    save_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        metadata=metadata,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / f"run-{int(time.time())}.json"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
