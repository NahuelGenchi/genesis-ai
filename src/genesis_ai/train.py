from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .checkpoint import export_inference_checkpoint, load_model, restore_training_state, save_checkpoint, tokenizer_from_payload
from .config import ModelConfig
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Genesis AI from random weights or resume exactly")
    parser.add_argument("--data", required=True, help="Filtered corpus directory")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=100, help="Target total step")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="runs/training/latest.pt")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--run-metadata", default="runs/training/run.json")
    parser.add_argument("--export")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=384)
    return parser.parse_args()


def _fixed_loss(model: GenesisLM, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, loss = model(x.to(device), y.to(device))
    if was_training:
        model.train()
    assert loss is not None
    return float(loss.detach().cpu())


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.checkpoint_every <= 0:
        raise ValueError("steps, batch-size, and checkpoint-every must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    file_tokenizer = ByteBPETokenizer.load(Path(args.tokenizer))
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(args.seed + 1)
    resumed_from_step = 0
    previous_elapsed = 0.0

    if args.resume:
        model, payload = load_model(args.resume, args.device)
        tokenizer = tokenizer_from_payload(payload)
        if tokenizer.merges != file_tokenizer.merges:
            raise ValueError("resume tokenizer does not match --tokenizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        restore_training_state(payload, optimizer, batch_generator)
        resumed_from_step = int(payload["step"])
        if args.steps <= resumed_from_step:
            raise ValueError("--steps must exceed resumed checkpoint step")
        metadata = dict(payload.get("metadata", {}))
        previous_elapsed = float(metadata.get("elapsed_seconds", 0.0))
    else:
        tokenizer = file_tokenizer
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=0.0,
        )
        model = GenesisLM(config).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        metadata = {}

    dataset = TokenDataset(args.data, tokenizer, model.config.context_length, split="train")
    probe_generator = torch.Generator(device="cpu")
    probe_generator.manual_seed(args.seed + 1_000_003)
    probe_x, probe_y = sample_batch(dataset, args.batch_size, probe_generator)
    probe_loss_before = float(metadata.get("probe_loss_before", _fixed_loss(model, probe_x, probe_y, args.device)))

    started = time.perf_counter()
    model.train()
    last_loss = float("nan")
    for step in range(resumed_from_step + 1, args.steps + 1):
        x, y = sample_batch(dataset, args.batch_size, batch_generator)
        x, y = x.to(args.device), y.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step == resumed_from_step + 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step} loss={last_loss:.4f}")
        if step % args.checkpoint_every == 0 and step != args.steps:
            partial_metadata = dict(metadata)
            partial_metadata.update({
                "seed": args.seed,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "probe_loss_before": probe_loss_before,
                "elapsed_seconds": previous_elapsed + (time.perf_counter() - started),
            })
            save_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                metadata=partial_metadata,
                tokenizer=tokenizer,
                batch_generator=batch_generator,
            )

    elapsed = previous_elapsed + (time.perf_counter() - started)
    probe_loss_after = _fixed_loss(model, probe_x, probe_y, args.device)
    run_metadata = {
        "seed": args.seed,
        "steps": args.steps,
        "resumed_from_step": resumed_from_step,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
        "parameter_count": model.parameter_count(),
        "estimated_active_flops_per_token": model.estimated_active_flops_per_token(),
        "elapsed_seconds": elapsed,
        "last_training_loss": last_loss,
        "probe_loss_before": probe_loss_before,
        "probe_loss_after": probe_loss_after,
        "probe_loss_decreased": probe_loss_after < probe_loss_before,
        "train_documents": dataset.document_count,
        "train_tokens": dataset.token_count,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": sha256_file(Path(args.tokenizer)),
        "data_manifest_sha256": sha256_file(Path(args.data) / "manifest.json"),
        "config": model.config.to_dict(),
    }
    save_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        step=args.steps,
        metadata=run_metadata,
        tokenizer=tokenizer,
        batch_generator=batch_generator,
    )
    if args.export:
        export_inference_checkpoint(args.checkpoint, args.export)

    run_path = Path(args.run_metadata)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(run_metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
