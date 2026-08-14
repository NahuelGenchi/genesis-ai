from __future__ import annotations

import random
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


def save_checkpoint(
    path: str | Path,
    *,
    model: GenesisLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict | None = None,
    tokenizer: ByteBPETokenizer | None = None,
    batch_generator: torch.Generator | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng: dict[str, object] = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    if batch_generator is not None:
        rng["batch_generator"] = batch_generator.get_state()
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "training",
            "config": model.config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "metadata": metadata or {},
            "tokenizer": tokenizer.to_dict() if tokenizer is not None else None,
            "rng": rng,
        },
        path,
    )


def _load_payload(path: str | Path, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported checkpoint schema")
    return payload


def load_model(path: str | Path, device: str = "cpu") -> tuple[GenesisLM, dict]:
    payload = _load_payload(path, device)
    config = ModelConfig.from_dict(payload["config"])
    model = GenesisLM(config).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def tokenizer_from_payload(payload: dict) -> ByteBPETokenizer:
    raw = payload.get("tokenizer")
    if raw is None:
        raise ValueError("checkpoint does not contain a tokenizer")
    return ByteBPETokenizer.from_dict(raw)


def restore_training_state(
    payload: dict,
    optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
) -> None:
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("checkpoint does not contain optimizer state")
    optimizer.load_state_dict(optimizer_state)
    rng = payload.get("rng")
    if not isinstance(rng, dict):
        raise ValueError("checkpoint does not contain RNG state")
    torch_cpu = rng.get("torch_cpu")
    python_state = rng.get("python")
    batch_state = rng.get("batch_generator")
    if not isinstance(torch_cpu, torch.Tensor) or python_state is None or not isinstance(batch_state, torch.Tensor):
        raise ValueError("checkpoint RNG state is incomplete")
    torch.set_rng_state(torch_cpu.cpu())
    random.setstate(python_state)
    batch_generator.set_state(batch_state.cpu())
    cuda_state = rng.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def export_inference_checkpoint(source: str | Path, destination: str | Path) -> None:
    payload = _load_payload(source, "cpu")
    if payload.get("tokenizer") is None:
        raise ValueError("training checkpoint does not contain a tokenizer")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "inference",
            "config": payload["config"],
            "model": payload["model"],
            "step": payload["step"],
            "metadata": payload.get("metadata", {}),
            "tokenizer": payload["tokenizer"],
        },
        destination,
    )
