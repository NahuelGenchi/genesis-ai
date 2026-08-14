from pathlib import Path

import torch

from .config import ModelConfig
from .model import GenesisLM

SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: GenesisLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "config": model.config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metadata": metadata or {},
        },
        path,
    )


def load_model(path: str | Path, device: str = "cpu") -> tuple[GenesisLM, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    config = ModelConfig.from_dict(payload["config"])
    model = GenesisLM(config).to(device)
    model.load_state_dict(payload["model"])
    return model, payload
