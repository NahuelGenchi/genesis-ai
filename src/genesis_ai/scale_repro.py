from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_model, tokenizer_from_payload
from .ingest import sha256_file

REPRO_VERSION = "m6-repro-v1"


def compare_checkpoints(primary: str | Path, replica: str | Path) -> dict[str, Any]:
    primary = Path(primary)
    replica = Path(replica)
    primary_model, primary_payload = load_model(primary, "cpu")
    replica_model, replica_payload = load_model(replica, "cpu")
    primary_state = primary_model.state_dict()
    replica_state = replica_model.state_dict()
    keys_equal = tuple(primary_state) == tuple(replica_state)
    weights_equal = keys_equal and all(
        torch.equal(primary_state[key], replica_state[key]) for key in primary_state
    )
    config_equal = primary_model.config.to_dict() == replica_model.config.to_dict()
    tokenizer_equal = tokenizer_from_payload(primary_payload).to_dict() == tokenizer_from_payload(replica_payload).to_dict()
    step_equal = primary_payload.get("step") == replica_payload.get("step")
    metadata_equal = primary_payload.get("metadata") == replica_payload.get("metadata")
    return {
        "format_version": "1.0",
        "repro_version": REPRO_VERSION,
        "primary_checkpoint_sha256": sha256_file(primary),
        "replica_checkpoint_sha256": sha256_file(replica),
        "checkpoint_bytes_equal": primary.read_bytes() == replica.read_bytes(),
        "state_keys_equal": keys_equal,
        "weights_equal": weights_equal,
        "config_equal": config_equal,
        "tokenizer_equal": tokenizer_equal,
        "step_equal": step_equal,
        "metadata_equal": metadata_equal,
        "reproducible": all((weights_equal, config_equal, tokenizer_equal, step_equal, metadata_equal)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two independently trained Genesis checkpoints semantically.")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_checkpoints(args.primary, args.replica)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if not result["reproducible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
