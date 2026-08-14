from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ingest import sha256_file


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_card(
    *,
    checkpoint: str | Path,
    training_path: str | Path,
    gate_path: str | Path,
    output_dir: str | Path,
) -> None:
    checkpoint = Path(checkpoint)
    training = _load(training_path)
    gate = _load(gate_path)
    if gate.get("promoted") is not True:
        raise ValueError("model card may be created only for a promoted candidate")
    if gate.get("candidate_checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("model card checkpoint does not match promotion gate")
    gci = gate["gci_v1"]
    domains = gate["domain_exact_accuracy"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "format_version": "1.0",
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": training["parameter_count"],
        "parent_checkpoint_sha256": training["parent_checkpoint_sha256"],
        "gci_v1": gci,
        "domain_exact_accuracy": domains,
        "cash_compute_cost_usd": training["cash_compute_cost_usd"],
        "training_policy": training["training_policy"],
        "processed_tokens": training["processed_tokens"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    relative = gci["relative_percent_change"]
    card = f"""# Genesis micro-2m v2

Promotion-gated multi-domain continuation checkpoint.

## Capability

- GCI-v1: **{gci['baseline']['score']:.4f} -> {gci['candidate']['score']:.4f}** ({relative:+.2f}% relative; {gci['absolute_point_change']:+.4f} points)
- Code exact: **{domains['code'] * 100:.2f}%**
- Math exact: **{domains['math'] * 100:.2f}%**
- Structured exact: **{domains['structured'] * 100:.2f}%**

## Lineage

- Independent Genesis lineage; continued only from the prior promoted Genesis checkpoint.
- No OpenAI/Anthropic weights, APIs, or model outputs were used as training targets.
- Parent checkpoint SHA-256: `{training['parent_checkpoint_sha256']}`
- Checkpoint SHA-256: `{sha256_file(checkpoint)}`
- Parameters: `{training['parameter_count']}`
- Continuation processed tokens: `{training['processed_tokens']}`
- Cash compute cost: `${training['cash_compute_cost_usd']:.2f}`

## Promotion boundary

Promotion required >=100% relative GCI improvement, >=90% code exact, >=50% math exact, >=50% structured exact, <=2% M3 loss regression, zero frozen-holdout prompt overlap, semantic reproduction, and $0 cash compute.
"""
    (output_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the promoted multi-domain Genesis model card.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_card(checkpoint=args.checkpoint, training_path=args.training, gate_path=args.gate, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
