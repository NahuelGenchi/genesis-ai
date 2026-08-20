from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .ingest import sha256_file

ADOPTION_VERSION = "weak-domain-adoption-v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _gci(eval_result: dict[str, Any]) -> float:
    domains = eval_result.get("domains")
    if not isinstance(domains, dict) or set(domains) != {"code", "math", "structured"}:
        raise ValueError("candidate evaluation must contain exact code/math/structured domains")
    values = []
    for domain in ("code", "math", "structured"):
        block = domains[domain]
        if not isinstance(block, dict) or not isinstance(block.get("exact_accuracy"), (int, float)):
            raise ValueError(f"candidate evaluation is missing exact_accuracy: {domain}")
        values.append(float(block["exact_accuracy"]))
    return sum(values) / len(values) * 100.0


def apply_adoption(
    *,
    state_path: str | Path,
    baseline_sha256: str,
    gate_path: str | Path,
    candidate_evaluation_path: str | Path,
    candidate_checkpoint: str | Path,
    destination_checkpoint: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    state_path = Path(state_path)
    candidate_checkpoint = Path(candidate_checkpoint)
    destination_checkpoint = Path(destination_checkpoint)
    output_path = Path(output_path)
    state = _load(state_path)
    gate = _load(gate_path)
    candidate_evaluation = _load(candidate_evaluation_path)

    if state.get("state_version") != "autonomous-state-v1" or state.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("autonomous state violates adoption contract")
    current_checkpoint = Path(str(state["incumbent_checkpoint"]))
    if not current_checkpoint.is_file():
        raise ValueError(f"current incumbent checkpoint is missing: {current_checkpoint}")
    current_sha = sha256_file(current_checkpoint)
    incumbent_unchanged = current_sha == baseline_sha256

    gate_promoted = gate.get("promoted") is True and gate.get("decision") == "promote"
    candidate_sha = sha256_file(candidate_checkpoint)
    if gate.get("candidate_checkpoint_sha256") != candidate_sha:
        raise ValueError("gate candidate hash does not match adoption checkpoint")
    if gate.get("baseline_checkpoint_sha256") != baseline_sha256:
        raise ValueError("gate baseline hash differs from training-start incumbent")

    adopted = gate_promoted and incumbent_unchanged
    if adopted:
        destination_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_checkpoint, destination_checkpoint)
        state["incumbent_checkpoint"] = destination_checkpoint.as_posix()
        state["incumbent_gci_v1"] = _gci(candidate_evaluation)
        state["autonomy_status"] = "running"
        state["circuit_breaker"] = {
            "active": False,
            "reason": "new incumbent promoted by weak-domain successive-halving funnel",
            "previous_incumbent_sha256": baseline_sha256,
            "new_incumbent_sha256": candidate_sha,
        }
        state["research_resume"] = {
            "reason": "new-incumbent-resets-same-incumbent-strategy-exhaustion",
            "source": "weak-domain-successive-halving-v1",
            "candidate_checkpoint_sha256": candidate_sha,
        }
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    adoption = {
        "format_version": "1.0",
        "adoption_version": ADOPTION_VERSION,
        "gate_promoted": gate_promoted,
        "baseline_checkpoint_sha256": baseline_sha256,
        "current_incumbent_checkpoint_sha256": current_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "incumbent_unchanged_since_training_started": incumbent_unchanged,
        "adopted": adopted,
        "reason": (
            "immutable gate passed and incumbent remained unchanged"
            if adopted
            else "immutable capability gate rejected candidate"
            if not gate_promoted
            else "incumbent changed during experiment; adoption failed closed"
        ),
        "cash_compute_cost_usd": 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(adoption, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return adoption


def main() -> None:
    parser = argparse.ArgumentParser(description="Adopt a weak-domain winner only after immutable gate + incumbent freshness checks.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = apply_adoption(
        state_path=args.state,
        baseline_sha256=args.baseline_sha256,
        gate_path=args.gate,
        candidate_evaluation_path=args.candidate_evaluation,
        candidate_checkpoint=args.candidate,
        destination_checkpoint=args.destination,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
