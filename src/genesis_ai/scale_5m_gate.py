from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .capability_index import compare_results
from .ingest import sha256_file
from .scale_5m_contract import load_scale_contract
from .scale_5m_curriculum import CURRICULUM_VERSION
from .scale_5m_training import TRAINING_POLICY_VERSION

GATE_VERSION = "m6-scale-5m-rope-promotion-gate-v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def evaluate_gate(
    *,
    experiment_path: str | Path,
    finalist_path: str | Path,
    preflight_path: str | Path,
    curriculum_path: str | Path,
    primary_training_path: str | Path,
    replica_training_path: str | Path,
    reproduction_path: str | Path,
    baseline_domain_path: str | Path,
    candidate_domain_path: str | Path,
    baseline_m3_path: str | Path,
    candidate_m3_path: str | Path,
    candidate_checkpoint_path: str | Path,
    baseline_ladder_path: str | Path,
    candidate_ladder_path: str | Path,
) -> dict[str, Any]:
    experiment, _, _, config = load_scale_contract(
        experiment_path=experiment_path,
        finalist_path=finalist_path,
        preflight_path=preflight_path,
    )
    curriculum = _load(curriculum_path)
    primary_training = _load(primary_training_path)
    replica_training = _load(replica_training_path)
    reproduction = _load(reproduction_path)
    baseline_domain = _load(baseline_domain_path)
    candidate_domain = _load(candidate_domain_path)
    baseline_m3 = _load(baseline_m3_path)
    candidate_m3 = _load(candidate_m3_path)
    baseline_ladder = _load(baseline_ladder_path)
    candidate_ladder = _load(candidate_ladder_path)

    if curriculum.get("curriculum_version") != CURRICULUM_VERSION or curriculum.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("invalid ~5M curriculum at promotion boundary")
    if curriculum.get("ladder_separation", {}).get("exact_training_prompt_overlap_count") != 0:
        raise ValueError("~5M training overlaps frozen ladder prompts")
    for training in (primary_training, replica_training):
        if training.get("training_policy") != TRAINING_POLICY_VERSION or training.get("cash_compute_cost_usd") != 0.0:
            raise ValueError("invalid ~5M training record at promotion boundary")
        if training.get("parameter_count") != 4_954_624 or training.get("config") != config.to_dict():
            raise ValueError("~5M training architecture drifted")
        if training.get("curriculum_sha256") != sha256_file(curriculum_path):
            raise ValueError("~5M training/curriculum binding drifted")
    if primary_training.get("schedule_sha256") != replica_training.get("schedule_sha256"):
        raise ValueError("primary/replica used different ~5M schedules")
    if primary_training.get("processed_tokens") != replica_training.get("processed_tokens"):
        raise ValueError("primary/replica processed-token accounting differs")

    candidate_sha = sha256_file(candidate_checkpoint_path)
    if primary_training.get("inference_checkpoint_sha256") != candidate_sha:
        raise ValueError("candidate checkpoint differs from primary training record")
    if reproduction.get("repro_version") != "m6-repro-v1":
        raise ValueError("unsupported semantic reproduction record")

    gci = compare_results(baseline_domain, candidate_domain)
    candidate_exact = gci["candidate"]["domain_exact_accuracy"]
    gate_contract = experiment.get("promotion_gate")
    if not isinstance(gate_contract, dict):
        raise ValueError("~5M promotion contract is missing")

    baseline_loss = float(baseline_m3["evaluation"]["loss"])
    candidate_loss = float(candidate_m3["evaluation"]["loss"])
    m3_regression = candidate_loss / baseline_loss - 1.0
    baseline_contamination = baseline_m3.get("contamination")
    candidate_contamination = candidate_m3.get("contamination")
    if not isinstance(baseline_contamination, dict) or not isinstance(candidate_contamination, dict):
        raise ValueError("M3 contamination records are missing")

    if baseline_ladder.get("metric_version") != "gci-ladder-v1" or candidate_ladder.get("metric_version") != "gci-ladder-v1":
        raise ValueError("~5M gate requires GCI-Ladder reports")
    if baseline_ladder.get("difficulty_suite_sha256") != candidate_ladder.get("difficulty_suite_sha256"):
        raise ValueError("~5M ladder reporting must use identical suite hashes")

    gates = [
        {
            "name": "gci_absolute_gain",
            "observed": gci["absolute_point_change"],
            "requirement": float(gate_contract["minimum_gci_absolute_gain"]),
            "passed": float(gci["absolute_point_change"]) >= float(gate_contract["minimum_gci_absolute_gain"]),
        },
        {
            "name": "code_exact_accuracy",
            "observed": candidate_exact["code"],
            "requirement": float(gate_contract["minimum_code_exact_accuracy"]),
            "passed": float(candidate_exact["code"]) >= float(gate_contract["minimum_code_exact_accuracy"]),
        },
        {
            "name": "math_exact_accuracy",
            "observed": candidate_exact["math"],
            "requirement": float(gate_contract["minimum_math_exact_accuracy"]),
            "passed": float(candidate_exact["math"]) >= float(gate_contract["minimum_math_exact_accuracy"]),
        },
        {
            "name": "structured_exact_accuracy",
            "observed": candidate_exact["structured"],
            "requirement": float(gate_contract["minimum_structured_exact_accuracy"]),
            "passed": float(candidate_exact["structured"]) >= float(gate_contract["minimum_structured_exact_accuracy"]),
        },
        {
            "name": "m3_validation_loss",
            "observed": {"baseline": baseline_loss, "candidate": candidate_loss, "regression_fraction": m3_regression},
            "requirement": {"maximum_regression_fraction": float(gate_contract["maximum_m3_loss_regression_fraction"])},
            "passed": m3_regression <= float(gate_contract["maximum_m3_loss_regression_fraction"]),
        },
        {
            "name": "m3_exact_contamination",
            "observed": {
                "baseline_exact_overlap_count": baseline_contamination.get("exact_overlap_count"),
                "candidate_exact_overlap_count": candidate_contamination.get("exact_overlap_count"),
            },
            "requirement": 0,
            "passed": baseline_contamination.get("blocking") is False
            and candidate_contamination.get("blocking") is False
            and int(baseline_contamination.get("exact_overlap_count", -1)) == 0
            and int(candidate_contamination.get("exact_overlap_count", -1)) == 0,
        },
        {
            "name": "ladder_training_prompt_overlap",
            "observed": int(curriculum["ladder_separation"]["exact_training_prompt_overlap_count"]),
            "requirement": 0,
            "passed": int(curriculum["ladder_separation"]["exact_training_prompt_overlap_count"]) == 0,
        },
        {
            "name": "semantic_reproduction",
            "observed": reproduction,
            "requirement": True,
            "passed": reproduction.get("reproducible") is True,
        },
        {
            "name": "zero_cash_compute",
            "observed": {
                "experiment": experiment.get("cash_compute_cost_usd"),
                "curriculum": curriculum.get("cash_compute_cost_usd"),
                "primary_training": primary_training.get("cash_compute_cost_usd"),
                "replica_training": replica_training.get("cash_compute_cost_usd"),
            },
            "requirement": 0.0,
            "passed": all(
                value == 0.0
                for value in (
                    experiment.get("cash_compute_cost_usd"),
                    curriculum.get("cash_compute_cost_usd"),
                    primary_training.get("cash_compute_cost_usd"),
                    replica_training.get("cash_compute_cost_usd"),
                )
            ),
        },
    ]
    promoted = all(bool(gate["passed"]) for gate in gates)
    return {
        "format_version": "1.0",
        "gate_version": GATE_VERSION,
        "decision": "promote" if promoted else "reject",
        "promoted": promoted,
        "candidate_checkpoint_sha256": candidate_sha,
        "gci_v1": gci,
        "gates": gates,
        "ladder_reporting": {
            "baseline": baseline_ladder,
            "candidate": candidate_ladder,
            "promotion_authority": False,
        },
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen first ~5M RoPE promotion gate.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--finalist", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--primary-training", type=Path, required=True)
    parser.add_argument("--replica-training", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--candidate-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-ladder", type=Path, required=True)
    parser.add_argument("--candidate-ladder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_gate(
        experiment_path=args.experiment,
        finalist_path=args.finalist,
        preflight_path=args.preflight,
        curriculum_path=args.curriculum,
        primary_training_path=args.primary_training,
        replica_training_path=args.replica_training,
        reproduction_path=args.reproduction,
        baseline_domain_path=args.baseline_domain,
        candidate_domain_path=args.candidate_domain,
        baseline_m3_path=args.baseline_m3,
        candidate_m3_path=args.candidate_m3,
        candidate_checkpoint_path=args.candidate_checkpoint,
        baseline_ladder_path=args.baseline_ladder,
        candidate_ladder_path=args.candidate_ladder,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    raise SystemExit(0 if result["promoted"] else 2)


if __name__ == "__main__":
    main()
