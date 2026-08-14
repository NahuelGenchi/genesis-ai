from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .capability_index import compare_results
from .ingest import sha256_file
from .multidomain_curriculum import CURRICULUM_VERSION
from .multidomain_training import TRAINING_POLICY_VERSION

GATE_VERSION = "m6-multidomain-promotion-v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def gate(
    *,
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    training_path: str | Path,
    reproduction_path: str | Path,
    curriculum_path: str | Path,
    baseline_domain_path: str | Path,
    candidate_domain_path: str | Path,
    baseline_m3_path: str | Path,
    candidate_m3_path: str | Path,
) -> dict[str, Any]:
    baseline_checkpoint = Path(baseline_checkpoint)
    candidate_checkpoint = Path(candidate_checkpoint)
    training = _load(training_path)
    reproduction = _load(reproduction_path)
    curriculum = _load(curriculum_path)
    baseline_domain = _load(baseline_domain_path)
    candidate_domain = _load(candidate_domain_path)
    baseline_m3 = _load(baseline_m3_path)
    candidate_m3 = _load(candidate_m3_path)

    if training.get("training_policy") != TRAINING_POLICY_VERSION:
        raise ValueError("unexpected multi-domain training policy")
    if curriculum.get("curriculum_version") != CURRICULUM_VERSION:
        raise ValueError("unexpected multi-domain curriculum")
    baseline_sha = sha256_file(baseline_checkpoint)
    candidate_sha = sha256_file(candidate_checkpoint)
    if training.get("parent_checkpoint_sha256") != baseline_sha:
        raise ValueError("candidate was not continued from evaluated incumbent")
    if training.get("inference_checkpoint_sha256") != candidate_sha:
        raise ValueError("training result does not bind candidate checkpoint")

    gci = compare_results(baseline_domain, candidate_domain)
    baseline_score = float(gci["baseline"]["score"])
    candidate_score = float(gci["candidate"]["score"])
    relative_gci = gci["relative_percent_change"]
    if relative_gci is None:
        raise ValueError("multi-domain gate requires a non-zero incumbent GCI")

    domain_scores = {
        domain: float(candidate_domain["domains"][domain]["exact_accuracy"])
        for domain in ("code", "math", "structured")
    }
    baseline_m3_loss = float(baseline_m3["evaluation"]["loss"])
    candidate_m3_loss = float(candidate_m3["evaluation"]["loss"])
    m3_regression = candidate_m3_loss / baseline_m3_loss - 1.0
    overlap = int(curriculum.get("evaluation", {}).get("exact_prompt_overlap_count", -1))

    gates = [
        {
            "name": "gci_v1_relative_gain",
            "passed": float(relative_gci) >= 100.0,
            "requirement": {"min_relative_percent": 100.0},
            "observed": {
                "baseline": baseline_score,
                "candidate": candidate_score,
                "absolute_points": gci["absolute_point_change"],
                "relative_percent": relative_gci,
            },
        },
        {"name": "code_exact", "passed": domain_scores["code"] >= 0.90, "requirement": 0.90, "observed": domain_scores["code"]},
        {"name": "math_exact", "passed": domain_scores["math"] >= 0.50, "requirement": 0.50, "observed": domain_scores["math"]},
        {"name": "structured_exact", "passed": domain_scores["structured"] >= 0.50, "requirement": 0.50, "observed": domain_scores["structured"]},
        {
            "name": "m3_validation_loss",
            "passed": m3_regression <= 0.02,
            "requirement": {"max_regression_fraction": 0.02},
            "observed": {"baseline": baseline_m3_loss, "candidate": candidate_m3_loss, "regression_fraction": m3_regression},
        },
        {
            "name": "reproducibility",
            "passed": reproduction.get("reproducible") is True and reproduction.get("weights_equal") is True,
            "requirement": {"semantic_weights_equal": True},
            "observed": reproduction,
        },
        {"name": "holdout_overlap", "passed": overlap == 0, "requirement": 0, "observed": overlap},
        {
            "name": "zero_cash_compute",
            "passed": training.get("cash_compute_cost_usd") == 0.0 and curriculum.get("cash_compute_cost_usd") == 0.0,
            "requirement": 0.0,
            "observed": {"training": training.get("cash_compute_cost_usd"), "curriculum": curriculum.get("cash_compute_cost_usd")},
        },
    ]
    promoted = all(bool(item["passed"]) for item in gates)
    return {
        "format_version": "1.0",
        "gate_version": GATE_VERSION,
        "decision": "promote" if promoted else "reject",
        "promoted": promoted,
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "gci_v1": gci,
        "domain_exact_accuracy": domain_scores,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the M6 multi-domain promotion gate.")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--candidate-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = gate(
        baseline_checkpoint=args.baseline_checkpoint,
        candidate_checkpoint=args.candidate,
        training_path=args.training,
        reproduction_path=args.reproduction,
        curriculum_path=args.curriculum,
        baseline_domain_path=args.baseline_domain,
        candidate_domain_path=args.candidate_domain,
        baseline_m3_path=args.baseline_m3,
        candidate_m3_path=args.candidate_m3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    raise SystemExit(0 if result["promoted"] else 2)


if __name__ == "__main__":
    main()
