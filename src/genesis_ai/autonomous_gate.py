from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomous_curriculum import CURRICULUM_VERSION
from .autonomous_training import TRAINING_POLICY_VERSION
from .capability_index import compare_results
from .improvement_controller import CONTROLLER_VERSION
from .ingest import sha256_file

GATE_VERSION = "autonomous-promotion-gate-v1"
DOMAINS = ("code", "math", "structured")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("controller_version") != CONTROLLER_VERSION:
        raise ValueError("unsupported controller plan")
    expected = plan.get("plan_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("plan hash missing")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    if _sha256_object(unhashed) != expected:
        raise ValueError("controller plan hash mismatch")


def gate(
    *,
    plan_path: str | Path,
    curriculum_path: str | Path,
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    training_path: str | Path,
    reproduction_path: str | Path,
    baseline_domain_path: str | Path,
    candidate_domain_path: str | Path,
    baseline_m3_path: str | Path,
    candidate_m3_path: str | Path,
) -> dict[str, Any]:
    plan = _load(plan_path)
    curriculum = _load(curriculum_path)
    training = _load(training_path)
    reproduction = _load(reproduction_path)
    baseline_domain = _load(baseline_domain_path)
    candidate_domain = _load(candidate_domain_path)
    baseline_m3 = _load(baseline_m3_path)
    candidate_m3 = _load(candidate_m3_path)
    _validate_plan(plan)

    if curriculum.get("curriculum_version") != CURRICULUM_VERSION or curriculum.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("curriculum is not bound to controller plan")
    if training.get("training_policy") != TRAINING_POLICY_VERSION or training.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("training is not bound to controller plan")

    baseline_sha = sha256_file(baseline_checkpoint)
    candidate_sha = sha256_file(candidate_checkpoint)
    if baseline_sha != plan["input"]["incumbent_checkpoint_sha256"]:
        raise ValueError("evaluated incumbent does not match controller input")
    if curriculum.get("incumbent_checkpoint_sha256") != baseline_sha or training.get("parent_checkpoint_sha256") != baseline_sha:
        raise ValueError("candidate lineage does not match incumbent")
    if training.get("inference_checkpoint_sha256") != candidate_sha:
        raise ValueError("training does not bind candidate checkpoint")
    if int(training.get("target_training_tokens", -1)) != int(plan["decision"]["target_training_tokens"]):
        raise ValueError("training budget differs from controller plan")

    gci = compare_results(baseline_domain, candidate_domain)
    if baseline_domain.get("suite_sha256") != curriculum.get("target_suite_sha256"):
        raise ValueError("incumbent was not evaluated on curriculum target suite")
    if candidate_domain.get("suite_sha256") != curriculum.get("target_suite_sha256"):
        raise ValueError("candidate was not evaluated on curriculum target suite")
    if plan["evaluation_transition"]["new_suite_required"] is not True:
        if baseline_domain.get("suite_sha256") != plan["input"]["suite_sha256"]:
            raise ValueError("repair cycle changed evaluator suite")

    focus = str(plan["decision"]["focus_domain"])
    replay = [str(domain) for domain in plan["decision"]["replay_domains"]]
    contract = plan["promotion_contract"]
    baseline_scores = {domain: float(baseline_domain["domains"][domain]["exact_accuracy"]) for domain in DOMAINS}
    candidate_scores = {domain: float(candidate_domain["domains"][domain]["exact_accuracy"]) for domain in DOMAINS}
    focus_gain = candidate_scores[focus] - baseline_scores[focus]
    nonfocus_regressions = {domain: candidate_scores[domain] - baseline_scores[domain] for domain in replay}
    baseline_m3_loss = float(baseline_m3["evaluation"]["loss"])
    candidate_m3_loss = float(candidate_m3["evaluation"]["loss"])
    m3_regression = candidate_m3_loss / baseline_m3_loss - 1.0
    contamination = candidate_m3.get("contamination", {})

    gates = [
        {
            "name": "focus_absolute_gain",
            "passed": focus_gain >= float(contract["minimum_focus_absolute_gain"]),
            "requirement": float(contract["minimum_focus_absolute_gain"]),
            "observed": focus_gain,
        },
        {
            "name": "gci_absolute_gain",
            "passed": float(gci["absolute_point_change"]) >= float(contract["minimum_gci_absolute_gain"]),
            "requirement": float(contract["minimum_gci_absolute_gain"]),
            "observed": float(gci["absolute_point_change"]),
        },
        {
            "name": "nonfocus_regressions",
            "passed": all(delta >= -float(contract["maximum_nonfocus_absolute_regression"]) for delta in nonfocus_regressions.values()),
            "requirement": {"minimum_delta_each": -float(contract["maximum_nonfocus_absolute_regression"])},
            "observed": nonfocus_regressions,
        },
        {
            "name": "m3_validation_loss",
            "passed": m3_regression <= float(contract["maximum_m3_loss_regression_fraction"]),
            "requirement": {"max_regression_fraction": float(contract["maximum_m3_loss_regression_fraction"])},
            "observed": {"baseline": baseline_m3_loss, "candidate": candidate_m3_loss, "regression_fraction": m3_regression},
        },
        {
            "name": "m3_exact_contamination",
            "passed": contamination.get("blocking") is False and int(contamination.get("exact_overlap_count", -1)) == 0,
            "requirement": {"blocking": False, "exact_overlap_count": 0},
            "observed": contamination,
        },
        {
            "name": "semantic_reproduction",
            "passed": reproduction.get("reproducible") is True and reproduction.get("weights_equal") is True,
            "requirement": True,
            "observed": reproduction,
        },
        {
            "name": "holdout_overlap",
            "passed": int(curriculum.get("exact_holdout_prompt_overlap_count", -1)) == 0,
            "requirement": 0,
            "observed": curriculum.get("exact_holdout_prompt_overlap_count"),
        },
        {
            "name": "zero_cash_compute",
            "passed": curriculum.get("cash_compute_cost_usd") == 0.0 and training.get("cash_compute_cost_usd") == 0.0,
            "requirement": 0.0,
            "observed": {"curriculum": curriculum.get("cash_compute_cost_usd"), "training": training.get("cash_compute_cost_usd")},
        },
    ]
    promoted = all(bool(item["passed"]) for item in gates)
    return {
        "format_version": "1.0",
        "gate_version": GATE_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "decision": "promote" if promoted else "reject",
        "promoted": promoted,
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "focus_domain": focus,
        "focus_absolute_gain": focus_gain,
        "nonfocus_deltas": nonfocus_regressions,
        "gci_v1": gci,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply one plan-bound autonomous Genesis promotion gate.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--candidate-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = gate(
        plan_path=args.plan,
        curriculum_path=args.curriculum,
        baseline_checkpoint=args.baseline_checkpoint,
        candidate_checkpoint=args.candidate,
        training_path=args.training,
        reproduction_path=args.reproduction,
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
