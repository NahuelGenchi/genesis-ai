from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .ingest import sha256_file
from .scale_repro import REPRO_VERSION
from .scale_training import TRAINING_POLICY_VERSION

GATE_VERSION = "m6-scale-promotion-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _gate(name: str, passed: bool, observed: object, requirement: object) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed, "requirement": requirement}


def decide_scale_promotion(
    *,
    candidate_checkpoint: str | Path,
    training_run_path: str | Path,
    reproduction_path: str | Path,
    baseline_domain_path: str | Path,
    candidate_domain_path: str | Path,
    baseline_m3_path: str | Path,
    candidate_m3_path: str | Path,
    ladder_result_path: str | Path,
    curriculum_lock_path: str | Path,
    expected_training_policy: str = TRAINING_POLICY_VERSION,
    required_alignment_policy: str | None = None,
) -> dict[str, Any]:
    candidate_checkpoint = Path(candidate_checkpoint)
    training_run_path = Path(training_run_path)
    reproduction_path = Path(reproduction_path)
    baseline_domain_path = Path(baseline_domain_path)
    candidate_domain_path = Path(candidate_domain_path)
    baseline_m3_path = Path(baseline_m3_path)
    candidate_m3_path = Path(candidate_m3_path)
    ladder_result_path = Path(ladder_result_path)
    curriculum_lock_path = Path(curriculum_lock_path)
    if not expected_training_policy:
        raise ValueError("expected_training_policy must be non-empty")

    training = _load_json(training_run_path)
    reproduction = _load_json(reproduction_path)
    baseline_domain = _load_json(baseline_domain_path)
    candidate_domain = _load_json(candidate_domain_path)
    baseline_m3 = _load_json(baseline_m3_path)
    candidate_m3 = _load_json(candidate_m3_path)
    ladder = _load_json(ladder_result_path)
    curriculum = _load_json(curriculum_lock_path)

    checkpoint_hash = sha256_file(candidate_checkpoint)
    if training.get("training_policy") != expected_training_policy:
        raise ValueError("unsupported M6 training policy")
    if training.get("inference_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("training record does not match candidate checkpoint")
    if reproduction.get("repro_version") != REPRO_VERSION or reproduction.get("primary_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("reproduction record does not match candidate checkpoint")
    for result, name in ((baseline_domain, "baseline domain"), (candidate_domain, "candidate domain")):
        if result.get("suite_version") != "m6-domain-selection-v1":
            raise ValueError(f"{name} uses unexpected domain suite")
    if baseline_domain.get("suite_sha256") != candidate_domain.get("suite_sha256"):
        raise ValueError("domain evaluation suites differ")
    if candidate_domain.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("candidate domain result does not match checkpoint")
    for field in ("suite_version", "suite_sha256", "data_manifest_sha256"):
        if baseline_m3.get(field) != candidate_m3.get(field):
            raise ValueError(f"M3 comparison mismatch: {field}")
    if candidate_m3.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("candidate M3 result does not match checkpoint")
    if ladder.get("next_stage") != "micro-2m":
        raise ValueError("micro-2m is not authorized by measured ladder")
    promotion = ladder.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError("scaling promotion policy missing")
    if curriculum.get("curriculum_version") != "m6-code-curriculum-v1":
        raise ValueError("unexpected curriculum lock")

    baseline_code = baseline_domain.get("domains", {}).get("code")
    candidate_code = candidate_domain.get("domains", {}).get("code")
    if not isinstance(baseline_code, dict) or not isinstance(candidate_code, dict):
        raise ValueError("code-domain metrics missing")
    baseline_accuracy = float(baseline_code["exact_accuracy"])
    candidate_accuracy = float(candidate_code["exact_accuracy"])
    accuracy_gain = candidate_accuracy - baseline_accuracy

    baseline_metric = baseline_m3.get("primary_metric")
    candidate_metric = candidate_m3.get("primary_metric")
    if not isinstance(baseline_metric, dict) or not isinstance(candidate_metric, dict):
        raise ValueError("M3 primary metric missing")
    if baseline_metric.get("name") != "validation_loss" or candidate_metric.get("name") != "validation_loss":
        raise ValueError("M6 scale gate requires M3 validation_loss")
    baseline_loss = float(baseline_metric["value"])
    candidate_loss = float(candidate_metric["value"])
    if not all(math.isfinite(value) and value > 0 for value in (baseline_loss, candidate_loss)):
        raise ValueError("invalid M3 validation loss")
    m3_regression = (candidate_loss - baseline_loss) / baseline_loss

    contamination = candidate_m3.get("contamination")
    if not isinstance(contamination, dict):
        raise ValueError("candidate contamination result missing")
    zero_contamination = contamination.get("blocking") is False and int(contamination.get("exact_overlap_count", -1)) == 0
    separation = curriculum.get("evaluation_separation")
    zero_domain_overlap = isinstance(separation, dict) and int(separation.get("exact_prompt_overlap_count", -1)) == 0

    expected_parameter_count = 1_895_808
    parameter_count = training.get("parameter_count")
    exact_mix = (
        math.isclose(float(training.get("procedural_step_fraction", -1)), 0.8, abs_tol=1e-12)
        and math.isclose(float(training.get("public_step_fraction", -1)), 0.2, abs_tol=1e-12)
    )
    target_tokens = int(training.get("target_training_tokens", 0))
    processed_tokens = int(training.get("processed_tokens", 0))
    zero_cash = training.get("cash_compute_cost_usd") == 0.0 and curriculum.get("cash_compute_cost_usd") == 0.0
    reproducible = reproduction.get("reproducible") is True

    min_gain = float(promotion["min_domain_accuracy_absolute_gain"])
    max_m3_regression = float(promotion["max_m3_validation_loss_regression_fraction"])
    gates = [
        _gate("code_exact_accuracy", accuracy_gain >= min_gain, {"baseline": baseline_accuracy, "candidate": candidate_accuracy, "absolute_gain": accuracy_gain}, {"min_absolute_gain": min_gain}),
        _gate("m3_validation_loss", m3_regression <= max_m3_regression, {"baseline": baseline_loss, "candidate": candidate_loss, "regression_fraction": m3_regression}, {"max_regression_fraction": max_m3_regression}),
        _gate("m3_exact_contamination", zero_contamination if promotion.get("require_zero_exact_contamination") is True else True, contamination, {"zero_overlap": bool(promotion.get("require_zero_exact_contamination"))}),
        _gate("domain_holdout_overlap", zero_domain_overlap, {"exact_prompt_overlap_count": separation.get("exact_prompt_overlap_count") if isinstance(separation, dict) else None}, {"exact_prompt_overlap_count": 0}),
        _gate("parameter_count", parameter_count == expected_parameter_count, parameter_count, expected_parameter_count),
        _gate("training_budget", target_tokens == 2_000_000 and processed_tokens >= target_tokens, {"target": target_tokens, "processed": processed_tokens}, {"target": 2_000_000, "processed_at_least_target": True}),
        _gate("training_mix", exact_mix, {"procedural": training.get("procedural_step_fraction"), "public": training.get("public_step_fraction")}, {"procedural": 0.8, "public": 0.2}),
        _gate("reproducibility", reproducible if promotion.get("require_reproducible_checkpoint") is True else True, reproduction, {"required": bool(promotion.get("require_reproducible_checkpoint"))}),
        _gate("zero_cash_compute", zero_cash if promotion.get("require_zero_cash_compute") is True else True, {"training": training.get("cash_compute_cost_usd"), "curriculum": curriculum.get("cash_compute_cost_usd")}, {"required": bool(promotion.get("require_zero_cash_compute"))}),
    ]
    if required_alignment_policy is not None:
        alignment = training.get("alignment")
        observed_alignment = alignment.get("policy") if isinstance(alignment, dict) else None
        target_position = alignment.get("target_model_position") if isinstance(alignment, dict) else None
        gates.append(
            _gate(
                "procedural_generation_alignment",
                observed_alignment == required_alignment_policy and target_position == 127,
                {"policy": observed_alignment, "target_model_position": target_position},
                {"policy": required_alignment_policy, "target_model_position": 127},
            )
        )
    promoted = all(gate["passed"] for gate in gates)
    result: dict[str, Any] = {
        "format_version": "1.0",
        "gate_version": GATE_VERSION,
        "candidate_checkpoint_sha256": checkpoint_hash,
        "expected_training_policy": expected_training_policy,
        "required_alignment_policy": required_alignment_policy,
        "inputs": {
            "training_run_sha256": sha256_file(training_run_path),
            "reproduction_sha256": sha256_file(reproduction_path),
            "baseline_domain_sha256": sha256_file(baseline_domain_path),
            "candidate_domain_sha256": sha256_file(candidate_domain_path),
            "baseline_m3_sha256": sha256_file(baseline_m3_path),
            "candidate_m3_sha256": sha256_file(candidate_m3_path),
            "ladder_result_sha256": sha256_file(ladder_result_path),
            "curriculum_lock_sha256": sha256_file(curriculum_lock_path),
        },
        "gates": gates,
        "promoted": promoted,
        "decision": "promote" if promoted else "reject",
        "primary_capability": {
            "metric": "code_exact_accuracy",
            "baseline": baseline_accuracy,
            "candidate": candidate_accuracy,
            "absolute_gain": accuracy_gain,
        },
    }
    result["decision_sha256"] = hashlib.sha256(_canonical(result).encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the M6 scale-promotion gate to micro-2m.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--candidate-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--ladder-result", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide_scale_promotion(
        candidate_checkpoint=args.candidate,
        training_run_path=args.training_run,
        reproduction_path=args.reproduction,
        baseline_domain_path=args.baseline_domain,
        candidate_domain_path=args.candidate_domain,
        baseline_m3_path=args.baseline_m3,
        candidate_m3_path=args.candidate_m3,
        ladder_result_path=args.ladder_result,
        curriculum_lock_path=args.curriculum_lock,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if not result["promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
