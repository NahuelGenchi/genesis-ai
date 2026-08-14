from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .checkpoint import load_model
from .ingest import sha256_file

PROMOTION_POLICY_VERSION = "promotion-v1"
CANDIDATE_POLICY_VERSION = "candidate-training-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _gate(name: str, passed: bool, *, observed: object, requirement: object) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def load_policy(path: str | Path) -> dict[str, Any]:
    policy = _load_json(path)
    required = {
        "policy_version",
        "required_suite_version",
        "min_validation_improvement_fraction",
        "max_decode_throughput_regression_fraction",
        "max_training_throughput_regression_fraction",
        "max_peak_rss_regression_fraction",
        "require_zero_exact_contamination",
        "require_same_parameter_count",
        "require_experience_loss_decreased",
    }
    if set(policy) != required or policy.get("policy_version") != PROMOTION_POLICY_VERSION:
        raise ValueError("unsupported promotion policy")
    for name in (
        "min_validation_improvement_fraction",
        "max_decode_throughput_regression_fraction",
        "max_training_throughput_regression_fraction",
        "max_peak_rss_regression_fraction",
    ):
        value = policy[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid promotion threshold: {name}")
    if not isinstance(policy["required_suite_version"], str) or not policy["required_suite_version"]:
        raise ValueError("required_suite_version must be non-empty")
    for name in (
        "require_zero_exact_contamination",
        "require_same_parameter_count",
        "require_experience_loss_decreased",
    ):
        if not isinstance(policy[name], bool):
            raise ValueError(f"promotion policy flag must be boolean: {name}")
    return policy


def _validate_eval_pair(parent: dict[str, Any], candidate: dict[str, Any], policy: dict[str, Any]) -> None:
    for field in ("suite_version", "suite_sha256", "data_manifest_sha256"):
        if parent.get(field) != candidate.get(field):
            raise ValueError(f"evaluation mismatch: {field}")
    if parent.get("suite_version") != policy["required_suite_version"]:
        raise ValueError("evaluation suite does not match promotion policy")
    for result in (parent, candidate):
        primary = result.get("primary_metric")
        if not isinstance(primary, dict) or primary.get("name") != "validation_loss" or primary.get("lower_is_better") is not True:
            raise ValueError("promotion requires validation_loss as the primary metric")
        value = primary.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("invalid validation loss")
        contamination = result.get("contamination")
        if not isinstance(contamination, dict):
            raise ValueError("evaluation contamination record missing")
        overlap = contamination.get("exact_overlap_count")
        blocking = contamination.get("blocking")
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0 or not isinstance(blocking, bool):
            raise ValueError("evaluation contamination record is invalid")


def _validate_benchmark_pair(parent: dict[str, Any], candidate: dict[str, Any]) -> None:
    if parent.get("format_version") != "1.0" or candidate.get("format_version") != "1.0":
        raise ValueError("unsupported benchmark format")
    if parent.get("device") != candidate.get("device") or parent.get("hardware") != candidate.get("hardware"):
        raise ValueError("benchmarks must use identical device/hardware metadata")
    for result in (parent, candidate):
        for section in ("training", "decode"):
            raw = result.get(section)
            value = raw.get("tokens_per_second") if isinstance(raw, dict) else None
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"invalid benchmark {section} throughput")
        rss = result.get("peak_process_rss_mb")
        if not isinstance(rss, (int, float)) or isinstance(rss, bool) or not math.isfinite(float(rss)) or float(rss) <= 0:
            raise ValueError("invalid benchmark peak RSS")
        params = result.get("parameter_count")
        if not isinstance(params, int) or isinstance(params, bool) or params <= 0:
            raise ValueError("invalid benchmark parameter count")


def decide_promotion(
    *,
    parent_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    parent_evaluation: str | Path,
    candidate_evaluation: str | Path,
    parent_benchmark: str | Path,
    candidate_benchmark: str | Path,
    policy_path: str | Path,
) -> dict[str, Any]:
    parent_checkpoint = Path(parent_checkpoint)
    candidate_checkpoint = Path(candidate_checkpoint)
    parent_evaluation = Path(parent_evaluation)
    candidate_evaluation = Path(candidate_evaluation)
    parent_benchmark = Path(parent_benchmark)
    candidate_benchmark = Path(candidate_benchmark)
    policy_path = Path(policy_path)
    policy = load_policy(policy_path)
    parent_eval = _load_json(parent_evaluation)
    candidate_eval = _load_json(candidate_evaluation)
    parent_bench = _load_json(parent_benchmark)
    candidate_bench = _load_json(candidate_benchmark)

    parent_hash = sha256_file(parent_checkpoint)
    candidate_hash = sha256_file(candidate_checkpoint)
    if parent_eval.get("checkpoint_sha256") != parent_hash or parent_bench.get("checkpoint_sha256") != parent_hash:
        raise ValueError("parent records do not match parent checkpoint")
    if candidate_eval.get("checkpoint_sha256") != candidate_hash or candidate_bench.get("checkpoint_sha256") != candidate_hash:
        raise ValueError("candidate records do not match candidate checkpoint")

    _validate_eval_pair(parent_eval, candidate_eval, policy)
    _validate_benchmark_pair(parent_bench, candidate_bench)

    _, candidate_payload = load_model(candidate_checkpoint, "cpu")
    metadata = candidate_payload.get("metadata")
    self_improvement = metadata.get("self_improvement") if isinstance(metadata, dict) else None
    if (
        not isinstance(self_improvement, dict)
        or self_improvement.get("policy") != CANDIDATE_POLICY_VERSION
        or self_improvement.get("parent_checkpoint_sha256") != parent_hash
    ):
        raise ValueError("candidate checkpoint lineage does not match parent")

    parent_loss = float(parent_eval["primary_metric"]["value"])
    candidate_loss = float(candidate_eval["primary_metric"]["value"])
    validation_improvement = (parent_loss - candidate_loss) / parent_loss

    parent_decode = float(parent_bench["decode"]["tokens_per_second"])
    candidate_decode = float(candidate_bench["decode"]["tokens_per_second"])
    decode_regression = max(0.0, 1.0 - candidate_decode / parent_decode)

    parent_training = float(parent_bench["training"]["tokens_per_second"])
    candidate_training = float(candidate_bench["training"]["tokens_per_second"])
    training_regression = max(0.0, 1.0 - candidate_training / parent_training)

    parent_rss = float(parent_bench["peak_process_rss_mb"])
    candidate_rss = float(candidate_bench["peak_process_rss_mb"])
    rss_regression = max(0.0, candidate_rss / parent_rss - 1.0)

    parent_contamination = parent_eval["contamination"]
    candidate_contamination = candidate_eval["contamination"]
    contamination_clear = (
        parent_contamination["blocking"] is False
        and candidate_contamination["blocking"] is False
        and parent_contamination["exact_overlap_count"] == 0
        and candidate_contamination["exact_overlap_count"] == 0
    )
    same_parameters = parent_bench["parameter_count"] == candidate_bench["parameter_count"]
    experience_loss_decreased = self_improvement.get("experience_loss_decreased") is True

    gates = [
        _gate(
            "validation_improvement",
            validation_improvement >= float(policy["min_validation_improvement_fraction"]),
            observed=validation_improvement,
            requirement={"min_fraction": policy["min_validation_improvement_fraction"]},
        ),
        _gate(
            "exact_contamination",
            contamination_clear if policy["require_zero_exact_contamination"] else True,
            observed={
                "parent_overlap": parent_contamination["exact_overlap_count"],
                "candidate_overlap": candidate_contamination["exact_overlap_count"],
            },
            requirement={"zero_overlap": policy["require_zero_exact_contamination"]},
        ),
        _gate(
            "decode_throughput",
            decode_regression <= float(policy["max_decode_throughput_regression_fraction"]),
            observed={"regression_fraction": decode_regression, "parent_tps": parent_decode, "candidate_tps": candidate_decode},
            requirement={"max_regression_fraction": policy["max_decode_throughput_regression_fraction"]},
        ),
        _gate(
            "training_throughput",
            training_regression <= float(policy["max_training_throughput_regression_fraction"]),
            observed={"regression_fraction": training_regression, "parent_tps": parent_training, "candidate_tps": candidate_training},
            requirement={"max_regression_fraction": policy["max_training_throughput_regression_fraction"]},
        ),
        _gate(
            "peak_rss",
            rss_regression <= float(policy["max_peak_rss_regression_fraction"]),
            observed={"regression_fraction": rss_regression, "parent_mb": parent_rss, "candidate_mb": candidate_rss},
            requirement={"max_regression_fraction": policy["max_peak_rss_regression_fraction"]},
        ),
        _gate(
            "parameter_count",
            same_parameters if policy["require_same_parameter_count"] else True,
            observed={"parent": parent_bench["parameter_count"], "candidate": candidate_bench["parameter_count"]},
            requirement={"same": policy["require_same_parameter_count"]},
        ),
        _gate(
            "experience_learning",
            experience_loss_decreased if policy["require_experience_loss_decreased"] else True,
            observed={
                "before": self_improvement.get("experience_loss_before"),
                "after": self_improvement.get("experience_loss_after"),
                "decreased": self_improvement.get("experience_loss_decreased"),
            },
            requirement={"decreased": policy["require_experience_loss_decreased"]},
        ),
    ]
    promoted = all(gate["passed"] for gate in gates)
    result: dict[str, Any] = {
        "format_version": "1.0",
        "policy_version": PROMOTION_POLICY_VERSION,
        "policy_sha256": sha256_file(policy_path),
        "parent_checkpoint_sha256": parent_hash,
        "candidate_checkpoint_sha256": candidate_hash,
        "measurement_inputs": {
            "parent_evaluation_sha256": sha256_file(parent_evaluation),
            "candidate_evaluation_sha256": sha256_file(candidate_evaluation),
            "parent_benchmark_sha256": sha256_file(parent_benchmark),
            "candidate_benchmark_sha256": sha256_file(candidate_benchmark),
        },
        "suite_version": parent_eval["suite_version"],
        "suite_sha256": parent_eval["suite_sha256"],
        "data_manifest_sha256": parent_eval["data_manifest_sha256"],
        "gates": gates,
        "promoted": promoted,
        "decision": "promote" if promoted else "reject",
    }
    result["decision_sha256"] = hashlib.sha256(_canonical(result).encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the reproducible Genesis checkpoint promotion gate.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent-eval", type=Path, required=True)
    parser.add_argument("--candidate-eval", type=Path, required=True)
    parser.add_argument("--parent-benchmark", type=Path, required=True)
    parser.add_argument("--candidate-benchmark", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = decide_promotion(
        parent_checkpoint=args.parent,
        candidate_checkpoint=args.candidate,
        parent_evaluation=args.parent_eval,
        candidate_evaluation=args.candidate_eval,
        parent_benchmark=args.parent_benchmark,
        candidate_benchmark=args.candidate_benchmark,
        policy_path=args.policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if not decision["promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
