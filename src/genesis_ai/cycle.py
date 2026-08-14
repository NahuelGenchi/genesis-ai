from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .experience import EXPERIENCE_FORMAT_VERSION, POLICY_VERSION
from .ingest import sha256_file

CYCLE_VERSION = "m5-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            count += 1
    return count


def validate_cycle_experience(
    experience_dir: str | Path,
    parent_checkpoint: str | Path,
    *,
    expected_tasks: int,
) -> dict[str, Any]:
    if expected_tasks <= 0:
        raise ValueError("expected_tasks must be positive")
    experience_dir = Path(experience_dir)
    parent_checkpoint = Path(parent_checkpoint)
    manifest_path = experience_dir / "manifest.json"
    accepted_path = experience_dir / "accepted.jsonl"
    audit_path = experience_dir / "audit.jsonl"
    manifest = _load_json(manifest_path)
    if manifest.get("format_version") != EXPERIENCE_FORMAT_VERSION:
        raise ValueError("unsupported experience manifest")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION or float(policy.get("min_score", -1.0)) != 1.0:
        raise ValueError("first cycle requires strict verified-experience-v1 score 1.0")
    producer = manifest.get("producer")
    parent_hash = sha256_file(parent_checkpoint)
    if (
        not isinstance(producer, dict)
        or producer.get("kind") != "genesis_checkpoint"
        or producer.get("checkpoint_sha256") != parent_hash
    ):
        raise ValueError("experience producer does not match cycle parent")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("experience file hashes missing")
    if files.get("accepted.jsonl") != sha256_file(accepted_path):
        raise ValueError("accepted experience hash mismatch")
    if files.get("audit.jsonl") != sha256_file(audit_path):
        raise ValueError("audit experience hash mismatch")
    accepted = _read_jsonl_count(accepted_path)
    attempted = _read_jsonl_count(audit_path)
    rejected = attempted - accepted
    if (
        attempted != expected_tasks
        or manifest.get("attempted") != attempted
        or manifest.get("accepted") != accepted
        or manifest.get("rejected") != rejected
    ):
        raise ValueError("experience counts do not match cycle task count")
    return {
        "parent_checkpoint_sha256": parent_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "accepted_sha256": files["accepted.jsonl"],
        "audit_sha256": files["audit.jsonl"],
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / attempted,
    }


def _base_record(
    *,
    parent_checkpoint: Path,
    tasks_path: Path,
    experience_dir: Path,
    task_count: int,
    min_accepted: int,
    challenge_seed: int,
    generation_seed: int,
    source_commit: str,
    workflow_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if min_accepted <= 0 or min_accepted > task_count:
        raise ValueError("min_accepted must be within task count")
    if not source_commit or not workflow_run_id:
        raise ValueError("workflow provenance is required")
    experience = validate_cycle_experience(experience_dir, parent_checkpoint, expected_tasks=task_count)
    base: dict[str, Any] = {
        "format_version": "1.0",
        "cycle_version": CYCLE_VERSION,
        "parent_checkpoint_sha256": experience["parent_checkpoint_sha256"],
        "challenge": {
            "task_count": task_count,
            "min_accepted": min_accepted,
            "difficulty": {"min": 1, "max": 1},
            "domains": ["math", "structured", "code"],
            "challenge_seed": challenge_seed,
            "generation_seed": generation_seed,
            "tasks_sha256": sha256_file(tasks_path),
        },
        "experience": {
            "policy": POLICY_VERSION,
            "min_score": 1.0,
            **{key: value for key, value in experience.items() if key != "parent_checkpoint_sha256"},
        },
        "provenance": {
            "source_commit": source_commit,
            "workflow_run_id": workflow_run_id,
        },
    }
    return base, experience


def build_no_candidate_record(
    *,
    parent_checkpoint: str | Path,
    tasks_path: str | Path,
    experience_dir: str | Path,
    task_count: int,
    min_accepted: int,
    challenge_seed: int,
    generation_seed: int,
    source_commit: str,
    workflow_run_id: str,
) -> dict[str, Any]:
    base, experience = _base_record(
        parent_checkpoint=Path(parent_checkpoint),
        tasks_path=Path(tasks_path),
        experience_dir=Path(experience_dir),
        task_count=task_count,
        min_accepted=min_accepted,
        challenge_seed=challenge_seed,
        generation_seed=generation_seed,
        source_commit=source_commit,
        workflow_run_id=workflow_run_id,
    )
    if experience["accepted"] >= min_accepted:
        raise ValueError("no-candidate record is invalid when threshold is met")
    base.update(
        {
            "status": "no_candidate",
            "reason": "insufficient_verified_experience",
            "candidate_trained": False,
            "promotion_attempted": False,
        }
    )
    base["record_sha256"] = hashlib.sha256(_canonical(base).encode("utf-8")).hexdigest()
    return base


def build_candidate_record(
    *,
    parent_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    tasks_path: str | Path,
    experience_dir: str | Path,
    parent_evaluation: str | Path,
    candidate_evaluation: str | Path,
    parent_benchmark: str | Path,
    candidate_benchmark: str | Path,
    promotion: str | Path,
    task_count: int,
    min_accepted: int,
    challenge_seed: int,
    generation_seed: int,
    source_commit: str,
    workflow_run_id: str,
) -> dict[str, Any]:
    parent_checkpoint = Path(parent_checkpoint)
    candidate_checkpoint = Path(candidate_checkpoint)
    base, experience = _base_record(
        parent_checkpoint=parent_checkpoint,
        tasks_path=Path(tasks_path),
        experience_dir=Path(experience_dir),
        task_count=task_count,
        min_accepted=min_accepted,
        challenge_seed=challenge_seed,
        generation_seed=generation_seed,
        source_commit=source_commit,
        workflow_run_id=workflow_run_id,
    )
    if experience["accepted"] < min_accepted:
        raise ValueError("candidate record requires the accepted threshold")
    promotion_path = Path(promotion)
    promotion_result = _load_json(promotion_path)
    candidate_hash = sha256_file(candidate_checkpoint)
    parent_hash = sha256_file(parent_checkpoint)
    if promotion_result.get("parent_checkpoint_sha256") != parent_hash:
        raise ValueError("promotion result parent hash mismatch")
    if promotion_result.get("candidate_checkpoint_sha256") != candidate_hash:
        raise ValueError("promotion result candidate hash mismatch")
    promoted = promotion_result.get("promoted")
    if not isinstance(promoted, bool):
        raise ValueError("promotion result is malformed")
    base.update(
        {
            "status": "candidate_promoted" if promoted else "candidate_rejected",
            "candidate_trained": True,
            "candidate_checkpoint_sha256": candidate_hash,
            "measurements": {
                "parent_evaluation_sha256": sha256_file(parent_evaluation),
                "candidate_evaluation_sha256": sha256_file(candidate_evaluation),
                "parent_benchmark_sha256": sha256_file(parent_benchmark),
                "candidate_benchmark_sha256": sha256_file(candidate_benchmark),
            },
            "promotion_attempted": True,
            "promotion_sha256": sha256_file(promotion_path),
            "promotion_decision_sha256": promotion_result.get("decision_sha256"),
            "promoted": promoted,
        }
    )
    base["record_sha256"] = hashlib.sha256(_canonical(base).encode("utf-8")).hexdigest()
    return base


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--min-accepted", type=int, required=True)
    parser.add_argument("--challenge-seed", type=int, required=True)
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact auditable Genesis M5 self-improvement-cycle record.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    no_candidate = subparsers.add_parser("no-candidate")
    _common(no_candidate)
    candidate = subparsers.add_parser("candidate")
    _common(candidate)
    candidate.add_argument("--candidate", type=Path, required=True)
    candidate.add_argument("--parent-eval", type=Path, required=True)
    candidate.add_argument("--candidate-eval", type=Path, required=True)
    candidate.add_argument("--parent-benchmark", type=Path, required=True)
    candidate.add_argument("--candidate-benchmark", type=Path, required=True)
    candidate.add_argument("--promotion", type=Path, required=True)
    args = parser.parse_args()

    common = {
        "parent_checkpoint": args.parent,
        "tasks_path": args.tasks,
        "experience_dir": args.experience,
        "task_count": args.task_count,
        "min_accepted": args.min_accepted,
        "challenge_seed": args.challenge_seed,
        "generation_seed": args.generation_seed,
        "source_commit": args.source_commit,
        "workflow_run_id": args.workflow_run_id,
    }
    if args.mode == "no-candidate":
        record = build_no_candidate_record(**common)
    else:
        record = build_candidate_record(
            **common,
            candidate_checkpoint=args.candidate,
            parent_evaluation=args.parent_eval,
            candidate_evaluation=args.candidate_eval,
            parent_benchmark=args.parent_benchmark,
            candidate_benchmark=args.candidate_benchmark,
            promotion=args.promotion,
        )
    _write(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
