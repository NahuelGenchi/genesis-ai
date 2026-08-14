from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_index import GCI_DOMAINS, score_result
from .domain_selection import generate_domain_tasks
from .ingest import sha256_file
from .terminated_eval import load_terminated_suite

LADDER_VERSION = "gci-ladder-v1"
REQUIRED_DIFFICULTIES = (1, 2, 3, 4, 5)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def _suite_task_hashes(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suite = load_terminated_suite(path)
    prompt_hashes: set[str] = set()
    task_hashes: set[str] = set()
    domain_counts: dict[str, int] = {}
    for ordinal, domain in enumerate(suite["domains"]):
        tasks = generate_domain_tasks(
            domain=domain,
            seed=int(suite["base_seed"]) + ordinal,
            count=int(suite["tasks_per_domain"]),
            difficulty=int(suite["difficulty"]),
        )
        domain_counts[domain] = len(tasks)
        for task in tasks:
            prompt_hashes.add(_sha256_text(str(task["prompt"])))
            task_hashes.add(_sha256_object(task))
    expected = int(suite["tasks_per_domain"]) * len(suite["domains"])
    if len(prompt_hashes) != expected or len(task_hashes) != expected:
        raise ValueError(f"suite contains duplicate generated tasks/prompts: {path}")
    return {
        "path": str(path),
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(path),
        "difficulty": int(suite["difficulty"]),
        "task_count": expected,
        "domain_counts": domain_counts,
        "prompt_hashes": prompt_hashes,
        "task_hashes": task_hashes,
    }


def build_ladder_manifest(suite_paths: Iterable[str | Path]) -> dict[str, Any]:
    entries = [_suite_task_hashes(path) for path in suite_paths]
    entries.sort(key=lambda item: int(item["difficulty"]))
    difficulties = tuple(int(item["difficulty"]) for item in entries)
    if difficulties != REQUIRED_DIFFICULTIES:
        raise ValueError(f"ladder requires difficulties {REQUIRED_DIFFICULTIES}, got {difficulties}")

    global_prompt_hashes: set[str] = set()
    global_task_hashes: set[str] = set()
    public_entries: list[dict[str, Any]] = []
    for entry in entries:
        prompt_hashes = entry.pop("prompt_hashes")
        task_hashes = entry.pop("task_hashes")
        prompt_overlap = global_prompt_hashes & prompt_hashes
        task_overlap = global_task_hashes & task_hashes
        if prompt_overlap or task_overlap:
            raise ValueError(
                f"difficulty {entry['difficulty']} overlaps earlier ladder suites: "
                f"prompts={len(prompt_overlap)} tasks={len(task_overlap)}"
            )
        global_prompt_hashes.update(prompt_hashes)
        global_task_hashes.update(task_hashes)
        public_entries.append({
            **entry,
            "prompt_set_sha256": _sha256_object(sorted(prompt_hashes)),
            "task_set_sha256": _sha256_object(sorted(task_hashes)),
        })

    return {
        "format_version": "1.0",
        "ladder_version": LADDER_VERSION,
        "suite_count": len(public_entries),
        "difficulties": list(REQUIRED_DIFFICULTIES),
        "total_tasks": sum(int(item["task_count"]) for item in public_entries),
        "exact_cross_suite_prompt_overlap_count": 0,
        "exact_cross_suite_task_overlap_count": 0,
        "global_prompt_set_sha256": _sha256_object(sorted(global_prompt_hashes)),
        "global_task_set_sha256": _sha256_object(sorted(global_task_hashes)),
        "suites": public_entries,
    }


def score_ladder(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(results)
    if len(values) != len(REQUIRED_DIFFICULTIES):
        raise ValueError("GCI-Ladder requires exactly five evaluation results")
    by_difficulty: dict[int, dict[str, Any]] = {}
    checkpoint_sha256: str | None = None
    for result in values:
        difficulty = result.get("difficulty")
        if not isinstance(difficulty, int) or difficulty not in REQUIRED_DIFFICULTIES:
            raise ValueError("invalid ladder result difficulty")
        if difficulty in by_difficulty:
            raise ValueError("duplicate ladder result difficulty")
        score = score_result(result)
        current_checkpoint = score.get("checkpoint_sha256")
        if checkpoint_sha256 is None:
            checkpoint_sha256 = current_checkpoint
        elif current_checkpoint != checkpoint_sha256:
            raise ValueError("all ladder results must score the same checkpoint")
        by_difficulty[difficulty] = score
    if tuple(sorted(by_difficulty)) != REQUIRED_DIFFICULTIES:
        raise ValueError("ladder result difficulties are incomplete")

    gci_values = [float(by_difficulty[difficulty]["score"]) for difficulty in REQUIRED_DIFFICULTIES]
    harmonic = 0.0 if any(value <= 0.0 for value in gci_values) else len(gci_values) / sum(1.0 / value for value in gci_values)
    worst_domain = min(
        100.0 * float(by_difficulty[difficulty]["domain_exact_accuracy"][domain])
        for difficulty in REQUIRED_DIFFICULTIES
        for domain in GCI_DOMAINS
    )
    return {
        "format_version": "1.0",
        "metric_version": LADDER_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "difficulty_gci": {str(difficulty): by_difficulty[difficulty]["score"] for difficulty in REQUIRED_DIFFICULTIES},
        "difficulty_suite_sha256": {str(difficulty): by_difficulty[difficulty]["suite_sha256"] for difficulty in REQUIRED_DIFFICULTIES},
        "ladder_score": harmonic,
        "worst_domain_exact_percent": worst_domain,
    }


def compare_ladders(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("metric_version") != LADDER_VERSION or candidate.get("metric_version") != LADDER_VERSION:
        raise ValueError("unsupported ladder comparison")
    if baseline.get("difficulty_suite_sha256") != candidate.get("difficulty_suite_sha256"):
        raise ValueError("ladder comparison requires identical suite hashes")
    before = float(baseline["ladder_score"])
    after = float(candidate["ladder_score"])
    absolute = after - before
    relative = None if before == 0.0 else 100.0 * absolute / before
    return {
        "format_version": "1.0",
        "metric_version": LADDER_VERSION,
        "baseline_score": before,
        "candidate_score": after,
        "absolute_point_change": absolute,
        "relative_percent_change": relative,
        "relative_percent_note": "N/A (zero baseline)" if relative is None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/score the frozen Genesis GCI difficulty ladder.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--suite", action="append", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--result", action="append", required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        output = build_ladder_manifest(args.suite)
    else:
        output = score_ladder([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.result])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
