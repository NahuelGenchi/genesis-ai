from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .domain_selection import generate_domain_tasks
from .ingest import sha256_file
from .terminated_eval import load_terminated_suite
from .weak_domain_funnel import FUNNEL_VERSION, TINY_SURVIVORS, select_survivors

DEV_TASKS_PER_DOMAIN = 20
DEV_SEED = int.from_bytes(hashlib.sha256(f"{FUNNEL_VERSION}:development-suite".encode("utf-8")).digest()[:8], "big") % 2_000_000_000
DEV_GENERATION_SEED = int.from_bytes(hashlib.sha256(f"{FUNNEL_VERSION}:development-generation".encode("utf-8")).digest()[:8], "big") % 2_000_000_000


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_hashes(suite: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for domain_ordinal, domain in enumerate(suite["domains"]):
        tasks = generate_domain_tasks(
            domain=domain,
            seed=int(suite["base_seed"]) + domain_ordinal,
            count=int(suite["tasks_per_domain"]),
            difficulty=int(suite["difficulty"]),
        )
        for task in tasks:
            hashes.add(_sha256_text(str(task["prompt"])))
    return hashes


def build_development_suite(*, frozen_suite_path: str | Path) -> dict[str, Any]:
    frozen = load_terminated_suite(frozen_suite_path)
    development = {
        "format_version": frozen["format_version"],
        "suite_version": frozen["suite_version"],
        "base_seed": DEV_SEED,
        "tasks_per_domain": DEV_TASKS_PER_DOMAIN,
        "difficulty": frozen["difficulty"],
        "domains": list(frozen["domains"]),
        "generation": {
            "max_new_tokens": frozen["generation"]["max_new_tokens"],
            "temperature": frozen["generation"]["temperature"],
            "top_k": 1,
            "seed": DEV_GENERATION_SEED,
        },
        "termination": dict(frozen["termination"]),
        "selection_rule": frozen["selection_rule"],
    }
    frozen_hashes = _prompt_hashes(frozen)
    development_hashes = _prompt_hashes(development)
    overlap = frozen_hashes & development_hashes
    if overlap:
        raise ValueError("development verifier overlaps frozen promotion holdout")
    development["research_metadata"] = {
        "research_funnel_version": FUNNEL_VERSION,
        "screening_only": True,
        "promotion_authority": False,
        "cash_compute_cost_usd": 0.0,
        "frozen_holdout_prompt_overlap_count": 0,
        "frozen_suite_sha256": sha256_file(frozen_suite_path),
    }
    return development


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summarize_candidate(
    *,
    variant_id: str,
    curriculum_path: str | Path,
    training_path: str | Path,
    baseline_eval_path: str | Path,
    candidate_eval_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    curriculum = _load(curriculum_path)
    training = _load(training_path)
    baseline = _load(baseline_eval_path)
    candidate = _load(candidate_eval_path)
    if curriculum.get("variant_id") != variant_id:
        raise ValueError("variant/curriculum mismatch")
    if curriculum.get("research_funnel_version") != FUNNEL_VERSION:
        raise ValueError("curriculum funnel version mismatch")
    if curriculum.get("screening_only") is not True or curriculum.get("promotion_authority") is not False:
        raise ValueError("curriculum screening authority drift")
    if training.get("screening_only") is not True or training.get("promotion_authority") is not False:
        raise ValueError("training screening authority drift")
    if float(training.get("cash_compute_cost_usd", -1.0)) != 0.0:
        raise ValueError("training violates zero-cash contract")
    if int(curriculum.get("exact_holdout_prompt_overlap_count", -1)) != 0:
        raise ValueError("curriculum overlaps frozen holdout")
    if baseline.get("suite_sha256") != candidate.get("suite_sha256"):
        raise ValueError("baseline/candidate development suite mismatch")

    focus = str(curriculum["focus_domain"])
    baseline_focus = float(baseline["domains"][focus]["exact_accuracy"])
    candidate_focus = float(candidate["domains"][focus]["exact_accuracy"])
    baseline_code = float(baseline["domains"]["code"]["exact_accuracy"])
    candidate_code = float(candidate["domains"]["code"]["exact_accuracy"])

    return {
        "format_version": "1.0",
        "research_funnel_version": FUNNEL_VERSION,
        "variant_id": variant_id,
        "funnel_stage": curriculum["funnel_stage"],
        "focus_domain": focus,
        "processed_tokens": int(training["processed_tokens"]),
        "target_training_tokens": int(training["target_training_tokens"]),
        "weak_domain_gain_pp": 100.0 * (candidate_focus - baseline_focus),
        "baseline_focus_exact_accuracy": baseline_focus,
        "candidate_focus_exact_accuracy": candidate_focus,
        "code_retention_pp": 100.0 * candidate_code,
        "code_change_pp": 100.0 * (candidate_code - baseline_code),
        "development_oracle_loss": float(candidate["domains"][focus]["terminated_oracle_loss"]),
        "baseline_development_oracle_loss": float(baseline["domains"][focus]["terminated_oracle_loss"]),
        "development_suite_sha256": candidate["suite_sha256"],
        "curriculum_sha256": sha256_file(curriculum_path),
        "training_run_sha256": sha256_file(training_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "holdout_prompt_overlap_count": 0,
        "screening_only": True,
        "promotion_authority": False,
        "cash_compute_cost_usd": 0.0,
    }


def rank_results(*, results_dir: str | Path, keep: int = TINY_SURVIVORS) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results = []
    for path in sorted(results_dir.glob("result-*.json")):
        results.append(_load(path))
    if len(results) != 7:
        raise ValueError(f"tiny screen requires all seven variant results, found {len(results)}")
    survivors = select_survivors(results, keep=keep)
    return {
        "format_version": "1.0",
        "research_funnel_version": FUNNEL_VERSION,
        "funnel_stage": "tiny",
        "candidate_count": len(results),
        "survivor_count": len(survivors),
        "results": sorted(results, key=lambda item: str(item["variant_id"])),
        "survivors": [str(item["variant_id"]) for item in survivors],
        "selection_rule": "highest weak-domain gain, then code retention, then lowest development oracle loss, then variant id",
        "screening_only": True,
        "promotion_authority": False,
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and summarize the weak-domain development-only tiny screen.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dev = subparsers.add_parser("build-dev-suite")
    dev.add_argument("--frozen-suite", type=Path, required=True)
    dev.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--variant", required=True)
    summarize.add_argument("--curriculum", type=Path, required=True)
    summarize.add_argument("--training", type=Path, required=True)
    summarize.add_argument("--baseline-eval", type=Path, required=True)
    summarize.add_argument("--candidate-eval", type=Path, required=True)
    summarize.add_argument("--checkpoint", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    rank = subparsers.add_parser("rank")
    rank.add_argument("--results-dir", type=Path, required=True)
    rank.add_argument("--keep", type=int, default=TINY_SURVIVORS)
    rank.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "build-dev-suite":
        payload = build_development_suite(frozen_suite_path=args.frozen_suite)
    elif args.command == "summarize":
        payload = summarize_candidate(
            variant_id=args.variant,
            curriculum_path=args.curriculum,
            training_path=args.training,
            baseline_eval_path=args.baseline_eval,
            candidate_eval_path=args.candidate_eval,
            checkpoint_path=args.checkpoint,
        )
    else:
        payload = rank_results(results_dir=args.results_dir, keep=args.keep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
