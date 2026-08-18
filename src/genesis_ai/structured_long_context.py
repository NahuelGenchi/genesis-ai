from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .autonomous_curriculum import (
    CURRICULUM_VERSION,
    _canonical,
    _domain_seed,
    _sha256_object,
    _validate_plan,
    generate_domain_records,
)
from .improvement_controller import CONTROLLER_VERSION
from .ingest import sha256_file
from .multidomain_curriculum import frozen_holdouts
from .terminated_eval import load_terminated_suite
from .tokenizer import ByteBPETokenizer

EXPERIMENT_VERSION = "structured-long-context-v1"
POOL_EXAMPLES = 6144
FOCUS_EXAMPLES = 4096
REPLAY_EXAMPLES = 1024
TARGET_TRAINING_TOKENS = 3_000_000
DOMAINS = ("code", "math", "structured")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_plan(
    *,
    evaluation_path: str | Path,
    state_path: str | Path,
    suite_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    evaluation = _load_object(evaluation_path)
    state = _load_object(state_path)
    suite = load_terminated_suite(suite_path)

    if state.get("autonomy_status") != "research_hold":
        raise ValueError("structured long-context experiment requires the committed research hold")
    breaker = state.get("circuit_breaker")
    if not isinstance(breaker, dict) or breaker.get("active") is not True:
        raise ValueError("structured long-context experiment requires the active circuit breaker")
    if state.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("committed autonomous state violates zero-cash contract")

    domains = evaluation.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
        raise ValueError("baseline evaluation must contain code/math/structured")
    metrics: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        block = domains[domain]
        if not isinstance(block, dict):
            raise ValueError(f"invalid baseline domain: {domain}")
        metrics[domain] = {
            "exact_accuracy": float(block["exact_accuracy"]),
            "terminated_oracle_loss": float(block["terminated_oracle_loss"]),
            "termination_rate": float(block["termination_rate"]),
        }

    difficulty = int(state["difficulty"])
    if int(suite["difficulty"]) != difficulty or int(evaluation["difficulty"]) != difficulty:
        raise ValueError("baseline/suite difficulty drifted from committed state")
    suite_sha = sha256_file(suite_path)
    if str(evaluation.get("suite_sha256")) != suite_sha:
        raise ValueError("baseline evaluation is not on the target frozen suite")
    checkpoint_sha = sha256_file(checkpoint_path)
    if str(evaluation.get("checkpoint_sha256")) != checkpoint_sha:
        raise ValueError("baseline evaluation is not from the committed incumbent")

    gci = 100.0 * sum(metrics[d]["exact_accuracy"] for d in DOMAINS) / len(DOMAINS)
    incumbent_gci = float(state["incumbent_gci_v1"])
    if not math.isclose(gci, incumbent_gci, abs_tol=1e-9):
        raise ValueError("fresh baseline GCI differs from committed incumbent GCI")

    weights = {"structured": 0.60, "code": 0.30, "math": 0.10}
    plan: dict[str, Any] = {
        "format_version": "1.0",
        "controller_version": CONTROLLER_VERSION,
        "input": {
            "cycle_index": int(state["cycle_index"]) + 1,
            "difficulty": difficulty,
            "suite_version": str(suite["suite_version"]),
            "suite_sha256": suite_sha,
            "incumbent_checkpoint_sha256": checkpoint_sha,
            "gci_v1": gci,
            "aggregate_domain_metrics": metrics,
        },
        "decision": {
            "mode": "research-repair",
            "focus_domain": "structured",
            "replay_domains": ["code", "math"],
            "focus_examples": FOCUS_EXAMPLES,
            "replay_examples_per_domain": REPLAY_EXAMPLES,
            "target_difficulty": difficulty,
            "target_training_tokens": TARGET_TRAINING_TOKENS,
            "procedural_fraction": 0.8,
            "public_fraction": 0.2,
            "public_min_chars": 0,
            "continuation_update_weights": weights,
            "mandatory_first_and_terminator_coverage": True,
            "unique_target_contexts_only": True,
            "cash_compute_cost_usd": 0.0,
        },
        "evaluation_transition": {
            "new_suite_required": False,
            "target_difficulty": difficulty,
            "incumbent_must_be_scored_on_target_suite_before_training": False,
            "cross_difficulty_improvement_comparison_forbidden": True,
        },
        "promotion_contract": {
            "minimum_gci_absolute_gain": 3.0,
            "minimum_focus_absolute_gain": 0.10,
            "maximum_nonfocus_absolute_regression": 0.05,
            "maximum_m3_loss_regression_fraction": 0.02,
            "same_suite_comparison_required": True,
            "semantic_reproduction_required": True,
            "zero_holdout_overlap_required": True,
            "zero_cash_compute_required": True,
            "live_incumbent_weight_mutation_forbidden": True,
        },
        "research_strategy": {
            "id": EXPERIMENT_VERSION,
            "focus_examples": FOCUS_EXAMPLES,
            "candidate_pool_examples": POOL_EXAMPLES,
            "continuation_update_weights": weights,
            "selection_rule": "within-generator longest terminated responses at proportional generator quotas",
            "diagnostic_basis": {
                "committed_incumbent_structured_exact": 0.0,
                "oracle_context_token_accuracy": 0.06070889894419306,
                "oracle_context_q4_token_accuracy": 0.009819967266775777,
                "free_generation_mean_correct_prefix_tokens": 0.0,
            },
        },
        "research_escalation": {
            "required": True,
            "reason": "all continuation strategies exhausted; committed diagnostic shows first-token and late-sequence structured representation failure",
            "actions": ["verifier-aligned-long-context-curriculum"],
            "all_predeclared_strategies_exhausted": True,
        },
        "history_summary": {
            "committed_cycle_index": int(state["cycle_index"]),
            "autonomy_status": str(state["autonomy_status"]),
            "circuit_breaker_reason": str(breaker["reason"]),
        },
    }
    plan["plan_sha256"] = _sha256_object(plan)
    _validate_plan(plan)
    return plan


def _proportional_quotas(groups: dict[str, list[dict[str, Any]]], total: int) -> dict[str, int]:
    population = sum(len(records) for records in groups.values())
    if population < total:
        raise ValueError("candidate pool smaller than requested focus set")
    raw = {name: total * len(records) / population for name, records in groups.items()}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(groups, key=lambda name: (-(raw[name] - quotas[name]), name))
    for name in order[:remainder]:
        quotas[name] += 1
    if sum(quotas.values()) != total:
        raise AssertionError("generator quota accounting drifted")
    return quotas


def _select_long_context_focus(
    pool: list[dict[str, Any]], tokenizer: ByteBPETokenizer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lengths: dict[str, int] = {}
    for record in pool:
        generator = str(record["provenance"]["generator"])
        groups[generator].append(record)
        lengths[str(record["id"])] = len(tokenizer.encode(str(record["response"]) + "\n"))
    quotas = _proportional_quotas(groups, FOCUS_EXAMPLES)

    selected: list[dict[str, Any]] = []
    generator_stats: dict[str, Any] = {}
    for generator in sorted(groups):
        ranked = sorted(groups[generator], key=lambda record: (-lengths[str(record["id"])], str(record["id"])))
        quota = quotas[generator]
        chosen = ranked[:quota]
        if len(chosen) != quota:
            raise ValueError(f"insufficient records for generator {generator}")
        selected.extend(chosen)
        generator_stats[generator] = {
            "pool_examples": len(ranked),
            "selected_examples": len(chosen),
            "pool_mean_terminated_response_tokens": sum(lengths[str(r["id"])] for r in ranked) / len(ranked),
            "selected_mean_terminated_response_tokens": sum(lengths[str(r["id"])] for r in chosen) / len(chosen),
            "selected_min_terminated_response_tokens": min(lengths[str(r["id"])] for r in chosen),
            "selected_max_terminated_response_tokens": max(lengths[str(r["id"])] for r in chosen),
        }

    selected.sort(key=lambda record: str(record["id"]))
    if len(selected) != FOCUS_EXAMPLES or len({str(r["id"]) for r in selected}) != FOCUS_EXAMPLES:
        raise AssertionError("structured focus selection is not unique and complete")
    pool_mean = sum(lengths[str(r["id"])] for r in pool) / len(pool)
    selected_mean = sum(lengths[str(r["id"])] for r in selected) / len(selected)
    if selected_mean < pool_mean:
        raise AssertionError("long-context selection unexpectedly shortened structured targets")
    return selected, {
        "pool_examples": len(pool),
        "selected_examples": len(selected),
        "pool_mean_terminated_response_tokens": pool_mean,
        "selected_mean_terminated_response_tokens": selected_mean,
        "mean_token_length_gain_fraction": selected_mean / pool_mean - 1.0,
        "generator_stats": generator_stats,
    }


def build_curriculum(
    *,
    plan_path: str | Path,
    suite_path: str | Path,
    tokenizer_path: str | Path,
    public_data: str | Path,
    records_path: str | Path,
) -> dict[str, Any]:
    plan = _load_object(plan_path)
    _validate_plan(plan)
    if plan.get("research_strategy", {}).get("id") != EXPERIMENT_VERSION:
        raise ValueError("plan is not bound to the structured long-context experiment")

    suite = load_terminated_suite(suite_path)
    actual_suite_sha = sha256_file(suite_path)
    if actual_suite_sha != plan["input"]["suite_sha256"]:
        raise ValueError("frozen target suite changed after plan declaration")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    _, holdout_prompt_hashes, _ = frozen_holdouts(suite_path)
    seen: set[str] = set()
    plan_sha = str(plan["plan_sha256"])

    pool, pool_metrics = generate_domain_records(
        tokenizer=tokenizer,
        domain="structured",
        role="focus",
        count=POOL_EXAMPLES,
        difficulty=int(plan["decision"]["target_difficulty"]),
        seed=_domain_seed(plan_sha, "structured"),
        holdout_prompt_hashes=holdout_prompt_hashes,
        global_seen_prompt_hashes=seen,
        context_length=128,
        plan_sha256=plan_sha,
    )
    focus_records, selection = _select_long_context_focus(pool, tokenizer)

    replay_records: list[dict[str, Any]] = []
    replay_summary: dict[str, Any] = {}
    for domain in ("code", "math"):
        records, metrics = generate_domain_records(
            tokenizer=tokenizer,
            domain=domain,
            role="replay",
            count=REPLAY_EXAMPLES,
            difficulty=int(plan["decision"]["target_difficulty"]),
            seed=_domain_seed(plan_sha, domain),
            holdout_prompt_hashes=holdout_prompt_hashes,
            global_seen_prompt_hashes=seen,
            context_length=128,
            plan_sha256=plan_sha,
        )
        replay_records.extend(records)
        replay_summary[domain] = {"role": "replay", **metrics}

    all_records = focus_records + replay_records
    selected_prompt_hashes = {_sha256_text(str(record["prompt"])) for record in all_records}
    if selected_prompt_hashes & holdout_prompt_hashes:
        raise ValueError("blocking structured long-context overlap with target holdout")
    if len(selected_prompt_hashes) != len(all_records):
        raise ValueError("structured long-context curriculum contains duplicate prompts")

    records_path = Path(records_path)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(_canonical(record) + "\n")

    focus_response_tokens = sum(len(tokenizer.encode(str(r["response"]) + "\n")) for r in focus_records)
    domain_records = {
        "structured": {
            "role": "focus",
            "examples": len(focus_records),
            "attempts": int(pool_metrics["attempts"]),
            "terminated_response_tokens": focus_response_tokens,
        },
        **replay_summary,
    }
    result = {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "screening_only": True,
        "promotion_authority": False,
        "plan_sha256": plan_sha,
        "incumbent_checkpoint_sha256": plan["input"]["incumbent_checkpoint_sha256"],
        "focus_domain": "structured",
        "focus_examples": FOCUS_EXAMPLES,
        "replay_examples_per_domain": REPLAY_EXAMPLES,
        "target_difficulty": int(plan["decision"]["target_difficulty"]),
        "target_training_tokens": TARGET_TRAINING_TOKENS,
        "procedural_fraction": 0.8,
        "public_fraction": 0.2,
        "public_min_chars": 0,
        "continuation_update_weights": plan["decision"]["continuation_update_weights"],
        "domain_records": domain_records,
        "structured_pool": selection,
        "record_count": len(all_records),
        "record_set_sha256": _sha256_object(all_records),
        "records_file_sha256": sha256_file(records_path),
        "target_suite_version": str(suite["suite_version"]),
        "target_suite_sha256": actual_suite_sha,
        "exact_holdout_prompt_overlap_count": 0,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "public_manifest_sha256": sha256_file(Path(public_data) / "manifest.json"),
        "cash_compute_cost_usd": 0.0,
        "research_strategy": plan["research_strategy"],
        "research_escalation": plan["research_escalation"],
        "history_summary": plan["history_summary"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the structured long-context research plan/curriculum.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--evaluation", type=Path, required=True)
    plan_parser.add_argument("--state", type=Path, required=True)
    plan_parser.add_argument("--suite", type=Path, required=True)
    plan_parser.add_argument("--checkpoint", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    curriculum_parser = subparsers.add_parser("curriculum")
    curriculum_parser.add_argument("--plan", type=Path, required=True)
    curriculum_parser.add_argument("--suite", type=Path, required=True)
    curriculum_parser.add_argument("--tokenizer", type=Path, required=True)
    curriculum_parser.add_argument("--public-data", type=Path, required=True)
    curriculum_parser.add_argument("--records", type=Path, required=True)
    curriculum_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(
            evaluation_path=args.evaluation,
            state_path=args.state,
            suite_path=args.suite,
            checkpoint_path=args.checkpoint,
        )
    else:
        result = build_curriculum(
            plan_path=args.plan,
            suite_path=args.suite,
            tokenizer_path=args.tokenizer,
            public_data=args.public_data,
            records_path=args.records,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
