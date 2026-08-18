from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .capability_index import GCI_DOMAINS, score_result

CONTROLLER_VERSION = "autonomous-improvement-controller-v1.5"
LEGACY_STRATEGY_ID = "legacy-focus-heavy-v1"
MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY = 3
MAX_NONPOSITIVE_GCI_ATTEMPTS_PER_STRATEGY = 2
MAX_REJECTED_ATTEMPTS_PER_STRATEGY = 4
HINT_END_TO_END_FAILURE_LIMIT = 5
FORBIDDEN_CONTENT_KEYS = {
    "task",
    "tasks",
    "prompt",
    "prompts",
    "response",
    "responses",
    "answer",
    "answers",
    "oracle",
    "oracles",
    "text",
    "texts",
}
PUBLIC_MIN_CHARS_VARIANTS = {
    "min-chars-20": 20,
    "min-chars-40": 40,
    "min-chars-80": 80,
}

# These interventions deliberately change high-level training behavior rather than
# merely changing a random seed. The strongest non-focus domain receives explicit
# anti-forgetting protection and the focus-example count changes sequence coverage.
RESEARCH_STRATEGIES = (
    {"id": "sequence-depth-v1", "focus_examples": 1_024, "focus_weight": 0.65, "strongest_replay_weight": 0.25},
    {"id": "anti-forgetting-v1", "focus_examples": 1_536, "focus_weight": 0.50, "strongest_replay_weight": 0.40},
    {"id": "balanced-transfer-v1", "focus_examples": 2_048, "focus_weight": 0.60, "strongest_replay_weight": 0.30},
    {"id": "broad-conservative-v1", "focus_examples": 4_096, "focus_weight": 0.55, "strongest_replay_weight": 0.35},
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _reject_holdout_content(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_CONTENT_KEYS:
                raise ValueError(f"controller input contains forbidden holdout content field: {path}.{key}")
            _reject_holdout_content(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_holdout_content(child, f"{path}[{index}]")


def _domain_metrics(evaluation: dict[str, Any]) -> dict[str, dict[str, float]]:
    domains = evaluation.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(GCI_DOMAINS):
        raise ValueError("controller requires exactly code, math, and structured aggregate domains")
    metrics: dict[str, dict[str, float]] = {}
    for domain in GCI_DOMAINS:
        block = domains[domain]
        if not isinstance(block, dict):
            raise ValueError(f"aggregate domain block is invalid: {domain}")
        accuracy = block.get("exact_accuracy")
        loss = block.get("terminated_oracle_loss")
        termination = block.get("termination_rate", 0.0)
        if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not 0.0 <= float(accuracy) <= 1.0:
            raise ValueError(f"invalid exact accuracy: {domain}")
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or float(loss) < 0.0:
            raise ValueError(f"invalid aggregate oracle loss: {domain}")
        if not isinstance(termination, (int, float)) or isinstance(termination, bool) or not 0.0 <= float(termination) <= 1.0:
            raise ValueError(f"invalid termination rate: {domain}")
        metrics[domain] = {
            "exact_accuracy": float(accuracy),
            "terminated_oracle_loss": float(loss),
            "termination_rate": float(termination),
        }
    return metrics


def _cycle_budget(accuracy: float) -> int:
    if accuracy < 0.50:
        return 3_000_000
    if accuracy < 0.80:
        return 2_500_000
    return 2_000_000


def _replay_examples(target_training_tokens: int) -> int:
    by_budget = {
        3_000_000: 1_024,
        2_500_000: 768,
        2_000_000: 512,
    }
    try:
        return by_budget[target_training_tokens]
    except KeyError as exc:
        raise ValueError("unsupported autonomous training budget") from exc


def _empty_history() -> dict[str, Any]:
    return {
        "cycles_considered": 0,
        "consecutive_rejections": 0,
        "strategy_attempts": {},
        "strategy_rejections": {},
        "strategy_zero_focus_gain": {},
        "strategy_nonpositive_gci": {},
        "hint_failures": {},
        "last_strategy": None,
    }


def _gate_gci_change(gate: dict[str, Any], cycle_dir: Path) -> float:
    gci = gate.get("gci_v1")
    if not isinstance(gci, dict):
        raise ValueError(f"missing GCI result in committed autonomous history: {cycle_dir}")
    change = gci.get("absolute_point_change")
    if not isinstance(change, (int, float)) or isinstance(change, bool):
        raise ValueError(f"invalid GCI change in committed autonomous history: {cycle_dir}")
    return float(change)


def _load_history(history_root: str | Path | None, incumbent_sha256: str) -> dict[str, Any]:
    empty = _empty_history()
    if history_root is None:
        return empty
    root = Path(history_root)
    if not root.exists():
        return empty
    if not root.is_dir():
        raise ValueError("autonomous history root must be a directory")

    records: list[dict[str, Any]] = []
    for cycle_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        plan_path = cycle_dir / "plan.json"
        gate_path = cycle_dir / "gate.json"
        if not plan_path.is_file() or not gate_path.is_file():
            continue
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"invalid committed autonomous history: {cycle_dir}") from exc
        if not isinstance(plan, dict) or not isinstance(gate, dict):
            raise ValueError(f"invalid committed autonomous history object: {cycle_dir}")
        if gate.get("baseline_checkpoint_sha256") != incumbent_sha256:
            continue

        strategy = plan.get("research_strategy")
        strategy_id = LEGACY_STRATEGY_ID
        if isinstance(strategy, dict) and isinstance(strategy.get("id"), str):
            strategy_id = str(strategy["id"])

        hint_key = None
        evidence = plan.get("research_evidence")
        if isinstance(evidence, dict):
            hint = evidence.get("applied_hint")
            if isinstance(hint, dict) and isinstance(hint.get("lane"), str) and isinstance(hint.get("variant"), str):
                hint_key = f"{hint['lane']}/{hint['variant']}"

        focus_gain = gate.get("focus_absolute_gain", 0.0)
        if not isinstance(focus_gain, (int, float)) or isinstance(focus_gain, bool):
            raise ValueError(f"invalid focus gain in committed autonomous history: {cycle_dir}")

        records.append(
            {
                "strategy": strategy_id,
                "decision": str(gate.get("decision", "")),
                "focus_gain": float(focus_gain),
                "gci_change": _gate_gci_change(gate, cycle_dir),
                "hint_key": hint_key,
            }
        )

    if not records:
        return empty

    strategy_attempts: dict[str, int] = {}
    strategy_rejections: dict[str, int] = {}
    strategy_zero_focus_gain: dict[str, int] = {}
    strategy_nonpositive_gci: dict[str, int] = {}
    hint_failures: dict[str, int] = {}

    for record in records:
        strategy = str(record["strategy"])
        strategy_attempts[strategy] = strategy_attempts.get(strategy, 0) + 1
        rejected = record["decision"] == "reject"
        zero_focus = rejected and float(record["focus_gain"]) <= 0.0
        nonpositive_gci = rejected and float(record["gci_change"]) <= 0.0

        if rejected:
            strategy_rejections[strategy] = strategy_rejections.get(strategy, 0) + 1
        if zero_focus:
            strategy_zero_focus_gain[strategy] = strategy_zero_focus_gain.get(strategy, 0) + 1
        if nonpositive_gci:
            strategy_nonpositive_gci[strategy] = strategy_nonpositive_gci.get(strategy, 0) + 1

        # A hint is an end-to-end failure if it cannot produce focus progress OR
        # if any focus progress comes at the expense of total frozen-suite GCI.
        if (zero_focus or nonpositive_gci) and record["hint_key"] is not None:
            key = str(record["hint_key"])
            hint_failures[key] = hint_failures.get(key, 0) + 1

    consecutive_rejections = 0
    for record in reversed(records):
        if record["decision"] != "reject":
            break
        consecutive_rejections += 1

    return {
        "cycles_considered": len(records),
        "consecutive_rejections": consecutive_rejections,
        "strategy_attempts": dict(sorted(strategy_attempts.items())),
        "strategy_rejections": dict(sorted(strategy_rejections.items())),
        "strategy_zero_focus_gain": dict(sorted(strategy_zero_focus_gain.items())),
        "strategy_nonpositive_gci": dict(sorted(strategy_nonpositive_gci.items())),
        "hint_failures": dict(sorted(hint_failures.items())),
        "last_strategy": records[-1]["strategy"],
    }


def _strategy_exhausted(history: dict[str, Any], strategy_id: str) -> bool:
    zero_gain = history.get("strategy_zero_focus_gain", {})
    nonpositive_gci = history.get("strategy_nonpositive_gci", {})
    rejections = history.get("strategy_rejections", {})
    if not isinstance(zero_gain, dict) or not isinstance(nonpositive_gci, dict) or not isinstance(rejections, dict):
        raise ValueError("invalid autonomous strategy history")
    return (
        int(zero_gain.get(strategy_id, 0)) >= MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY
        or int(nonpositive_gci.get(strategy_id, 0)) >= MAX_NONPOSITIVE_GCI_ATTEMPTS_PER_STRATEGY
        or int(rejections.get(strategy_id, 0)) >= MAX_REJECTED_ATTEMPTS_PER_STRATEGY
    )


def _failure_counts(history: dict[str, Any], strategy_id: str) -> dict[str, int]:
    attempts = history.get("strategy_attempts", {})
    rejections = history.get("strategy_rejections", {})
    zero_gain = history.get("strategy_zero_focus_gain", {})
    nonpositive_gci = history.get("strategy_nonpositive_gci", {})
    if not all(isinstance(value, dict) for value in (attempts, rejections, zero_gain, nonpositive_gci)):
        raise ValueError("invalid autonomous strategy history")
    return {
        "attempts": int(attempts.get(strategy_id, 0)),
        "rejections": int(rejections.get(strategy_id, 0)),
        "zero_focus_gain_rejections": int(zero_gain.get(strategy_id, 0)),
        "nonpositive_gci_rejections": int(nonpositive_gci.get(strategy_id, 0)),
    }


def _choose_strategy(
    history: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    focus: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if int(history.get("cycles_considered", 0)) == 0:
        return None, None

    # Keep the legacy policy until there is strong end-to-end evidence that it
    # stalls, regresses total GCI, or simply keeps getting rejected.
    if not _strategy_exhausted(history, LEGACY_STRATEGY_ID):
        return None, None

    selected: dict[str, Any] | None = None
    for strategy in RESEARCH_STRATEGIES:
        if not _strategy_exhausted(history, str(strategy["id"])):
            selected = strategy
            break

    if selected is None:
        counts = {
            strategy_id: _failure_counts(history, strategy_id)
            for strategy_id in (LEGACY_STRATEGY_ID, *(str(item["id"]) for item in RESEARCH_STRATEGIES))
        }
        raise ValueError(
            "all predeclared repair strategies exhausted; refusing to train another candidate "
            "from the unchanged incumbent until novel research evidence or controller strategy "
            f"is committed. failure_counts={_canonical(counts)}"
        )

    replay = [domain for domain in GCI_DOMAINS if domain != focus]
    strongest = max(
        replay,
        key=lambda domain: (
            metrics[domain]["exact_accuracy"],
            -metrics[domain]["terminated_oracle_loss"],
            domain,
        ),
    )
    other = next(domain for domain in replay if domain != strongest)
    other_weight = 1.0 - float(selected["focus_weight"]) - float(selected["strongest_replay_weight"])
    weights = {
        focus: float(selected["focus_weight"]),
        strongest: float(selected["strongest_replay_weight"]),
        other: other_weight,
    }

    strategy_id = str(selected["id"])
    counts = _failure_counts(history, strategy_id)
    strategy_block = {
        "id": strategy_id,
        "focus_examples": int(selected["focus_examples"]),
        "continuation_update_weights": weights,
        "strongest_replay_domain": strongest,
        "prior_attempts_same_incumbent": counts["attempts"],
        "prior_rejections_same_incumbent": counts["rejections"],
        "prior_zero_focus_gain_rejections": counts["zero_focus_gain_rejections"],
        "prior_nonpositive_gci_rejections": counts["nonpositive_gci_rejections"],
        "failure_limits": {
            "zero_focus_gain_rejections": MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY,
            "nonpositive_gci_rejections": MAX_NONPOSITIVE_GCI_ATTEMPTS_PER_STRATEGY,
            "total_rejections": MAX_REJECTED_ATTEMPTS_PER_STRATEGY,
        },
    }
    escalation = {
        "required": counts["attempts"] == 0,
        "reason": "legacy/high-level strategy exhausted",
        "actions": ["architecture-tournament", "tracked-research-issue"],
        "all_predeclared_strategies_exhausted": False,
    }
    return strategy_block, escalation


def _research_policy(
    evidence: dict[str, Any] | None,
    history: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    if evidence is None:
        return 0, None
    _reject_holdout_content(evidence, "research")
    if evidence.get("format_version") != "1.0" or evidence.get("farm_version") != "cpu-screen-v1":
        raise ValueError("unsupported CPU research evidence")
    if evidence.get("guards_pass") is not True:
        raise ValueError("CPU research evidence guards did not pass")
    if evidence.get("screening_only") is not True or evidence.get("promotion_eligible") is not False:
        raise ValueError("CPU research evidence must be screening-only and non-promoting")
    if evidence.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("CPU research evidence violates zero-cash contract")

    policy = evidence.get("gpu_policy")
    if not isinstance(policy, dict):
        raise ValueError("CPU research evidence GPU policy is missing")
    if policy.get("eligible_only_after_cpu_screen") is not True:
        raise ValueError("CPU research evidence does not enforce CPU screening")
    if policy.get("full_reproduction_and_frozen_evaluation_required_before_promotion") is not True:
        raise ValueError("CPU research evidence weakens promotion verification")
    if policy.get("screening_result_can_promote_checkpoint") is not False:
        raise ValueError("CPU research evidence grants screening promotion authority")

    source_commit = evidence.get("source_commit")
    run_id = evidence.get("workflow_run_id")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("CPU research evidence source commit is invalid")
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        raise ValueError("CPU research evidence workflow run id is invalid")

    eligible = evidence.get("expensive_stage_eligible")
    if not isinstance(eligible, list):
        raise ValueError("CPU research evidence eligible list is invalid")

    failures = {} if history is None else history.get("hint_failures", {})
    if not isinstance(failures, dict):
        raise ValueError("invalid autonomous hint history")

    applied: dict[str, Any] | None = None
    ignored: list[dict[str, Any]] = []
    retired: list[dict[str, Any]] = []
    public_min_chars = 0

    for raw in eligible:
        if not isinstance(raw, dict):
            raise ValueError("CPU research eligible hint must be an object")
        lane = raw.get("lane")
        variant = raw.get("variant")
        improvement = raw.get("improvement_fraction")
        if not isinstance(lane, str) or not isinstance(variant, str):
            raise ValueError("CPU research eligible hint identity is invalid")
        if not isinstance(improvement, (int, float)) or isinstance(improvement, bool) or float(improvement) < 0.0:
            raise ValueError("CPU research eligible hint improvement is invalid")

        hint = {
            "lane": lane,
            "variant": variant,
            "improvement_fraction": float(improvement),
        }
        hint_key = f"{lane}/{variant}"
        failure_count = int(failures.get(hint_key, 0))
        if failure_count >= HINT_END_TO_END_FAILURE_LIMIT:
            retired.append(
                {
                    **hint,
                    "end_to_end_zero_gain_rejections": failure_count,
                    "retirement_threshold": HINT_END_TO_END_FAILURE_LIMIT,
                }
            )
            continue

        if lane == "data-filtering":
            if variant not in PUBLIC_MIN_CHARS_VARIANTS:
                raise ValueError(f"unsupported screened data-filtering hint: {variant}")
            if applied is not None:
                raise ValueError("multiple data-filtering hints cannot be applied in one autonomous cycle")
            public_min_chars = PUBLIC_MIN_CHARS_VARIANTS[variant]
            applied = hint
        else:
            ignored.append(hint)

    summary = {
        "farm_version": evidence["farm_version"],
        "evidence_sha256": _sha256_object(evidence),
        "workflow_run_id": run_id,
        "source_commit": source_commit,
        "applied_hint": applied,
        "ignored_eligible_hints": ignored,
        "retired_eligible_hints": retired,
        "promotion_authority": False,
    }
    return public_min_chars, summary


def plan_next_cycle(
    evaluation: dict[str, Any],
    *,
    incumbent_checkpoint_sha256: str,
    cycle_index: int = 1,
    max_difficulty: int = 5,
    research_evidence: dict[str, Any] | None = None,
    history_root: str | Path | None = None,
) -> dict[str, Any]:
    if not incumbent_checkpoint_sha256 or len(incumbent_checkpoint_sha256) != 64:
        raise ValueError("incumbent checkpoint SHA-256 is required")
    if not isinstance(cycle_index, int) or isinstance(cycle_index, bool) or cycle_index < 1:
        raise ValueError("cycle_index must be a positive integer")
    if max_difficulty < 1:
        raise ValueError("max_difficulty must be positive")

    _reject_holdout_content(evaluation)
    metrics = _domain_metrics(evaluation)
    history = _load_history(history_root, incumbent_checkpoint_sha256)
    public_min_chars, research = _research_policy(research_evidence, history)

    difficulty = evaluation.get("difficulty")
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= max_difficulty:
        raise ValueError("evaluation difficulty is invalid")

    suite_version = evaluation.get("suite_version")
    suite_sha256 = evaluation.get("suite_sha256")
    if not isinstance(suite_version, str) or not suite_version:
        raise ValueError("suite_version is required")
    if not isinstance(suite_sha256, str) or len(suite_sha256) != 64:
        raise ValueError("suite_sha256 is required")

    focus = min(
        GCI_DOMAINS,
        key=lambda domain: (
            metrics[domain]["exact_accuracy"],
            -metrics[domain]["terminated_oracle_loss"],
            domain,
        ),
    )
    focus_accuracy = metrics[focus]["exact_accuracy"]
    all_mastered = all(metrics[domain]["exact_accuracy"] >= 0.80 for domain in GCI_DOMAINS)
    target_difficulty = min(difficulty + 1, max_difficulty) if all_mastered else difficulty
    requires_new_suite = target_difficulty != difficulty
    mode = "raise-difficulty" if requires_new_suite else "repair-weakest-domain"
    replay = [domain for domain in GCI_DOMAINS if domain != focus]
    gci = score_result(evaluation)
    target_training_tokens = _cycle_budget(focus_accuracy)

    strategy: dict[str, Any] | None = None
    escalation: dict[str, Any] | None = None
    # Strategy history at one difficulty must not block a legitimate difficulty
    # transition. The new suite is re-baselined before training by the workflow.
    if not requires_new_suite:
        strategy, escalation = _choose_strategy(history, metrics, focus)

    focus_examples = 4_096
    continuation_weights: dict[str, float] | dict[str, Any] = {
        "focus": 0.70,
        "each_replay_domain": 0.15,
    }
    if strategy is not None:
        focus_examples = int(strategy["focus_examples"])
        continuation_weights = dict(strategy["continuation_update_weights"])
        mode = "research-repair"

    plan = {
        "format_version": "1.0",
        "controller_version": CONTROLLER_VERSION,
        "input": {
            "cycle_index": cycle_index,
            "incumbent_checkpoint_sha256": incumbent_checkpoint_sha256,
            "suite_version": suite_version,
            "suite_sha256": suite_sha256,
            "difficulty": difficulty,
            "gci_v1": gci["score"],
            "aggregate_domain_metrics": metrics,
        },
        "evaluation_transition": {
            "new_suite_required": requires_new_suite,
            "target_difficulty": target_difficulty,
            "incumbent_must_be_scored_on_target_suite_before_training": requires_new_suite,
            "cross_difficulty_improvement_comparison_forbidden": True,
        },
        "decision": {
            "mode": mode,
            "focus_domain": focus,
            "target_difficulty": target_difficulty,
            "target_training_tokens": target_training_tokens,
            "focus_examples": focus_examples,
            "replay_domains": replay,
            "replay_examples_per_domain": _replay_examples(target_training_tokens),
            "continuation_update_weights": continuation_weights,
            "mandatory_first_and_terminator_coverage": True,
            "unique_target_contexts_only": True,
            "procedural_fraction": 0.80,
            "public_fraction": 0.20,
            "public_min_chars": public_min_chars,
            "cash_compute_cost_usd": 0.0,
        },
        "promotion_contract": {
            "same_suite_comparison_required": True,
            "minimum_focus_absolute_gain": 0.10 if focus_accuracy < 0.90 else 0.02,
            "minimum_gci_absolute_gain": 3.0,
            "maximum_nonfocus_absolute_regression": 0.05,
            "maximum_m3_loss_regression_fraction": 0.02,
            "zero_holdout_overlap_required": True,
            "semantic_reproduction_required": True,
            "zero_cash_compute_required": True,
            "live_incumbent_weight_mutation_forbidden": True,
        },
    }
    if history_root is not None:
        plan["history_summary"] = history
    if strategy is not None:
        plan["research_strategy"] = strategy
    if escalation is not None:
        plan["research_escalation"] = escalation
    if research is not None:
        plan["research_evidence"] = research

    plan["plan_sha256"] = _sha256_object(plan)
    return plan


def _scheduled_cycle_index(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    state_path = os.environ.get("STATE")
    if not state_path:
        raise ValueError("cycle index is required: pass --cycle-index or provide STATE")
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("autonomous state must be a JSON object")
    current = state.get("cycle_index")
    if not isinstance(current, int) or isinstance(current, bool) or current < 0:
        raise ValueError("autonomous state cycle_index is invalid")
    return current + 1


def _require_autonomy_running() -> None:
    """Fail closed before planning/training when persistent state is on research hold."""
    state_path = os.environ.get("STATE")
    if not state_path:
        return
    path = Path(state_path)
    if not path.is_file():
        raise SystemExit(f"fail-closed: autonomous state is missing: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise SystemExit("fail-closed: autonomous state must be a JSON object")
    status = state.get("autonomy_status", "running")
    breaker = state.get("circuit_breaker") or {}
    if not isinstance(breaker, dict):
        raise SystemExit("fail-closed: circuit_breaker state is invalid")
    if status != "running" or breaker.get("active") is True:
        reason = breaker.get("reason") or status
        raise SystemExit(f"autonomous candidate training is on research hold: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan one autonomous Genesis improvement cycle from aggregate metrics only."
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--incumbent-sha256", required=True)
    parser.add_argument("--cycle-index", type=int)
    parser.add_argument("--research-evidence", type=Path)
    parser.add_argument("--history-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require_autonomy_running()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a JSON object")

    research_evidence = None
    if args.research_evidence is not None:
        research_evidence = json.loads(args.research_evidence.read_text(encoding="utf-8"))
        if not isinstance(research_evidence, dict):
            raise ValueError("research evidence must be a JSON object")

    result = plan_next_cycle(
        evaluation,
        incumbent_checkpoint_sha256=args.incumbent_sha256,
        cycle_index=_scheduled_cycle_index(args.cycle_index),
        research_evidence=research_evidence,
        history_root=args.history_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
