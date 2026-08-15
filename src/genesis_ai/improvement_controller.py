from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .capability_index import GCI_DOMAINS, score_result

CONTROLLER_VERSION = "autonomous-improvement-controller-v1.3"
FORBIDDEN_CONTENT_KEYS = {"task", "tasks", "prompt", "prompts", "response", "responses", "answer", "answers", "oracle", "oracles", "text", "texts"}
PUBLIC_MIN_CHARS_VARIANTS = {
    "min-chars-20": 20,
    "min-chars-40": 40,
    "min-chars-80": 80,
}


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
        metrics[domain] = {"exact_accuracy": float(accuracy), "terminated_oracle_loss": float(loss), "termination_rate": float(termination)}
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


def _research_policy(evidence: dict[str, Any] | None) -> tuple[int, dict[str, Any] | None]:
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

    applied: dict[str, Any] | None = None
    ignored: list[dict[str, Any]] = []
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
        hint = {"lane": lane, "variant": variant, "improvement_fraction": float(improvement)}
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
) -> dict[str, Any]:
    if not incumbent_checkpoint_sha256 or len(incumbent_checkpoint_sha256) != 64:
        raise ValueError("incumbent checkpoint SHA-256 is required")
    if not isinstance(cycle_index, int) or isinstance(cycle_index, bool) or cycle_index < 1:
        raise ValueError("cycle_index must be a positive integer")
    if max_difficulty < 1:
        raise ValueError("max_difficulty must be positive")
    _reject_holdout_content(evaluation)
    metrics = _domain_metrics(evaluation)
    public_min_chars, research = _research_policy(research_evidence)
    difficulty = evaluation.get("difficulty")
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= max_difficulty:
        raise ValueError("evaluation difficulty is invalid")
    suite_version = evaluation.get("suite_version")
    suite_sha256 = evaluation.get("suite_sha256")
    if not isinstance(suite_version, str) or not suite_version:
        raise ValueError("suite_version is required")
    if not isinstance(suite_sha256, str) or len(suite_sha256) != 64:
        raise ValueError("suite_sha256 is required")

    focus = min(GCI_DOMAINS, key=lambda domain: (metrics[domain]["exact_accuracy"], -metrics[domain]["terminated_oracle_loss"], domain))
    focus_accuracy = metrics[focus]["exact_accuracy"]
    all_mastered = all(metrics[domain]["exact_accuracy"] >= 0.80 for domain in GCI_DOMAINS)
    target_difficulty = min(difficulty + 1, max_difficulty) if all_mastered else difficulty
    requires_new_suite = target_difficulty != difficulty
    mode = "raise-difficulty" if requires_new_suite else "repair-weakest-domain"
    replay = [domain for domain in GCI_DOMAINS if domain != focus]
    gci = score_result(evaluation)
    target_training_tokens = _cycle_budget(focus_accuracy)

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
            "focus_examples": 4_096,
            "replay_domains": replay,
            "replay_examples_per_domain": _replay_examples(target_training_tokens),
            "continuation_update_weights": {"focus": 0.70, "each_replay_domain": 0.15},
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan one autonomous Genesis improvement cycle from aggregate metrics only.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--incumbent-sha256", required=True)
    parser.add_argument("--cycle-index", type=int)
    parser.add_argument("--research-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
