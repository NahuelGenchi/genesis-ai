from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .capability_index import GCI_DOMAINS, score_result

CONTROLLER_VERSION = "autonomous-improvement-controller-v1"
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
    # With 4,096 focus examples + 512 replay examples for each preserved
    # domain, mandatory first/terminator anchors consume 10,240 target updates.
    # These floors keep meaningful continuation capacity after anchor coverage.
    if accuracy < 0.50:
        return 3_000_000
    if accuracy < 0.80:
        return 2_500_000
    return 2_000_000


def plan_next_cycle(
    evaluation: dict[str, Any],
    *,
    incumbent_checkpoint_sha256: str,
    max_difficulty: int = 5,
) -> dict[str, Any]:
    """Create a deterministic next-cycle plan from aggregate metrics only.

    The controller intentionally cannot consume prompts, answers, task bodies, or
    other private-holdout content. It may inspect only aggregate per-domain
    scores/losses plus suite identity metadata.
    """

    if not incumbent_checkpoint_sha256 or len(incumbent_checkpoint_sha256) != 64:
        raise ValueError("incumbent checkpoint SHA-256 is required")
    if max_difficulty < 1:
        raise ValueError("max_difficulty must be positive")
    _reject_holdout_content(evaluation)
    metrics = _domain_metrics(evaluation)
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
    mode = "raise-difficulty" if all_mastered and target_difficulty > difficulty else "repair-weakest-domain"
    budget = _cycle_budget(focus_accuracy)

    gci = score_result(evaluation)
    replay = [domain for domain in GCI_DOMAINS if domain != focus]
    plan = {
        "format_version": "1.0",
        "controller_version": CONTROLLER_VERSION,
        "input": {
            "incumbent_checkpoint_sha256": incumbent_checkpoint_sha256,
            "suite_version": suite_version,
            "suite_sha256": suite_sha256,
            "difficulty": difficulty,
            "gci_v1": gci["score"],
            "aggregate_domain_metrics": metrics,
        },
        "decision": {
            "mode": mode,
            "focus_domain": focus,
            "target_difficulty": target_difficulty,
            "target_training_tokens": budget,
            "focus_examples": 4_096,
            "replay_domains": replay,
            "replay_examples_per_domain": 512,
            "continuation_update_weights": {
                "focus": 0.70,
                "each_replay_domain": 0.15,
            },
            "mandatory_first_and_terminator_coverage": True,
            "unique_target_contexts_only": True,
            "procedural_fraction": 0.80,
            "public_fraction": 0.20,
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
    plan["plan_sha256"] = _sha256_object(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan one autonomous Genesis improvement cycle from aggregate metrics only.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--incumbent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a JSON object")
    result = plan_next_cycle(evaluation, incumbent_checkpoint_sha256=args.incumbent_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
