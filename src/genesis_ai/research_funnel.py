from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FUNNEL_VERSION = "weak-domain-successive-halving-v1"
WEAK_DOMAINS = ("math", "structured")


@dataclass(frozen=True)
class Stage:
    name: str
    target_training_tokens: int
    focus_examples: int
    replay_examples_per_domain: int
    survivors: int


STAGES: dict[str, Stage] = {
    "tiny": Stage("tiny", 150_000, 256, 64, 3),
    "medium": Stage("medium", 750_000, 1_024, 256, 1),
    "full": Stage("full", 3_000_000, 4_096, 1_024, 1),
}

# Each entry is a materially different hypothesis, not a seed-only retry.
VARIANTS: tuple[dict[str, Any], ...] = (
    {"id": "structured-full-sort-v1", "focus_domain": "structured", "curriculum_mode": "full-sort"},
    {"id": "structured-pairwise-rank-v1", "focus_domain": "structured", "curriculum_mode": "pairwise-rank"},
    {"id": "structured-prefix-next-v1", "focus_domain": "structured", "curriculum_mode": "prefix-next"},
    {"id": "structured-partial-completion-v1", "focus_domain": "structured", "curriculum_mode": "partial-completion"},
    {"id": "structured-short-to-long-v1", "focus_domain": "structured", "curriculum_mode": "short-to-long"},
    {"id": "structured-mixed-decomposition-v1", "focus_domain": "structured", "curriculum_mode": "mixed-decomposition"},
    {"id": "math-operation-decomposition-v1", "focus_domain": "math", "curriculum_mode": "operation-decomposition"},
    {"id": "math-direct-plus-steps-v1", "focus_domain": "math", "curriculum_mode": "direct-plus-steps"},
)

VARIANT_BY_ID = {str(item["id"]): item for item in VARIANTS}


def stage_config(name: str) -> Stage:
    try:
        return STAGES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported research-funnel stage: {name}") from exc


def variant_config(variant_id: str) -> dict[str, Any]:
    try:
        return dict(VARIANT_BY_ID[variant_id])
    except KeyError as exc:
        raise ValueError(f"unsupported weak-domain curriculum variant: {variant_id}") from exc


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _domain(eval_result: dict[str, Any], name: str) -> dict[str, float]:
    domains = eval_result.get("domains")
    if not isinstance(domains, dict) or name not in domains or not isinstance(domains[name], dict):
        raise ValueError(f"evaluation is missing aggregate domain: {name}")
    block = domains[name]
    accuracy = block.get("exact_accuracy")
    loss = block.get("terminated_oracle_loss")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise ValueError(f"invalid exact accuracy for {name}")
    if not isinstance(loss, (int, float)) or isinstance(loss, bool) or float(loss) <= 0.0:
        raise ValueError(f"invalid terminated oracle loss for {name}")
    return {"exact_accuracy": float(accuracy), "terminated_oracle_loss": float(loss)}


def score_screen(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    training: dict[str, Any],
    variant_id: str,
) -> dict[str, Any]:
    variant = variant_config(variant_id)
    processed_tokens = training.get("processed_tokens")
    if not isinstance(processed_tokens, int) or isinstance(processed_tokens, bool) or processed_tokens <= 0:
        raise ValueError("screen training result must report positive processed_tokens")
    if training.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("screen violated zero-cash compute contract")

    deltas: dict[str, float] = {}
    loss_reductions: dict[str, float] = {}
    for domain in ("code", "math", "structured"):
        before = _domain(baseline, domain)
        after = _domain(candidate, domain)
        deltas[domain] = after["exact_accuracy"] - before["exact_accuracy"]
        loss_reductions[domain] = 1.0 - after["terminated_oracle_loss"] / before["terminated_oracle_loss"]

    weak_exact_gain = sum(deltas[domain] for domain in WEAK_DOMAINS) / len(WEAK_DOMAINS)
    weak_loss_reduction = sum(loss_reductions[domain] for domain in WEAK_DOMAINS) / len(WEAK_DOMAINS)
    code_delta = deltas["code"]
    focus = str(variant["focus_domain"])
    focus_gain = deltas[focus]

    # Exact accuracy is primary. Oracle-loss movement is only a screening
    # tie-breaker so a 0%-exact tiny run can still reveal useful learning signal.
    raw_quality = 100.0 * weak_exact_gain + 10.0 * weak_loss_reduction + 25.0 * max(focus_gain, 0.0)
    quality_per_million_tokens = raw_quality / (processed_tokens / 1_000_000.0)

    hard_code_regression = code_delta < -0.05
    no_weak_signal = weak_exact_gain <= 0.0 and weak_loss_reduction <= 0.0 and focus_gain <= 0.0
    advance_eligible = not hard_code_regression and not no_weak_signal
    early_stop_reason = None
    if hard_code_regression:
        early_stop_reason = "code-retention-boundary"
    elif no_weak_signal:
        early_stop_reason = "no-positive-weak-domain-signal"

    return {
        "format_version": "1.0",
        "funnel_version": FUNNEL_VERSION,
        "variant_id": variant_id,
        "focus_domain": focus,
        "processed_tokens": processed_tokens,
        "domain_exact_deltas": deltas,
        "domain_oracle_loss_reductions": loss_reductions,
        "weak_exact_gain": weak_exact_gain,
        "weak_oracle_loss_reduction": weak_loss_reduction,
        "focus_exact_gain": focus_gain,
        "code_exact_delta": code_delta,
        "quality_score": raw_quality,
        "quality_per_million_tokens": quality_per_million_tokens,
        "advance_eligible": advance_eligible,
        "early_stop_reason": early_stop_reason,
        "promotion_authority": False,
        "cash_compute_cost_usd": 0.0,
    }


def rank_screen_directory(
    *,
    baseline_path: str | Path,
    candidates_root: str | Path,
    stage: str,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    root = Path(candidates_root)
    config = stage_config(stage)
    scored: list[dict[str, Any]] = []
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        variant_id = variant_dir.name
        if variant_id not in VARIANT_BY_ID:
            continue
        evaluation_path = variant_dir / "evaluation.json"
        training_path = variant_dir / "training.json"
        if not evaluation_path.is_file() or not training_path.is_file():
            continue
        scored.append(
            score_screen(
                baseline=baseline,
                candidate=_load(evaluation_path),
                training=_load(training_path),
                variant_id=variant_id,
            )
        )
    if not scored:
        raise ValueError("no complete screening candidates found")

    scored.sort(
        key=lambda item: (
            bool(item["advance_eligible"]),
            float(item["quality_per_million_tokens"]),
            float(item["weak_exact_gain"]),
            float(item["weak_oracle_loss_reduction"]),
            str(item["variant_id"]),
        ),
        reverse=True,
    )
    eligible = [item for item in scored if bool(item["advance_eligible"])]
    survivors = eligible[: config.survivors]
    return {
        "format_version": "1.0",
        "funnel_version": FUNNEL_VERSION,
        "stage": stage,
        "target_training_tokens_per_candidate": config.target_training_tokens,
        "candidate_count": len(scored),
        "eligible_count": len(eligible),
        "advance": bool(survivors),
        "survivor_ids": [str(item["variant_id"]) for item in survivors],
        "ranking": scored,
        "promotion_authority": False,
        "cash_compute_cost_usd": 0.0,
    }


def should_continue_full_screen(summary: dict[str, Any]) -> bool:
    if summary.get("stage") != "medium" or summary.get("funnel_version") != FUNNEL_VERSION:
        raise ValueError("expected medium-stage funnel summary")
    ranking = summary.get("ranking")
    survivors = summary.get("survivor_ids")
    if not isinstance(ranking, list) or not isinstance(survivors, list) or len(survivors) > 1:
        raise ValueError("invalid medium-stage summary")
    return bool(summary.get("advance")) and len(survivors) == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan/rank resource-efficient weak-domain research screens.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("variants")
    list_parser.add_argument("--output", type=Path)

    rank_parser = sub.add_parser("rank")
    rank_parser.add_argument("--baseline", type=Path, required=True)
    rank_parser.add_argument("--candidates", type=Path, required=True)
    rank_parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    rank_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "variants":
        result = {
            "format_version": "1.0",
            "funnel_version": FUNNEL_VERSION,
            "variants": list(VARIANTS),
            "stages": {name: stage.__dict__ for name, stage in STAGES.items()},
            "promotion_authority": False,
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(rendered, end="")
        return

    result = rank_screen_directory(
        baseline_path=args.baseline,
        candidates_root=args.candidates,
        stage=args.stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
