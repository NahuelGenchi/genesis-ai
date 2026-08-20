from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import autonomous_training
from .weak_domain_funnel import FUNNEL_VERSION, MEDIUM_TOKEN_BUDGET, TINY_TOKEN_BUDGET

SCREEN_REPLAY_EXAMPLES_BY_BUDGET = {
    TINY_TOKEN_BUDGET: 128,
    MEDIUM_TOKEN_BUDGET: 256,
}
SCREEN_STAGE_BY_BUDGET = {
    TINY_TOKEN_BUDGET: "tiny",
    MEDIUM_TOKEN_BUDGET: "medium",
}


def _load_contract(curriculum_lock: str | Path) -> dict[str, Any]:
    value = json.loads(Path(curriculum_lock).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("weak-domain curriculum lock must be a JSON object")
    return value


def validate_screening_contract(curriculum_lock: str | Path) -> dict[str, Any]:
    curriculum = _load_contract(curriculum_lock)
    if curriculum.get("research_funnel_version") != FUNNEL_VERSION:
        raise ValueError("weak-domain curriculum is not bound to the frozen funnel version")
    if curriculum.get("screening_only") is not True:
        raise ValueError("weak-domain screen must be screening-only")
    if curriculum.get("promotion_authority") is not False:
        raise ValueError("weak-domain screen cannot have promotion authority")
    if float(curriculum.get("cash_compute_cost_usd", -1.0)) != 0.0:
        raise ValueError("weak-domain screen violates zero-cash contract")
    if int(curriculum.get("exact_holdout_prompt_overlap_count", -1)) != 0:
        raise ValueError("weak-domain screen overlaps a frozen holdout")

    budget = int(curriculum.get("target_training_tokens", -1))
    if budget not in SCREEN_REPLAY_EXAMPLES_BY_BUDGET:
        raise ValueError("weak-domain screen budget must be a predeclared tiny or medium budget")
    if curriculum.get("funnel_stage") != SCREEN_STAGE_BY_BUDGET[budget]:
        raise ValueError("weak-domain funnel stage does not match token budget")
    replay_examples = int(curriculum.get("replay_examples_per_domain", -1))
    if replay_examples != SCREEN_REPLAY_EXAMPLES_BY_BUDGET[budget]:
        raise ValueError("weak-domain replay count is outside the frozen screening contract")
    return curriculum


def train_screen(
    *,
    parent_checkpoint: str | Path,
    curriculum_lock: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    export_path: str | Path,
    run_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    curriculum = validate_screening_contract(curriculum_lock)
    budget = int(curriculum["target_training_tokens"])
    expected_replay = SCREEN_REPLAY_EXAMPLES_BY_BUDGET[budget]

    original = dict(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET)
    try:
        autonomous_training.REPLAY_EXAMPLES_BY_BUDGET[budget] = expected_replay
        result = autonomous_training.train_continuation(
            parent_checkpoint=parent_checkpoint,
            curriculum_lock=curriculum_lock,
            records_path=records_path,
            public_data=public_data,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            export_path=export_path,
            run_path=run_path,
            device=device,
        )
    finally:
        autonomous_training.REPLAY_EXAMPLES_BY_BUDGET.clear()
        autonomous_training.REPLAY_EXAMPLES_BY_BUDGET.update(original)

    if not isinstance(result, dict):
        raise ValueError("weak-domain trainer returned an invalid result")
    result = dict(result)
    result.update(
        {
            "research_funnel_version": FUNNEL_VERSION,
            "funnel_stage": curriculum["funnel_stage"],
            "screening_only": True,
            "promotion_authority": False,
            "cash_compute_cost_usd": 0.0,
        }
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(run_path).parent.mkdir(parents=True, exist_ok=True)
    Path(run_path).write_text(rendered, encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fail-closed weak-domain screening continuation.")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = train_screen(
        parent_checkpoint=args.parent_checkpoint,
        curriculum_lock=args.curriculum_lock,
        records_path=args.records,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        export_path=args.export,
        run_path=args.run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
