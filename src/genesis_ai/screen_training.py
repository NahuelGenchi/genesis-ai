from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import autonomous_training
from .research_funnel import FUNNEL_VERSION, STAGES

SCREEN_TRAINING_VERSION = "weak-domain-screen-training-v1"
SCREEN_BUDGETS = {
    STAGES["tiny"].target_training_tokens: STAGES["tiny"].replay_examples_per_domain,
    STAGES["medium"].target_training_tokens: STAGES["medium"].replay_examples_per_domain,
}


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    curriculum = _load(curriculum_lock)
    if curriculum.get("funnel_version") != FUNNEL_VERSION:
        raise ValueError("screen curriculum is not bound to the resource-efficient funnel")
    if curriculum.get("screening_only") is not True:
        raise ValueError("screen curriculum must be screening-only")
    if curriculum.get("promotion_eligible") is not False or curriculum.get("promotion_authority") is not False:
        raise ValueError("screen curriculum must have zero promotion authority")
    budget = curriculum.get("target_training_tokens")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget not in SCREEN_BUDGETS:
        raise ValueError("screen training budget is outside the tiny/medium allow-list")
    expected_replay = SCREEN_BUDGETS[budget]
    if int(curriculum.get("replay_examples_per_domain", -1)) != expected_replay:
        raise ValueError("screen replay count does not match the frozen stage budget")

    # Reuse the already-tested deterministic continuation implementation while
    # extending its in-process budget allow-list only for this non-promoting call.
    # The canonical trainer module remains unchanged for normal promotion cycles.
    previous = dict(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET)
    try:
        autonomous_training.REPLAY_EXAMPLES_BY_BUDGET.update(SCREEN_BUDGETS)
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
        autonomous_training.REPLAY_EXAMPLES_BY_BUDGET.update(previous)

    result = dict(result)
    result.update(
        {
            "screen_training_version": SCREEN_TRAINING_VERSION,
            "funnel_version": FUNNEL_VERSION,
            "screening_only": True,
            "promotion_eligible": False,
            "promotion_authority": False,
            "variant_id": curriculum["variant_id"],
            "stage": curriculum["stage"],
        }
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(run_path).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one deterministic non-promoting weak-domain screen.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_screen(
        parent_checkpoint=args.parent,
        curriculum_lock=args.curriculum_lock,
        records_path=args.records,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        export_path=args.export,
        run_path=args.run,
        device=args.device,
    )


if __name__ == "__main__":
    main()
