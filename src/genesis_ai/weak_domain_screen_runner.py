from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .research_funnel import VARIANTS, rank_screen_directory, stage_config
from .screen_training import train_screen
from .terminated_eval import run_terminated_selection
from .weak_domain_curriculum import build_curriculum


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _variant_ids(stage: str, prior_summary: Path | None) -> list[str]:
    if stage == "tiny":
        if prior_summary is not None:
            raise ValueError("tiny stage must not consume a prior summary")
        return [str(item["id"]) for item in VARIANTS]
    if stage != "medium" or prior_summary is None:
        raise ValueError("medium stage requires the tiny-stage summary")
    summary = _load(prior_summary)
    if summary.get("stage") != "tiny" or summary.get("promotion_authority") is not False:
        raise ValueError("medium screen requires a non-promoting tiny-stage summary")
    survivors = summary.get("survivor_ids")
    if not isinstance(survivors, list) or not 1 <= len(survivors) <= 3:
        raise ValueError("tiny-stage summary has no valid survivor set")
    return [str(value) for value in survivors]


def run_stage(
    *,
    parent_checkpoint: Path,
    train_suite: Path,
    dev_suite: Path,
    holdout_suites: list[Path],
    tokenizer_path: Path,
    public_data: Path,
    stage: str,
    work_root: Path,
    baseline_dev: Path,
    summary_output: Path,
    prior_summary: Path | None = None,
) -> dict[str, Any]:
    config = stage_config(stage)
    if stage not in {"tiny", "medium"}:
        raise ValueError("screen runner only executes tiny/medium non-promoting stages")
    variant_ids = _variant_ids(stage, prior_summary)
    stage_root = work_root / stage
    stage_root.mkdir(parents=True, exist_ok=True)

    for variant_id in variant_ids:
        variant_root = stage_root / variant_id
        variant_root.mkdir(parents=True, exist_ok=True)
        records = variant_root / "records.jsonl"
        plan = variant_root / "plan.json"
        curriculum_path = variant_root / "curriculum.json"
        curriculum = build_curriculum(
            parent_checkpoint=parent_checkpoint,
            suite_path=train_suite,
            holdout_suites=holdout_suites,
            tokenizer_path=tokenizer_path,
            public_data=public_data,
            variant_id=variant_id,
            stage=stage,
            records_path=records,
            plan_path=plan,
        )
        if curriculum.get("screening_only") is not True or curriculum.get("promotion_authority") is not False:
            raise ValueError("screen curriculum unexpectedly acquired promotion authority")
        _write(curriculum_path, curriculum)

        training_checkpoint = variant_root / "training.pt"
        candidate = variant_root / "candidate.pt"
        training_path = variant_root / "training.json"
        started = time.monotonic()
        training = train_screen(
            parent_checkpoint=parent_checkpoint,
            curriculum_lock=curriculum_path,
            records_path=records,
            public_data=public_data,
            tokenizer_path=tokenizer_path,
            checkpoint_path=training_checkpoint,
            export_path=candidate,
            run_path=training_path,
            device="cpu",
        )
        elapsed = time.monotonic() - started
        training = dict(training)
        training["wall_time_seconds"] = elapsed
        training["quality_measurement_scope"] = "development-screen-only"
        _write(training_path, training)

        evaluation = run_terminated_selection(checkpoint=candidate, suite_path=dev_suite, device="cpu")
        evaluation["promotion_authority"] = False
        evaluation["development_screen_only"] = True
        _write(variant_root / "evaluation.json", evaluation)

        # Screening checkpoints are intentionally ephemeral. Only aggregate
        # evidence survives; a winning full candidate is retrained under the
        # promotion-eligible frozen contract and independently reproduced.
        training_checkpoint.unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)

    summary = rank_screen_directory(
        baseline_path=baseline_dev,
        candidates_root=stage_root,
        stage=stage,
    )
    summary["wall_time_seconds_total"] = sum(
        float(_load(stage_root / variant_id / "training.json").get("wall_time_seconds", 0.0))
        for variant_id in variant_ids
    )
    summary["processed_tokens_total"] = config.target_training_tokens * len(variant_ids)
    _write(summary_output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one non-promoting weak-domain successive-halving stage.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--train-suite", type=Path, required=True)
    parser.add_argument("--dev-suite", type=Path, required=True)
    parser.add_argument("--holdout-suites", nargs="+", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--stage", choices=("tiny", "medium"), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--baseline-dev", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path)
    args = parser.parse_args()
    result = run_stage(
        parent_checkpoint=args.parent,
        train_suite=args.train_suite,
        dev_suite=args.dev_suite,
        holdout_suites=list(args.holdout_suites),
        tokenizer_path=args.tokenizer,
        public_data=args.public_data,
        stage=args.stage,
        work_root=args.work_root,
        baseline_dev=args.baseline_dev,
        summary_output=args.summary,
        prior_summary=args.prior_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
