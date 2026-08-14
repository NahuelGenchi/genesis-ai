from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aligned_training import ALIGNED_TRAINING_POLICY_VERSION
from .scale_gate import decide_scale_promotion

ALIGNMENT_POLICY = "rolling_last_context_predict_final_position"


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the unchanged M6 scale gate to the generation-aligned candidate.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--candidate-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--ladder-result", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide_scale_promotion(
        candidate_checkpoint=args.candidate,
        training_run_path=args.training_run,
        reproduction_path=args.reproduction,
        baseline_domain_path=args.baseline_domain,
        candidate_domain_path=args.candidate_domain,
        baseline_m3_path=args.baseline_m3,
        candidate_m3_path=args.candidate_m3,
        ladder_result_path=args.ladder_result,
        curriculum_lock_path=args.curriculum_lock,
        expected_training_policy=ALIGNED_TRAINING_POLICY_VERSION,
        required_alignment_policy=ALIGNMENT_POLICY,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if not result["promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
