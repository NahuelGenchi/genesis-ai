from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ingest import sha256_file

MODEL_NAME = "genesis-micro-2m-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_terminated_model_card(
    *,
    checkpoint_path: str | Path,
    training_run_path: str | Path,
    baseline_domain_path: str | Path,
    candidate_domain_path: str | Path,
    baseline_m3_path: str | Path,
    candidate_m3_path: str | Path,
    gate_path: str | Path,
    curriculum_lock_path: str | Path,
) -> tuple[dict[str, Any], str]:
    checkpoint_path = Path(checkpoint_path)
    training = _load_json(training_run_path)
    baseline_domain = _load_json(baseline_domain_path)
    candidate_domain = _load_json(candidate_domain_path)
    baseline_m3 = _load_json(baseline_m3_path)
    candidate_m3 = _load_json(candidate_m3_path)
    gate = _load_json(gate_path)
    curriculum = _load_json(curriculum_lock_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if gate.get("promoted") is not True or gate.get("decision") != "promote":
        raise ValueError("model card may only be built for a promoted terminated M6 checkpoint")
    if training.get("inference_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("training record does not match checkpoint")
    if candidate_domain.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("candidate domain result does not match checkpoint")
    if candidate_m3.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("candidate M3 result does not match checkpoint")
    if gate.get("candidate_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("gate does not match checkpoint")

    baseline_code = baseline_domain["domains"]["code"]
    candidate_code = candidate_domain["domains"]["code"]
    baseline_loss = float(baseline_m3["primary_metric"]["value"])
    candidate_loss = float(candidate_m3["primary_metric"]["value"])
    record: dict[str, Any] = {
        "format_version": "1.0",
        "model": MODEL_NAME,
        "checkpoint": {"sha256": checkpoint_hash, "size_bytes": checkpoint_path.stat().st_size},
        "architecture": training["architecture"],
        "parameter_count": training["parameter_count"],
        "training": {
            "policy": training["training_policy"],
            "dataset_version": training["dataset_version"],
            "termination_delimiter": training["termination_delimiter"],
            "steps": training["steps"],
            "processed_tokens": training["processed_tokens"],
            "procedural_step_fraction": training["procedural_step_fraction"],
            "public_step_fraction": training["public_step_fraction"],
            "seed": training["seed"],
            "cash_compute_cost_usd": training["cash_compute_cost_usd"],
            "first_response_target_coverage": training["termination"]["first_response_target_coverage"],
            "terminator_target_coverage": training["termination"]["terminator_target_coverage"],
            "schedule_target_updates": training["termination"]["schedule_target_updates"],
        },
        "capability": {
            "domain": "restricted integer-expression synthesis",
            "suite": candidate_domain["suite_version"],
            "task_count": candidate_code["task_count"],
            "baseline_exact_accuracy": baseline_code["exact_accuracy"],
            "candidate_exact_accuracy": candidate_code["exact_accuracy"],
            "absolute_gain": float(candidate_code["exact_accuracy"]) - float(baseline_code["exact_accuracy"]),
            "baseline_termination_rate": baseline_code["termination_rate"],
            "candidate_termination_rate": candidate_code["termination_rate"],
            "candidate_terminated_oracle_loss": candidate_code["terminated_oracle_loss"],
        },
        "general_language": {
            "suite": candidate_m3["suite_version"],
            "baseline_validation_loss": baseline_loss,
            "candidate_validation_loss": candidate_loss,
            "regression_fraction": (candidate_loss - baseline_loss) / baseline_loss,
            "exact_contamination_count": candidate_m3["contamination"]["exact_overlap_count"],
        },
        "curriculum": {
            "version": curriculum["curriculum_version"],
            "procedural_examples": curriculum["training"]["procedural"]["examples"],
            "public_documents": curriculum["public_text"]["document_count"],
            "public_tokens": curriculum["public_text"]["token_count"],
            "holdout_prompt_overlap": curriculum["evaluation_separation"]["exact_prompt_overlap_count"],
        },
        "promotion": {"gate_version": gate["gate_version"], "decision_sha256": gate["decision_sha256"]},
        "limitations": [
            "Useful-domain research checkpoint, not a general-purpose assistant.",
            "Demonstrated capability is restricted integer-expression synthesis at difficulty 1.",
            "The answer-stop protocol is a generated newline, not a dedicated learned EOS token.",
            "General-language training data remains extremely small.",
            "No preference tuning, tool-use training, safety tuning, or broad factual training.",
            "Success on the frozen 60-task holdout does not imply broad coding ability.",
        ],
    }
    limitations = "\n".join(f"- {item}" for item in record["limitations"])
    markdown = f"""# {MODEL_NAME}

## Status
**Promoted M6 useful-domain checkpoint.** Research model only; not a general assistant.

## Architecture
- Parameters: {record['parameter_count']:,}
- Context: {record['architecture']['context_length']}
- Width: {record['architecture']['d_model']}
- Heads: {record['architecture']['n_heads']}
- Layers: {record['architecture']['n_layers']}
- FFN: {record['architecture']['d_ff']}
- Position encoding: {record['architecture']['position_encoding']}

## Training
- Policy: `{record['training']['policy']}`
- Response dataset: `{record['training']['dataset_version']}`
- Answer terminator: generated newline
- Processed tokens: {record['training']['processed_tokens']:,}
- Steps: {record['training']['steps']}
- Mix: {record['training']['procedural_step_fraction']:.0%} procedural / {record['training']['public_step_fraction']:.0%} public-domain
- First-response-target coverage: {record['training']['first_response_target_coverage']:.0%}
- Terminator-target coverage: {record['training']['terminator_target_coverage']:.0%}
- Generation-aligned target updates: {record['training']['schedule_target_updates']:,}
- Seed: {record['training']['seed']}
- Required cash compute: ${record['training']['cash_compute_cost_usd']:.2f}

## Demonstrated capability
Frozen stop-aware domain: **restricted integer-expression synthesis**.

- Strict stopped exact accuracy: {record['capability']['baseline_exact_accuracy']:.2%} → **{record['capability']['candidate_exact_accuracy']:.2%}**
- Absolute gain: **{record['capability']['absolute_gain']:.2%}**
- Candidate termination rate: {record['capability']['candidate_termination_rate']:.2%}
- Candidate terminated-oracle loss: {record['capability']['candidate_terminated_oracle_loss']:.6f}
- Holdout tasks: {record['capability']['task_count']}

## General-language regression gate
- M3 validation loss: {record['general_language']['baseline_validation_loss']:.6f} → {record['general_language']['candidate_validation_loss']:.6f}
- Regression fraction: {record['general_language']['regression_fraction']:.2%}
- Exact contamination overlap: {record['general_language']['exact_contamination_count']}

## Curriculum
- Procedural examples: {record['curriculum']['procedural_examples']:,}
- Public-domain documents: {record['curriculum']['public_documents']:,}
- Public-domain tokens: {record['curriculum']['public_tokens']:,}
- Frozen holdout prompt overlap: {record['curriculum']['holdout_prompt_overlap']}

## Checkpoint
- SHA-256: `{checkpoint_hash}`
- Bytes: {record['checkpoint']['size_bytes']:,}

## Limitations
{limitations}
"""
    return record, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the promoted newline-terminated M6 model card.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--candidate-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record, markdown = build_terminated_model_card(
        checkpoint_path=args.checkpoint,
        training_run_path=args.training_run,
        baseline_domain_path=args.baseline_domain,
        candidate_domain_path=args.candidate_domain,
        baseline_m3_path=args.baseline_m3,
        candidate_m3_path=args.candidate_m3,
        gate_path=args.gate,
        curriculum_lock_path=args.curriculum_lock,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (args.output_dir / "MODEL_CARD.md").write_text(markdown, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
