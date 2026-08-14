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


def build_scale_model_card(
    *,
    checkpoint_path: str | Path,
    training_run_path: str | Path,
    domain_result_path: str | Path,
    baseline_domain_path: str | Path,
    baseline_m3_path: str | Path,
    candidate_m3_path: str | Path,
    gate_path: str | Path,
    curriculum_lock_path: str | Path,
    model_name: str = MODEL_NAME,
    alignment_result_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ValueError("model_name must be a non-empty path-safe name")
    checkpoint_path = Path(checkpoint_path)
    training = _load_json(training_run_path)
    domain = _load_json(domain_result_path)
    baseline_domain = _load_json(baseline_domain_path)
    baseline_m3 = _load_json(baseline_m3_path)
    candidate_m3 = _load_json(candidate_m3_path)
    gate = _load_json(gate_path)
    curriculum = _load_json(curriculum_lock_path)
    alignment = _load_json(alignment_result_path) if alignment_result_path is not None else None
    checkpoint_hash = sha256_file(checkpoint_path)
    if gate.get("promoted") is not True or gate.get("decision") != "promote":
        raise ValueError("model card may only be built for a promoted M6 checkpoint")
    if gate.get("candidate_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("gate does not match checkpoint")
    if training.get("inference_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("training record does not match checkpoint")
    if alignment is not None and alignment.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("alignment result does not match checkpoint")

    baseline_code = baseline_domain["domains"]["code"]
    candidate_code = domain["domains"]["code"]
    baseline_loss = float(baseline_m3["primary_metric"]["value"])
    candidate_loss = float(candidate_m3["primary_metric"]["value"])
    record: dict[str, Any] = {
        "format_version": "1.0",
        "model": model_name,
        "checkpoint": {"sha256": checkpoint_hash, "size_bytes": checkpoint_path.stat().st_size},
        "architecture": training["architecture"],
        "parameter_count": training["parameter_count"],
        "training": {
            "policy": training["training_policy"],
            "steps": training["steps"],
            "processed_tokens": training["processed_tokens"],
            "procedural_step_fraction": training["procedural_step_fraction"],
            "public_step_fraction": training["public_step_fraction"],
            "seed": training["seed"],
            "cash_compute_cost_usd": training["cash_compute_cost_usd"],
            "curriculum_lock_sha256": training["inputs"]["curriculum_lock_sha256"],
        },
        "capability": {
            "domain": "restricted integer-expression synthesis",
            "suite": domain["suite_version"],
            "task_count": candidate_code["task_count"],
            "baseline_exact_accuracy": baseline_code["exact_accuracy"],
            "candidate_exact_accuracy": candidate_code["exact_accuracy"],
            "absolute_gain": float(candidate_code["exact_accuracy"]) - float(baseline_code["exact_accuracy"]),
            "legacy_end_anchored_oracle_target_loss": candidate_code["oracle_target_loss"],
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
            "The demonstrated capability is restricted integer-expression synthesis at difficulty 1.",
            "General-language training data remains extremely small.",
            "No instruction tuning, preference tuning, tool-use training, safety tuning, or broad factual training.",
            "Success on the frozen 60-task holdout does not imply broad coding ability.",
        ],
    }
    if alignment is not None:
        rolling = alignment.get("generation_aligned_rolling")
        if not isinstance(rolling, dict):
            raise ValueError("alignment result is missing generation_aligned_rolling")
        record["generation_alignment"] = {
            "diagnostic_version": alignment.get("diagnostic_version"),
            "rolling_loss": rolling.get("mean_loss"),
            "greedy_token_accuracy": rolling.get("greedy_token_accuracy"),
            "first_token_greedy_correct_rate": rolling.get("first_token_greedy_correct_rate"),
            "all_greedy_tokens_correct_rate": rolling.get("all_greedy_tokens_correct_rate"),
        }

    limitations = "\n".join(f"- {item}" for item in record["limitations"])
    alignment_markdown = ""
    if "generation_alignment" in record:
        aligned = record["generation_alignment"]
        alignment_markdown = f"""
## Generation-aligned diagnostics
- Rolling teacher-forced loss: {aligned['rolling_loss']:.6f}
- Rolling greedy token accuracy: {aligned['greedy_token_accuracy']:.2%}
- First-answer-token greedy accuracy: {aligned['first_token_greedy_correct_rate']:.2%}
- Entire oracle sequence greedy-correct under rolling teacher forcing: {aligned['all_greedy_tokens_correct_rate']:.2%}
"""
    markdown = f"""# {model_name}

## Status
**Promoted M6 useful-domain checkpoint.** This is still a research model, not a general assistant.

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
- Processed tokens: {record['training']['processed_tokens']:,}
- Steps: {record['training']['steps']}
- Mix: {record['training']['procedural_step_fraction']:.0%} procedural / {record['training']['public_step_fraction']:.0%} public-domain text
- Seed: {record['training']['seed']}
- Required cash compute: ${record['training']['cash_compute_cost_usd']:.2f}

## Demonstrated capability
Frozen domain: **restricted integer-expression synthesis**.

- Exact accuracy: {record['capability']['baseline_exact_accuracy']:.2%} → **{record['capability']['candidate_exact_accuracy']:.2%}**
- Absolute gain: **{record['capability']['absolute_gain']:.2%}**
- Legacy end-anchored oracle-target loss: {record['capability']['legacy_end_anchored_oracle_target_loss']:.6f}
- Holdout tasks: {record['capability']['task_count']}
{alignment_markdown}
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
    parser = argparse.ArgumentParser(description="Build a promoted Genesis micro-2m model card.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--domain-result", type=Path, required=True)
    parser.add_argument("--baseline-domain", type=Path, required=True)
    parser.add_argument("--baseline-m3", type=Path, required=True)
    parser.add_argument("--candidate-m3", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--alignment-result", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record, markdown = build_scale_model_card(
        checkpoint_path=args.checkpoint,
        training_run_path=args.training_run,
        domain_result_path=args.domain_result,
        baseline_domain_path=args.baseline_domain,
        baseline_m3_path=args.baseline_m3,
        candidate_m3_path=args.candidate_m3,
        gate_path=args.gate,
        curriculum_lock_path=args.curriculum_lock,
        model_name=args.model_name,
        alignment_result_path=args.alignment_result,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (args.output_dir / "MODEL_CARD.md").write_text(markdown, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
