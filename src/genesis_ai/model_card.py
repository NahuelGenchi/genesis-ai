from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import sha256_file

MODEL_NAME = "genesis-tiny-v0"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_model_record(
    *,
    run_path: Path,
    evaluation_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    source_lock_path: Path,
    sample_path: Path,
) -> tuple[dict[str, object], str]:
    run = _load_json(run_path)
    evaluation = _load_json(evaluation_path)
    tokenizer = _load_json(tokenizer_path)
    source_lock = _load_json(source_lock_path)
    sample = sample_path.read_text(encoding="utf-8", errors="replace").strip()

    training = tokenizer.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("tokenizer training metadata missing")
    sources = source_lock.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("source lock sources missing")

    record: dict[str, object] = {
        "format_version": "1.0",
        "model": MODEL_NAME,
        "checkpoint": {
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "architecture": run.get("config"),
        "parameter_count": run.get("parameter_count"),
        "training": {
            "steps": run.get("steps"),
            "resumed_from_step": run.get("resumed_from_step"),
            "batch_size": run.get("batch_size"),
            "learning_rate": run.get("learning_rate"),
            "train_documents": run.get("train_documents"),
            "train_tokens": run.get("train_tokens"),
            "probe_loss_before": run.get("probe_loss_before"),
            "probe_loss_after": run.get("probe_loss_after"),
            "probe_loss_decreased": run.get("probe_loss_decreased"),
            "last_training_loss": run.get("last_training_loss"),
            "elapsed_seconds": run.get("elapsed_seconds"),
            "data_manifest_sha256": run.get("data_manifest_sha256"),
        },
        "evaluation": evaluation,
        "tokenizer": {
            "name": "genesis-v0",
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.get("vocab_size"),
            "bytes_per_token": training.get("bytes_per_token"),
            "source_count": training.get("corpus_source_count"),
        },
        "sources": [
            {
                "id": source.get("id"),
                "ebook_id": source.get("ebook_id"),
                "language": source.get("language"),
                "sample_sha256": source.get("sample_sha256"),
            }
            for source in sources
            if isinstance(source, dict)
        ],
        "generation_sample": sample,
        "limitations": [
            "Pipeline-validation model; not a useful general assistant.",
            "Training corpus is tiny and literature-only.",
            "Training languages are limited to English, Spanish, and French source samples.",
            "No instruction tuning, preference tuning, tool training, safety tuning, or factuality training.",
            "Validation is document-disjoint but comes from the same small source pool.",
            "Generated text may be incoherent or malformed.",
        ],
    }

    architecture = record["architecture"] if isinstance(record["architecture"], dict) else {}
    train = record["training"] if isinstance(record["training"], dict) else {}
    eval_record = record["evaluation"] if isinstance(record["evaluation"], dict) else {}
    tokenizer_record = record["tokenizer"] if isinstance(record["tokenizer"], dict) else {}
    source_lines = "\n".join(
        f"- `{source['id']}` — language `{source['language']}` — sample SHA `{source['sample_sha256']}`"
        for source in record["sources"]
        if isinstance(source, dict)
    )
    limitations = "\n".join(f"- {item}" for item in record["limitations"])
    markdown = f"""# {MODEL_NAME}

## Purpose
First end-to-end model trained from random weights. **Pipeline baseline only.**

## Architecture
- Parameters: {record['parameter_count']:,}
- Vocabulary: {architecture.get('vocab_size')}
- Context: {architecture.get('context_length')}
- Width: {architecture.get('d_model')}
- Heads: {architecture.get('n_heads')}
- Layers: {architecture.get('n_layers')}
- FFN: {architecture.get('d_ff')}

## Training
- Steps: {train.get('steps')}
- Intentional resume point: {train.get('resumed_from_step')}
- Train documents: {train.get('train_documents')}
- Train tokens: {train.get('train_tokens')}
- Probe loss: {train.get('probe_loss_before')} → {train.get('probe_loss_after')}
- Last training loss: {train.get('last_training_loss')}
- CPU time: {train.get('elapsed_seconds')} s

## Evaluation
- Split: {eval_record.get('split')}
- Documents: {eval_record.get('documents')}
- Tokens: {eval_record.get('tokens')}
- Loss: {eval_record.get('loss')}
- Perplexity: {eval_record.get('perplexity')}

## Tokenizer
- `genesis-v0`
- Vocabulary: {tokenizer_record.get('vocab_size')}
- Bytes/token: {tokenizer_record.get('bytes_per_token')}

## Training sources
{source_lines}

## Checkpoint
- SHA-256: `{record['checkpoint']['sha256']}`
- Bytes: {record['checkpoint']['size_bytes']}

## Seeded sample
```text
{sample}
```

## Limitations
{limitations}
"""
    return record, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the auditable Genesis tiny baseline model card.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record, markdown = build_model_record(
        run_path=args.run,
        evaluation_path=args.evaluation,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        source_lock_path=args.source_lock,
        sample_path=args.sample,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "MODEL_CARD.md").write_text(markdown, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
