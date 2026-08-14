from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluate import evaluate_checkpoint
from .filtering import exact_fingerprint, iter_input_documents
from .generate import generate_text
from .ingest import sha256_file


class ContaminationError(RuntimeError):
    def __init__(self, result: dict[str, object]) -> None:
        super().__init__("blocking exact train/evaluation overlap detected")
        self.result = result


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_suite(path: Path) -> dict:
    suite = _load_json(path)
    required = {"suite_version", "validation", "generation", "contamination", "comparison"}
    if set(suite) != required:
        raise ValueError("evaluation suite has unexpected fields")
    if not isinstance(suite["suite_version"], str) or not suite["suite_version"]:
        raise ValueError("suite_version is required")
    return suite


def document_split(document_id: str, validation_fraction: float) -> str:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    threshold = int(validation_fraction * 10_000)
    bucket = int.from_bytes(hashlib.sha256(document_id.encode("utf-8")).digest()[:4], "big") % 10_000
    return "validation" if bucket < threshold else "train"


def check_exact_contamination(data_dir: Path, validation_fraction: float = 0.1) -> dict[str, object]:
    train: dict[str, list[str]] = {}
    validation: dict[str, list[str]] = {}
    train_documents = 0
    validation_documents = 0
    for document in iter_input_documents(data_dir):
        document_id = document.get("id")
        text = document.get("text")
        if not isinstance(document_id, str) or not isinstance(text, str):
            raise ValueError("filtered documents require id and text")
        fingerprint = exact_fingerprint(text)
        target = validation if document_split(document_id, validation_fraction) == "validation" else train
        target.setdefault(fingerprint, []).append(document_id)
        if target is validation:
            validation_documents += 1
        else:
            train_documents += 1

    overlap = sorted(set(train) & set(validation))
    examples = [
        {
            "fingerprint": fingerprint,
            "train_ids": train[fingerprint][:3],
            "validation_ids": validation[fingerprint][:3],
        }
        for fingerprint in overlap[:10]
    ]
    return {
        "method": "sha256-nfkc-collapsed-whitespace",
        "train_documents": train_documents,
        "validation_documents": validation_documents,
        "exact_overlap_count": len(overlap),
        "blocking": bool(overlap),
        "examples": examples,
    }


def run_suite(checkpoint: Path, data_dir: Path, suite_path: Path, device: str = "cpu") -> dict[str, object]:
    suite = load_suite(suite_path)
    contamination_config = suite["contamination"]
    if not isinstance(contamination_config, dict):
        raise ValueError("contamination config must be an object")
    validation_fraction = float(contamination_config.get("validation_fraction", 0.1))
    contamination = check_exact_contamination(data_dir, validation_fraction)

    base: dict[str, object] = {
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(suite_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "contamination": contamination,
    }
    if contamination["blocking"] and contamination_config.get("block_exact_overlap") is True:
        raise ContaminationError(base)

    validation = suite["validation"]
    if not isinstance(validation, dict):
        raise ValueError("validation config must be an object")
    evaluation = evaluate_checkpoint(
        str(checkpoint),
        str(data_dir),
        batch_size=int(validation["batch_size"]),
        batches=int(validation["batches"]),
        split=str(validation["split"]),
        device=device,
    )

    generation = suite["generation"]
    if not isinstance(generation, dict) or not isinstance(generation.get("prompts"), list):
        raise ValueError("generation config is invalid")
    generations: list[dict[str, object]] = []
    for prompt in generation["prompts"]:
        if not isinstance(prompt, str):
            raise ValueError("generation prompt must be a string")
        text = generate_text(
            str(checkpoint),
            prompt,
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_k=int(generation["top_k"]),
            seed=int(generation["seed"]),
            device=device,
        )
        generations.append({
            "prompt": prompt,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })

    base.update({
        "evaluation": evaluation,
        "generations": generations,
        "primary_metric": {
            "name": "validation_loss",
            "value": evaluation["loss"],
            "lower_is_better": True,
        },
    })
    return base


def compare_results(incumbent: dict, candidate: dict) -> dict[str, object]:
    for field in ("suite_version", "suite_sha256", "data_manifest_sha256"):
        if incumbent.get(field) != candidate.get(field):
            raise ValueError(f"cannot compare different {field}")
    incumbent_metric = incumbent.get("primary_metric")
    candidate_metric = candidate.get("primary_metric")
    if not isinstance(incumbent_metric, dict) or not isinstance(candidate_metric, dict):
        raise ValueError("primary_metric missing")
    if incumbent_metric.get("name") != candidate_metric.get("name"):
        raise ValueError("primary metrics differ")
    before = float(incumbent_metric["value"])
    after = float(candidate_metric["value"])
    lower_is_better = bool(candidate_metric.get("lower_is_better"))
    delta = after - before
    if after == before:
        winner = "tie"
    elif (after < before) == lower_is_better:
        winner = "candidate"
    else:
        winner = "incumbent"
    return {
        "suite_version": incumbent["suite_version"],
        "metric": incumbent_metric["name"],
        "incumbent": before,
        "candidate": after,
        "delta": delta,
        "winner": winner,
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a versioned Genesis AI evaluation suite.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = run_suite(args.checkpoint, args.data, args.suite, args.device)
    except ContaminationError as exc:
        _write(args.output, exc.result)
        raise SystemExit(str(exc)) from exc
    _write(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
