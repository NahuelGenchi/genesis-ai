from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from .challenger import build_task
from .domain_selection import generate_domain_tasks, load_suite, oracle_response
from .filtering import iter_input_documents
from .ingest import sha256_file
from .tokenizer import ByteBPETokenizer

CURRICULUM_VERSION = "m6-code-curriculum-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = _load_json(path)
    required = {
        "format_version",
        "curriculum_version",
        "selected_domain",
        "required_selection_suite_version",
        "required_selection_suite_sha256",
        "evaluation_base_seed",
        "training_seed",
        "procedural_examples",
        "difficulty",
        "context_length",
        "procedural_batch_fraction",
        "public_text_batch_fraction",
        "target_training_tokens",
        "tokenizer_sha256",
    }
    if set(spec) != required:
        raise ValueError("invalid curriculum spec fields")
    if spec.get("format_version") != "1.0" or spec.get("curriculum_version") != CURRICULUM_VERSION:
        raise ValueError("unsupported curriculum spec")
    if spec.get("selected_domain") != "code":
        raise ValueError("M6 curriculum v1 is locked to selected domain code")
    for field in ("evaluation_base_seed", "training_seed", "procedural_examples", "difficulty", "context_length", "target_training_tokens"):
        value = spec[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
    if spec["procedural_examples"] <= 0 or spec["context_length"] <= 0 or spec["target_training_tokens"] <= 0:
        raise ValueError("curriculum counts must be positive")
    if not 1 <= spec["difficulty"] <= 5:
        raise ValueError("difficulty must be in [1,5]")
    if spec["training_seed"] == spec["evaluation_base_seed"]:
        raise ValueError("training and evaluation base seeds must differ")
    for field in ("procedural_batch_fraction", "public_text_batch_fraction"):
        value = spec[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be in [0,1]")
    total_fraction = float(spec["procedural_batch_fraction"]) + float(spec["public_text_batch_fraction"])
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("curriculum batch fractions must sum to 1")
    for field in ("required_selection_suite_sha256", "tokenizer_sha256"):
        value = spec[field]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{field} must be lowercase SHA-256")
    if not isinstance(spec["required_selection_suite_version"], str) or not spec["required_selection_suite_version"]:
        raise ValueError("required_selection_suite_version is required")
    return spec


def _evaluation_code_tasks(spec: dict[str, Any], evaluation_suite_path: Path) -> tuple[list[dict[str, Any]], int]:
    suite = load_suite(evaluation_suite_path)
    if suite["suite_version"] != spec["required_selection_suite_version"]:
        raise ValueError("evaluation suite version does not match curriculum")
    if sha256_file(evaluation_suite_path) != spec["required_selection_suite_sha256"]:
        raise ValueError("evaluation suite hash does not match curriculum")
    if int(suite["base_seed"]) != spec["evaluation_base_seed"]:
        raise ValueError("evaluation base seed does not match curriculum")
    try:
        ordinal = list(suite["domains"]).index("code")
    except ValueError as exc:
        raise ValueError("evaluation suite does not contain code domain") from exc
    domain_seed = int(suite["base_seed"]) + ordinal
    if domain_seed == spec["training_seed"]:
        raise ValueError("training seed collides with evaluation domain seed")
    tasks = generate_domain_tasks(
        domain="code",
        seed=domain_seed,
        count=int(suite["tasks_per_domain"]),
        difficulty=int(suite["difficulty"]),
    )
    return tasks, domain_seed


def generate_procedural_records(
    spec: dict[str, Any],
    evaluation_tasks: list[dict[str, Any]],
    tokenizer: ByteBPETokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation_prompt_hashes = {_sha256_text(str(task["prompt"])) for task in evaluation_tasks}
    rng = random.Random(int(spec["training_seed"]))
    desired = int(spec["procedural_examples"])
    context_length = int(spec["context_length"])
    records: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    response_token_count = 0
    prompt_response_token_count = 0
    max_prompt_response_tokens = 0
    attempts = 0
    max_attempts = desired * 200

    while len(records) < desired and attempts < max_attempts:
        attempts += 1
        task = build_task(rng, "code", int(spec["difficulty"]))
        prompt = str(task["prompt"])
        prompt_hash = _sha256_text(prompt)
        if prompt_hash in evaluation_prompt_hashes or prompt_hash in seen_prompt_hashes:
            continue
        response = oracle_response(task)
        prompt_ids = tokenizer.encode(prompt + "\nAnswer:")
        response_ids = tokenizer.encode(response)
        if not prompt_ids or not response_ids:
            raise ValueError("procedural example tokenized to an empty sequence")
        sequence_tokens = len(prompt_ids) + len(response_ids)
        if sequence_tokens > context_length:
            raise ValueError(
                f"procedural example exceeds context: {sequence_tokens} > {context_length} for {task['id']}"
            )
        seen_prompt_hashes.add(prompt_hash)
        response_token_count += len(response_ids)
        prompt_response_token_count += sequence_tokens
        max_prompt_response_tokens = max(max_prompt_response_tokens, sequence_tokens)
        record_base = {
            "format_version": "1.0",
            "curriculum": CURRICULUM_VERSION,
            "domain": "code",
            "difficulty": int(spec["difficulty"]),
            "prompt": prompt,
            "response": response,
            "source_task_id": task["id"],
            "provenance": {
                "kind": "procedural_oracle",
                "generator": task["generator"],
                "training_seed": int(spec["training_seed"]),
                "ordinal": len(records),
                "attempt": attempts,
            },
        }
        record = {"id": f"curr-{_sha256_object(record_base)[:20]}", **record_base}
        records.append(record)

    if len(records) != desired:
        raise RuntimeError(f"curriculum generation exhausted after {attempts} attempts")

    overlap = seen_prompt_hashes & evaluation_prompt_hashes
    if overlap:
        raise ValueError("blocking exact prompt overlap with frozen evaluation")

    summary = {
        "examples": len(records),
        "generation_attempts": attempts,
        "record_set_sha256": _sha256_object(records),
        "prompt_set_sha256": _sha256_object(sorted(seen_prompt_hashes)),
        "response_set_sha256": _sha256_object([record["response"] for record in records]),
        "response_tokens": response_token_count,
        "prompt_response_tokens": prompt_response_token_count,
        "max_prompt_response_tokens": max_prompt_response_tokens,
    }
    return records, summary


def summarize_public_text(
    public_data: Path,
    tokenizer: ByteBPETokenizer,
    *,
    source_lock: Path,
    source_catalog: Path,
) -> dict[str, Any]:
    identities: list[dict[str, str]] = []
    document_count = 0
    token_count = 0
    utf8_bytes = 0
    for document in iter_input_documents(public_data):
        document_id = document.get("id")
        text = document.get("text")
        if not isinstance(document_id, str) or not isinstance(text, str):
            raise ValueError("public corpus document requires id and text")
        document_count += 1
        token_count += len(tokenizer.encode(text))
        utf8_bytes += len(text.encode("utf-8"))
        identities.append({"id": document_id, "text_sha256": _sha256_text(text)})
    if document_count <= 0 or token_count <= 0:
        raise ValueError("public corpus is empty")
    return {
        "document_count": document_count,
        "token_count": token_count,
        "utf8_bytes": utf8_bytes,
        "manifest_sha256": sha256_file(public_data / "manifest.json"),
        "content_set_sha256": _sha256_object(identities),
        "source_lock_sha256": sha256_file(source_lock),
        "source_catalog_sha256": sha256_file(source_catalog),
    }


def build_curriculum(
    *,
    spec_path: str | Path,
    selection_result_path: str | Path,
    evaluation_suite_path: str | Path,
    tokenizer_path: str | Path,
    public_data: str | Path,
    public_source_lock: str | Path,
    public_source_catalog: str | Path,
    records_path: str | Path,
) -> dict[str, Any]:
    spec_path = Path(spec_path)
    selection_result_path = Path(selection_result_path)
    evaluation_suite_path = Path(evaluation_suite_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    public_source_lock = Path(public_source_lock)
    public_source_catalog = Path(public_source_catalog)
    records_path = Path(records_path)

    spec = load_spec(spec_path)
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != spec["tokenizer_sha256"]:
        raise ValueError("tokenizer hash does not match curriculum")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)

    selection = _load_json(selection_result_path)
    if selection.get("suite_version") != spec["required_selection_suite_version"]:
        raise ValueError("selection result suite version does not match curriculum")
    if selection.get("suite_sha256") != spec["required_selection_suite_sha256"]:
        raise ValueError("selection result suite hash does not match curriculum")
    if selection.get("selected_domain") != spec["selected_domain"]:
        raise ValueError("selection result does not select code")
    domains = selection.get("domains")
    if not isinstance(domains, dict) or not isinstance(domains.get("code"), dict):
        raise ValueError("selection result is missing code-domain metrics")

    evaluation_tasks, evaluation_domain_seed = _evaluation_code_tasks(spec, evaluation_suite_path)
    evaluation_task_hash = _sha256_object(evaluation_tasks)
    if domains["code"].get("task_set_sha256") != evaluation_task_hash:
        raise ValueError("reconstructed code holdout does not match frozen selection result")

    records, procedural = generate_procedural_records(spec, evaluation_tasks, tokenizer)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical(record) + "\n")
    procedural["records_file_sha256"] = sha256_file(records_path)
    procedural["records_file_size_bytes"] = records_path.stat().st_size

    evaluation_prompt_hashes = sorted(_sha256_text(str(task["prompt"])) for task in evaluation_tasks)
    training_prompt_hashes = {_sha256_text(str(record["prompt"])) for record in records}
    exact_prompt_overlap_count = len(training_prompt_hashes & set(evaluation_prompt_hashes))
    if exact_prompt_overlap_count:
        raise ValueError("blocking curriculum/evaluation prompt overlap")

    public_summary = summarize_public_text(
        public_data,
        tokenizer,
        source_lock=public_source_lock,
        source_catalog=public_source_catalog,
    )

    return {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "spec_sha256": sha256_file(spec_path),
        "selected_domain": "code",
        "selection": {
            "result_sha256": sha256_file(selection_result_path),
            "suite_version": selection["suite_version"],
            "suite_sha256": selection["suite_sha256"],
            "baseline_exact_accuracy": domains["code"]["exact_accuracy"],
            "baseline_oracle_target_loss": domains["code"]["oracle_target_loss"],
        },
        "training": {
            "seed": int(spec["training_seed"]),
            "difficulty": int(spec["difficulty"]),
            "context_length": int(spec["context_length"]),
            "procedural_batch_fraction": float(spec["procedural_batch_fraction"]),
            "public_text_batch_fraction": float(spec["public_text_batch_fraction"]),
            "target_training_tokens": int(spec["target_training_tokens"]),
            "procedural": procedural,
        },
        "evaluation_separation": {
            "evaluation_base_seed": int(spec["evaluation_base_seed"]),
            "evaluation_domain_seed": evaluation_domain_seed,
            "evaluation_task_count": len(evaluation_tasks),
            "evaluation_task_set_sha256": evaluation_task_hash,
            "evaluation_prompt_set_sha256": _sha256_object(evaluation_prompt_hashes),
            "exact_prompt_overlap_count": exact_prompt_overlap_count,
        },
        "public_text": public_summary,
        "tokenizer_sha256": tokenizer_hash,
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic M6 useful-domain curriculum lock.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation-suite", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-source-lock", type=Path, required=True)
    parser.add_argument("--public-source-catalog", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_curriculum(
        spec_path=args.spec,
        selection_result_path=args.selection,
        evaluation_suite_path=args.evaluation_suite,
        tokenizer_path=args.tokenizer,
        public_data=args.public_data,
        public_source_lock=args.public_source_lock,
        public_source_catalog=args.public_source_catalog,
        records_path=args.records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
