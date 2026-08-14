from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .challenger import DOMAINS, build_task
from .domain_selection import generate_domain_tasks, oracle_response
from .ingest import sha256_file
from .terminated_eval import load_terminated_suite
from .tokenizer import ByteBPETokenizer

CURRICULUM_VERSION = "m6-multidomain-curriculum-v1"
TRAINING_SEED = 98_001
EXAMPLES_PER_DOMAIN = 1_365
CONTEXT_LENGTH = 128
TARGET_TRAINING_TOKENS = 2_000_000
PROCEDURAL_FRACTION = 0.8
PUBLIC_FRACTION = 0.2


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def frozen_holdouts(suite_path: str | Path) -> tuple[dict[str, list[dict[str, Any]]], set[str], dict[str, Any]]:
    suite = load_terminated_suite(suite_path)
    holdouts: dict[str, list[dict[str, Any]]] = {}
    prompt_hashes: set[str] = set()
    for ordinal, domain in enumerate(suite["domains"]):
        tasks = generate_domain_tasks(
            domain=domain,
            seed=int(suite["base_seed"]) + ordinal,
            count=int(suite["tasks_per_domain"]),
            difficulty=int(suite["difficulty"]),
        )
        holdouts[domain] = tasks
        prompt_hashes.update(_sha256_text(str(task["prompt"])) for task in tasks)
    return holdouts, prompt_hashes, suite


def generate_records(
    *,
    tokenizer: ByteBPETokenizer,
    holdout_prompt_hashes: set[str],
    examples_per_domain: int = EXAMPLES_PER_DOMAIN,
    seed: int = TRAINING_SEED,
    difficulty: int = 1,
    context_length: int = CONTEXT_LENGTH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if examples_per_domain <= 0:
        raise ValueError("examples_per_domain must be positive")
    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty must be in [1,5]")
    records: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    per_domain: dict[str, dict[str, int]] = {}

    for domain_ordinal, domain in enumerate(DOMAINS):
        rng = random.Random(seed + domain_ordinal * 100_000)
        accepted = 0
        attempts = 0
        max_attempts = examples_per_domain * 200
        response_tokens = 0
        while accepted < examples_per_domain and attempts < max_attempts:
            attempts += 1
            task = build_task(rng, domain, difficulty)
            prompt = str(task["prompt"])
            prompt_hash = _sha256_text(prompt)
            if prompt_hash in holdout_prompt_hashes or prompt_hash in seen_prompt_hashes:
                continue
            response = oracle_response(task)
            if "\n" in response:
                raise ValueError("multi-domain oracle contains reserved newline terminator")
            prompt_ids = tokenizer.encode(prompt + "\nAnswer:")
            response_ids = tokenizer.encode(response + "\n")
            if not prompt_ids or not response_ids:
                raise ValueError("multi-domain example tokenized empty")
            if len(response_ids) > context_length:
                raise ValueError("multi-domain response exceeds context length")
            base = {
                "format_version": "1.0",
                "curriculum": CURRICULUM_VERSION,
                "domain": domain,
                "difficulty": difficulty,
                "prompt": prompt,
                "response": response,
                "source_task_id": task["id"],
                "provenance": {
                    "kind": "procedural_oracle",
                    "generator": task["generator"],
                    "training_seed": seed,
                    "domain_seed": seed + domain_ordinal * 100_000,
                    "domain_ordinal": domain_ordinal,
                    "ordinal": accepted,
                    "attempt": attempts,
                },
            }
            records.append({"id": f"multi-{_sha256_object(base)[:20]}", **base})
            seen_prompt_hashes.add(prompt_hash)
            response_tokens += len(response_ids)
            accepted += 1
        if accepted != examples_per_domain:
            raise RuntimeError(f"{domain} generation exhausted after {attempts} attempts")
        per_domain[domain] = {
            "examples": accepted,
            "attempts": attempts,
            "terminated_response_tokens": response_tokens,
        }

    overlap = seen_prompt_hashes & holdout_prompt_hashes
    if overlap:
        raise ValueError("blocking prompt overlap with frozen holdouts")
    return records, {
        "examples": len(records),
        "examples_per_domain": examples_per_domain,
        "domains": per_domain,
        "prompt_set_sha256": _sha256_object(sorted(seen_prompt_hashes)),
        "record_set_sha256": _sha256_object(records),
        "exact_holdout_prompt_overlap_count": 0,
    }


def build_curriculum(
    *,
    suite_path: str | Path,
    tokenizer_path: str | Path,
    public_data: str | Path,
    records_path: str | Path,
    examples_per_domain: int = EXAMPLES_PER_DOMAIN,
) -> dict[str, Any]:
    suite_path = Path(suite_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    records_path = Path(records_path)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    holdouts, holdout_hashes, suite = frozen_holdouts(suite_path)
    records, procedural = generate_records(
        tokenizer=tokenizer,
        holdout_prompt_hashes=holdout_hashes,
        examples_per_domain=examples_per_domain,
        difficulty=int(suite["difficulty"]),
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical(record) + "\n")
    procedural["records_file_sha256"] = sha256_file(records_path)
    procedural["records_file_size_bytes"] = records_path.stat().st_size

    holdout_summary = {
        domain: {
            "tasks": len(tasks),
            "task_set_sha256": _sha256_object(tasks),
            "prompt_set_sha256": _sha256_object(sorted(_sha256_text(str(task["prompt"])) for task in tasks)),
        }
        for domain, tasks in holdouts.items()
    }
    return {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "training_seed": TRAINING_SEED,
        "difficulty": int(suite["difficulty"]),
        "context_length": CONTEXT_LENGTH,
        "target_training_tokens": TARGET_TRAINING_TOKENS,
        "procedural_fraction": PROCEDURAL_FRACTION,
        "public_fraction": PUBLIC_FRACTION,
        "procedural": procedural,
        "evaluation": {
            "suite_version": suite["suite_version"],
            "suite_sha256": sha256_file(suite_path),
            "domains": holdout_summary,
            "exact_prompt_overlap_count": 0,
        },
        "public_text": {"manifest_sha256": sha256_file(public_data / "manifest.json")},
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic M6 multi-domain verifier-backed curriculum.")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-domain", type=int, default=EXAMPLES_PER_DOMAIN)
    args = parser.parse_args()
    result = build_curriculum(
        suite_path=args.suite,
        tokenizer_path=args.tokenizer,
        public_data=args.public_data,
        records_path=args.records,
        examples_per_domain=args.examples_per_domain,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
