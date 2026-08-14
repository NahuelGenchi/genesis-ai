from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .challenger import DOMAINS, build_task
from .domain_selection import oracle_response
from .ingest import sha256_file
from .multidomain_curriculum import frozen_holdouts
from .terminated_eval import load_terminated_suite
from .tokenizer import ByteBPETokenizer

CURRICULUM_VERSION = "m6-multidomain-curriculum-v2"
TRAINING_SEED = 100_001
EXAMPLES_PER_DOMAIN = 4_096
CONTEXT_LENGTH = 128
TARGET_TRAINING_TOKENS = 6_000_000
PROCEDURAL_FRACTION = 0.8
PUBLIC_FRACTION = 0.2


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def generate_records(
    *,
    tokenizer: ByteBPETokenizer,
    holdout_prompt_hashes: set[str],
    examples_per_domain: int = EXAMPLES_PER_DOMAIN,
    seed: int = TRAINING_SEED,
    difficulty: int = 1,
    context_length: int = CONTEXT_LENGTH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if examples_per_domain != EXAMPLES_PER_DOMAIN:
        raise ValueError(f"v2 freezes examples_per_domain={EXAMPLES_PER_DOMAIN}")
    if difficulty != 1:
        raise ValueError("v2 freezes difficulty=1")

    records: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    per_domain: dict[str, dict[str, int]] = {}

    for domain_ordinal, domain in enumerate(DOMAINS):
        rng = random.Random(seed + domain_ordinal * 100_000)
        accepted = 0
        attempts = 0
        response_tokens = 0
        max_attempts = examples_per_domain * 200
        while accepted < examples_per_domain and attempts < max_attempts:
            attempts += 1
            task = build_task(rng, domain, difficulty)
            prompt = str(task["prompt"])
            prompt_hash = _sha256_text(prompt)
            if prompt_hash in holdout_prompt_hashes or prompt_hash in seen_prompt_hashes:
                continue
            response = oracle_response(task)
            if "\n" in response:
                raise ValueError("oracle contains reserved newline terminator")
            prompt_ids = tokenizer.encode(prompt + "\nAnswer:")
            response_ids = tokenizer.encode(response + "\n")
            if not prompt_ids or not response_ids:
                raise ValueError("example tokenized empty")
            if len(response_ids) > context_length:
                raise ValueError("response exceeds context length")
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
            records.append({"id": f"multi-v2-{_sha256_object(base)[:20]}", **base})
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

    if seen_prompt_hashes & holdout_prompt_hashes:
        raise ValueError("blocking prompt overlap with frozen holdouts")
    if len(records) != EXAMPLES_PER_DOMAIN * len(DOMAINS):
        raise AssertionError("v2 record count drifted")
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
) -> dict[str, Any]:
    suite_path = Path(suite_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    records_path = Path(records_path)
    suite = load_terminated_suite(suite_path)
    if int(suite["difficulty"]) != 1:
        raise ValueError("v2 requires frozen difficulty 1 suite")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    _, holdout_hashes, _ = frozen_holdouts(suite_path)
    records, procedural = generate_records(tokenizer=tokenizer, holdout_prompt_hashes=holdout_hashes)

    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical(record) + "\n")
    procedural["records_file_sha256"] = sha256_file(records_path)
    procedural["records_file_size_bytes"] = records_path.stat().st_size

    return {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "training_seed": TRAINING_SEED,
        "difficulty": 1,
        "context_length": CONTEXT_LENGTH,
        "target_training_tokens": TARGET_TRAINING_TOKENS,
        "procedural_fraction": PROCEDURAL_FRACTION,
        "public_fraction": PUBLIC_FRACTION,
        "procedural": procedural,
        "evaluation": {
            "suite_version": suite["suite_version"],
            "suite_sha256": sha256_file(suite_path),
            "exact_prompt_overlap_count": 0,
        },
        "public_text": {"manifest_sha256": sha256_file(public_data / "manifest.json")},
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen M6 multi-domain full-capacity curriculum v2.")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_curriculum(
        suite_path=args.suite,
        tokenizer_path=args.tokenizer,
        public_data=args.public_data,
        records_path=args.records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
