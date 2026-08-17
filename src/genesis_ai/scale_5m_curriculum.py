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
from .scale_5m_contract import LADDER_SUITES, evidence_hashes, load_scale_contract
from .tokenizer import ByteBPETokenizer

CURRICULUM_VERSION = "m6-scale-5m-curriculum-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def _all_ladder_holdout_hashes(suite_paths: tuple[Path, ...] = LADDER_SUITES) -> tuple[set[str], dict[str, str]]:
    prompt_hashes: set[str] = set()
    suite_hashes: dict[str, str] = {}
    for path in suite_paths:
        if not path.is_file():
            raise ValueError(f"frozen GCI-Ladder suite missing: {path}")
        _, prompts, _ = frozen_holdouts(path)
        overlap = prompt_hashes & prompts
        if overlap:
            raise ValueError(f"frozen ladder suites unexpectedly overlap while building training exclusion: {path}")
        prompt_hashes.update(prompts)
        suite_hashes[path.as_posix()] = sha256_file(path)
    return prompt_hashes, suite_hashes


def generate_records(
    *,
    tokenizer: ByteBPETokenizer,
    holdout_prompt_hashes: set[str],
    examples_per_domain: int,
    seed: int,
    difficulty: int,
    context_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if examples_per_domain != 8_192:
        raise ValueError("~5M curriculum freezes 8192 examples/domain")
    if difficulty != 1:
        raise ValueError("~5M curriculum freezes verifier training at difficulty 1")
    records: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    per_domain: dict[str, dict[str, int]] = {}
    for domain_ordinal, domain in enumerate(DOMAINS):
        domain_seed = seed + domain_ordinal * 200_000
        rng = random.Random(domain_seed)
        accepted = 0
        attempts = 0
        terminated_response_tokens = 0
        max_attempts = examples_per_domain * 400
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
                raise ValueError("~5M verifier example tokenized empty")
            if len(response_ids) > context_length:
                raise ValueError("~5M verifier response exceeds model context length")
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
                    "domain_seed": domain_seed,
                    "domain_ordinal": domain_ordinal,
                    "ordinal": accepted,
                    "attempt": attempts,
                },
            }
            records.append({"id": f"scale5m-{_sha256_object(base)[:20]}", **base})
            seen_prompt_hashes.add(prompt_hash)
            terminated_response_tokens += len(response_ids)
            accepted += 1
        if accepted != examples_per_domain:
            raise RuntimeError(f"~5M {domain} generation exhausted after {attempts} attempts")
        per_domain[domain] = {
            "examples": accepted,
            "attempts": attempts,
            "terminated_response_tokens": terminated_response_tokens,
        }
    if seen_prompt_hashes & holdout_prompt_hashes:
        raise ValueError("blocking ~5M training overlap with frozen GCI-Ladder prompts")
    return records, {
        "examples": len(records),
        "examples_per_domain": examples_per_domain,
        "domains": per_domain,
        "prompt_set_sha256": _sha256_object(sorted(seen_prompt_hashes)),
        "record_set_sha256": _sha256_object(records),
        "exact_ladder_prompt_overlap_count": 0,
        "excluded_ladder_prompt_count": len(holdout_prompt_hashes),
    }


def build_curriculum(
    *,
    experiment_path: str | Path,
    finalist_path: str | Path,
    preflight_path: str | Path,
    tokenizer_path: str | Path,
    public_data: str | Path,
    records_path: str | Path,
) -> dict[str, Any]:
    experiment_path = Path(experiment_path)
    finalist_path = Path(finalist_path)
    preflight_path = Path(preflight_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    records_path = Path(records_path)
    experiment, _, _, config = load_scale_contract(
        experiment_path=experiment_path,
        finalist_path=finalist_path,
        preflight_path=preflight_path,
    )
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("~5M tokenizer vocabulary differs from frozen model config")
    holdout_hashes, suite_hashes = _all_ladder_holdout_hashes()
    training = experiment["training"]
    records, procedural = generate_records(
        tokenizer=tokenizer,
        holdout_prompt_hashes=holdout_hashes,
        examples_per_domain=int(training["examples_per_domain"]),
        seed=int(training["seed"]),
        difficulty=int(training["difficulty"]),
        context_length=config.context_length,
    )
    expected_records = 8_192 * len(DOMAINS)
    if len(records) != expected_records:
        raise ValueError("~5M curriculum record count drifted")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical(record) + "\n")
    procedural["records_file_sha256"] = sha256_file(records_path)
    procedural["records_file_size_bytes"] = records_path.stat().st_size
    result = {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        **evidence_hashes(
            experiment_path=experiment_path,
            finalist_path=finalist_path,
            preflight_path=preflight_path,
        ),
        "training_seed": int(training["seed"]),
        "difficulty": int(training["difficulty"]),
        "context_length": config.context_length,
        "expected_parameter_count": int(experiment["model"]["expected_parameter_count"]),
        "target_training_tokens": int(training["target_training_tokens"]),
        "target_tokens_per_step": int(training["target_tokens_per_step"]),
        "procedural_step_fraction": float(training["procedural_step_fraction"]),
        "public_step_fraction": float(training["public_step_fraction"]),
        "mandatory_first_and_terminator_coverage": True,
        "unique_target_contexts_only": True,
        "procedural": procedural,
        "ladder_separation": {
            "suite_sha256": suite_hashes,
            "excluded_prompt_set_sha256": _sha256_object(sorted(holdout_hashes)),
            "excluded_prompt_count": len(holdout_hashes),
            "exact_training_prompt_overlap_count": 0,
        },
        "public_text": {"manifest_sha256": sha256_file(public_data / "manifest.json")},
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "cash_compute_cost_usd": 0.0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen ladder-separated ~5M RoPE verifier curriculum.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--finalist", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_curriculum(
        experiment_path=args.experiment,
        finalist_path=args.finalist,
        preflight_path=args.preflight,
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
