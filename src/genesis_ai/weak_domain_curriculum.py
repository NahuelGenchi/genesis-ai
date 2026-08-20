from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .autonomous_curriculum import CURRICULUM_VERSION
from .challenger import build_task
from .domain_selection import oracle_response
from .ingest import sha256_file
from .multidomain_curriculum import frozen_holdouts
from .terminated_eval import load_terminated_suite
from .tokenizer import ByteBPETokenizer
from .weak_domain_funnel import FUNNEL_VERSION, build_catalog, validate_catalog
from .weak_domain_training import SCREEN_REPLAY_EXAMPLES_BY_BUDGET, SCREEN_STAGE_BY_BUDGET

FOCUS_EXAMPLES_BY_STAGE = {"tiny": 256, "medium": 1024}
CONTINUATION_WEIGHTS = {"focus": 0.70, "each_replay_domain": 0.15}
PUBLIC_MIN_CHARS = 0
CONTEXT_LENGTH = 128


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def _structured_values(task: dict[str, Any]) -> list[int]:
    prompt = str(task.get("prompt", ""))
    marker = "sort this integer array ascending: "
    if marker not in prompt:
        raise ValueError("structured weak-domain source must be an ascending-sort task")
    raw = prompt.split(marker, 1)[1]
    values = json.loads(raw)
    if not isinstance(values, list) or not values or any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise ValueError("structured weak-domain source array is invalid")
    return values


def build_variant_pair(variant_id: str, task: dict[str, Any], *, ordinal: int) -> tuple[str, str, str]:
    if variant_id == "math-operation-level":
        if task.get("domain") != "math":
            raise ValueError("math-operation-level requires a math task")
        return str(task["prompt"]), oracle_response(task), "operation-level difficulty progression"

    if task.get("domain") != "structured":
        raise ValueError("structured weak-domain variant requires a structured task")
    values = _structured_values(task)
    ordered = sorted(values)

    if variant_id == "structured-full-sort":
        return str(task["prompt"]), _canonical(ordered), "full transformation target"

    if variant_id == "structured-pairwise-rank":
        ranks = [[value, rank] for rank, value in enumerate(sorted(set(values)))]
        prompt = f"Return only JSON: assign each unique value its zero-based ascending rank: {_canonical(values)}"
        return prompt, _canonical(ranks), "pairwise comparison and rank decomposition"

    if variant_id == "structured-prefix-next":
        prefix_len = ordinal % len(ordered)
        prefix = ordered[:prefix_len]
        prompt = (
            "Return only JSON: given the source array and its already-correct ascending prefix, "
            f"return the next integer. source={_canonical(values)} prefix={_canonical(prefix)}"
        )
        return prompt, _canonical(ordered[prefix_len]), "prefix construction and next-element prediction"

    if variant_id == "structured-partial-completion":
        prefix_len = 1 + (ordinal % max(1, len(ordered) - 1))
        prefix = ordered[:prefix_len]
        prompt = (
            "Return only JSON: complete the remaining ascending suffix for this source array. "
            f"source={_canonical(values)} sorted_prefix={_canonical(prefix)}"
        )
        return prompt, _canonical(ordered[prefix_len:]), "partial sorted-sequence completion"

    if variant_id == "structured-length-progression":
        target_len = 1 + (ordinal % len(ordered))
        prompt = (
            f"Return only JSON: sort ascending and return only the first {target_len} values: "
            f"{_canonical(values)}"
        )
        return prompt, _canonical(ordered[:target_len]), "response-length progression from short to full transformations"

    if variant_id == "structured-mixed-decomposition":
        modes = (
            "structured-full-sort",
            "structured-pairwise-rank",
            "structured-prefix-next",
            "structured-partial-completion",
        )
        selected = modes[ordinal % len(modes)]
        prompt, response, _ = build_variant_pair(selected, task, ordinal=ordinal)
        return prompt, response, f"mixed decomposition plus full transformation ({selected})"

    raise ValueError(f"unsupported weak-domain curriculum variant: {variant_id}")


def _variant_spec(catalog: dict[str, Any], variant_id: str) -> dict[str, Any]:
    validate_catalog(catalog)
    matches = [item for item in catalog["variants"] if item.get("id") == variant_id]
    if len(matches) != 1:
        raise ValueError("variant must appear exactly once in the frozen funnel catalog")
    return dict(matches[0])


def _stage_budget(catalog: dict[str, Any], stage: str) -> int:
    if stage not in FOCUS_EXAMPLES_BY_STAGE:
        raise ValueError("weak-domain curriculum generation supports only tiny and medium screening stages")
    value = catalog.get("stages", {}).get(stage, {}).get("token_budget")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("frozen funnel stage budget is invalid")
    if SCREEN_STAGE_BY_BUDGET.get(value) != stage:
        raise ValueError("funnel stage/budget drifted from the screening execution adapter")
    return value


def _difficulty_for_focus(variant_id: str, *, ordinal: int, target_difficulty: int) -> int:
    if variant_id == "math-operation-level":
        return 1 + (ordinal % target_difficulty)
    return target_difficulty


def _record(
    *,
    plan_sha256: str,
    role: str,
    domain: str,
    difficulty: int,
    prompt: str,
    response: str,
    task: dict[str, Any],
    seed: int,
    ordinal: int,
    attempt: int,
    supervision: str,
    variant_id: str,
) -> dict[str, Any]:
    base = {
        "format_version": "1.0",
        "curriculum": CURRICULUM_VERSION,
        "plan_sha256": plan_sha256,
        "role": role,
        "domain": domain,
        "difficulty": difficulty,
        "prompt": prompt,
        "response": response,
        "source_task_id": task["id"],
        "provenance": {
            "kind": "procedural_oracle",
            "generator": task["generator"],
            "domain_seed": seed,
            "ordinal": ordinal,
            "attempt": attempt,
            "research_funnel_version": FUNNEL_VERSION,
            "variant_id": variant_id,
            "supervision": supervision,
        },
    }
    return {"id": f"weak-{_sha256_object(base)[:20]}", **base}


def _generate_records(
    *,
    tokenizer: ByteBPETokenizer,
    variant_id: str,
    domain: str,
    role: str,
    count: int,
    target_difficulty: int,
    seed: int,
    plan_sha256: str,
    holdout_prompt_hashes: set[str],
    seen_prompt_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    attempts = 0
    response_tokens = 0
    max_attempts = count * 500
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        ordinal = len(records)
        difficulty = _difficulty_for_focus(variant_id, ordinal=ordinal, target_difficulty=target_difficulty) if role == "focus" else target_difficulty
        task = build_task(rng, domain, difficulty)
        if role == "focus":
            try:
                prompt, response, supervision = build_variant_pair(variant_id, task, ordinal=ordinal)
            except ValueError:
                continue
        else:
            prompt = str(task["prompt"])
            response = oracle_response(task)
            supervision = "standard non-focus replay"
        if "\n" in response:
            raise ValueError("weak-domain oracle contains reserved newline terminator")
        prompt_hash = _sha256_text(prompt)
        if prompt_hash in holdout_prompt_hashes or prompt_hash in seen_prompt_hashes:
            continue
        prompt_ids = tokenizer.encode(prompt + "\nAnswer:")
        response_ids = tokenizer.encode(response + "\n")
        if not prompt_ids or not response_ids or len(response_ids) > CONTEXT_LENGTH:
            continue
        records.append(
            _record(
                plan_sha256=plan_sha256,
                role=role,
                domain=domain,
                difficulty=difficulty,
                prompt=prompt,
                response=response,
                task=task,
                seed=seed,
                ordinal=ordinal,
                attempt=attempts,
                supervision=supervision,
                variant_id=variant_id,
            )
        )
        seen_prompt_hashes.add(prompt_hash)
        response_tokens += len(response_ids)
    if len(records) != count:
        raise RuntimeError(f"weak-domain {variant_id}/{domain}/{role} exhausted after {attempts} attempts")
    return records, {"examples": len(records), "attempts": attempts, "terminated_response_tokens": response_tokens}


def build_screen_curriculum(
    *,
    catalog_path: str | Path,
    variant_id: str,
    stage: str,
    suite_path: str | Path,
    tokenizer_path: str | Path,
    public_data: str | Path,
    parent_checkpoint: str | Path,
    records_path: str | Path,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    suite_path = Path(suite_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    parent_checkpoint = Path(parent_checkpoint)
    records_path = Path(records_path)

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("weak-domain funnel catalog must be a JSON object")
    spec = _variant_spec(catalog, variant_id)
    budget = _stage_budget(catalog, stage)
    replay_count = SCREEN_REPLAY_EXAMPLES_BY_BUDGET[budget]
    focus_count = FOCUS_EXAMPLES_BY_STAGE[stage]

    suite = load_terminated_suite(suite_path)
    target_difficulty = int(suite["difficulty"])
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    _, holdout_prompt_hashes, _ = frozen_holdouts(suite_path)
    focus_domains = list(spec["domains"])
    if len(focus_domains) != 1 or focus_domains[0] not in {"structured", "math"}:
        raise ValueError("weak-domain variant must have exactly one supported focus domain")
    focus_domain = str(focus_domains[0])
    replay_domains = [domain for domain in ("code", "math", "structured") if domain != focus_domain]

    identity = {
        "research_funnel_version": FUNNEL_VERSION,
        "catalog_sha256": catalog["catalog_sha256"],
        "variant_id": variant_id,
        "funnel_stage": stage,
        "training_seed": int(spec["training_seed"]),
        "target_training_tokens": budget,
        "focus_examples": focus_count,
        "replay_examples_per_domain": replay_count,
        "target_suite_sha256": sha256_file(suite_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "incumbent_checkpoint_sha256": sha256_file(parent_checkpoint),
    }
    plan_sha256 = _sha256_object(identity)
    seen: set[str] = set()
    all_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    focus_records, metrics = _generate_records(
        tokenizer=tokenizer,
        variant_id=variant_id,
        domain=focus_domain,
        role="focus",
        count=focus_count,
        target_difficulty=target_difficulty,
        seed=int(spec["training_seed"]),
        plan_sha256=plan_sha256,
        holdout_prompt_hashes=holdout_prompt_hashes,
        seen_prompt_hashes=seen,
    )
    all_records.extend(focus_records)
    summary[focus_domain] = {"role": "focus", **metrics}

    for ordinal, domain in enumerate(replay_domains, 1):
        replay_records, replay_metrics = _generate_records(
            tokenizer=tokenizer,
            variant_id=variant_id,
            domain=domain,
            role="replay",
            count=replay_count,
            target_difficulty=target_difficulty,
            seed=int(spec["training_seed"]) + ordinal * 100_003,
            plan_sha256=plan_sha256,
            holdout_prompt_hashes=holdout_prompt_hashes,
            seen_prompt_hashes=seen,
        )
        all_records.extend(replay_records)
        summary[domain] = {"role": "replay", **replay_metrics}

    if seen & holdout_prompt_hashes:
        raise ValueError("blocking weak-domain curriculum overlap with frozen holdout")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(_canonical(record) + "\n")

    return {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "research_funnel_version": FUNNEL_VERSION,
        "catalog_sha256": catalog["catalog_sha256"],
        "plan_sha256": plan_sha256,
        "variant_id": variant_id,
        "funnel_stage": stage,
        "training_seed": int(spec["training_seed"]),
        "incumbent_checkpoint_sha256": identity["incumbent_checkpoint_sha256"],
        "focus_domain": focus_domain,
        "focus_examples": focus_count,
        "replay_examples_per_domain": replay_count,
        "target_difficulty": target_difficulty,
        "target_training_tokens": budget,
        "procedural_fraction": 0.8,
        "public_fraction": 0.2,
        "public_min_chars": PUBLIC_MIN_CHARS,
        "continuation_update_weights": CONTINUATION_WEIGHTS,
        "domain_records": summary,
        "record_count": len(all_records),
        "record_set_sha256": _sha256_object(all_records),
        "records_file_sha256": sha256_file(records_path),
        "target_suite_version": suite["suite_version"],
        "target_suite_sha256": identity["target_suite_sha256"],
        "exact_holdout_prompt_overlap_count": 0,
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "public_manifest_sha256": identity["public_manifest_sha256"],
        "screening_only": True,
        "promotion_authority": False,
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic holdout-separated weak-domain screen curriculum.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--stage", choices=tuple(FOCUS_EXAMPLES_BY_STAGE), required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_screen_curriculum(
        catalog_path=args.catalog,
        variant_id=args.variant,
        stage=args.stage,
        suite_path=args.suite,
        tokenizer_path=args.tokenizer,
        public_data=args.public_data,
        parent_checkpoint=args.parent_checkpoint,
        records_path=args.records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
