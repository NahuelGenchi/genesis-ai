from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .autonomous_curriculum import CURRICULUM_VERSION, generate_domain_records
from .challenger import build_task
from .domain_selection import oracle_response
from .improvement_controller import CONTROLLER_VERSION
from .ingest import sha256_file
from .multidomain_curriculum import frozen_holdouts
from .research_funnel import FUNNEL_VERSION, stage_config, variant_config
from .terminated_eval import load_terminated_suite
from .tokenizer import ByteBPETokenizer

DOMAINS = ("code", "math", "structured")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def _seed(variant_id: str, stage: str, domain: str) -> int:
    digest = hashlib.sha256(f"{FUNNEL_VERSION}:{variant_id}:{stage}:{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_000_000_000


def _structured_source(task: dict[str, Any]) -> list[int]:
    prompt = str(task["prompt"])
    marker = "ascending: "
    if marker not in prompt:
        raise ValueError("weak-domain structured decomposition currently requires canonical sort tasks")
    values = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(values, list) or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("structured source array is invalid")
    return values


def _math_operands(task: dict[str, Any]) -> tuple[int, int]:
    prompt = str(task["prompt"])
    match = re.fullmatch(r"Return only the integer result of (-?\d+) \+ (-?\d+)\.", prompt)
    if match is None:
        raise ValueError("weak-domain math decomposition currently requires difficulty-1 addition tasks")
    return int(match.group(1)), int(match.group(2))


def _focus_example(task: dict[str, Any], *, mode: str, ordinal: int) -> tuple[str, str, str]:
    domain = str(task["domain"])
    if domain == "structured":
        source = _structured_source(task)
        expected = list(task["verifier"]["expected"])
        if mode == "full-sort":
            return str(task["prompt"]), oracle_response(task), "full-sort"
        if mode == "mixed-decomposition":
            mode = ("full-sort", "pairwise-rank", "prefix-next", "partial-completion")[ordinal % 4]
        if mode == "pairwise-rank":
            value = source[ordinal % len(source)]
            response = [sum(item < value for item in source), sum(item <= value for item in source)]
            prompt = (
                "Return only JSON [count_less,count_less_or_equal]. "
                f"Array: {_canonical(source)}; value: {value}"
            )
            return prompt, _canonical(response), "pairwise-rank"
        if mode == "prefix-next":
            prefix_len = ordinal % len(expected)
            prefix = expected[:prefix_len]
            prompt = (
                "Return only the next integer in the ascending sorted result as JSON. "
                f"Source: {_canonical(source)}; sorted prefix: {_canonical(prefix)}"
            )
            return prompt, _canonical(expected[prefix_len]), "prefix-next"
        if mode == "partial-completion":
            prefix_len = ordinal % len(expected)
            prefix = expected[:prefix_len]
            prompt = (
                "Return only JSON: complete the remaining ascending sorted suffix. "
                f"Source: {_canonical(source)}; sorted prefix: {_canonical(prefix)}"
            )
            return prompt, _canonical(expected[prefix_len:]), "partial-completion"
        if mode == "short-to-long":
            length = 2 + ordinal % (len(source) - 1)
            subset = source[:length]
            prompt = f"Return only JSON: sort this shorter integer array ascending: {_canonical(subset)}"
            return prompt, _canonical(sorted(subset)), "short-to-long"
        raise ValueError(f"unsupported structured curriculum mode: {mode}")

    if domain == "math":
        a, b = _math_operands(task)
        if mode == "operation-decomposition":
            prompt = f"Return only JSON [left,right,sum] for this addition: {a} + {b}."
            return prompt, _canonical([a, b, a + b]), "operation-decomposition"
        if mode == "direct-plus-steps":
            if ordinal % 2 == 0:
                return str(task["prompt"]), oracle_response(task), "direct"
            prompt = f"Return only JSON [left,right,sum] for this addition: {a} + {b}."
            return prompt, _canonical([a, b, a + b]), "operation-decomposition"
        raise ValueError(f"unsupported math curriculum mode: {mode}")

    raise ValueError(f"focus decomposition is unsupported for domain: {domain}")


def _generate_focus_records(
    *,
    tokenizer: ByteBPETokenizer,
    domain: str,
    count: int,
    difficulty: int,
    variant_id: str,
    mode: str,
    stage: str,
    holdout_prompt_hashes: set[str],
    global_seen_prompt_hashes: set[str],
    context_length: int,
    plan_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(_seed(variant_id, stage, domain))
    records: list[dict[str, Any]] = []
    attempts = 0
    response_tokens = 0
    decomposition_counts: dict[str, int] = {}
    while len(records) < count and attempts < count * 500:
        attempts += 1
        task = build_task(rng, domain, difficulty)
        prompt, response, decomposition = _focus_example(task, mode=mode, ordinal=len(records))
        prompt_hash = _sha256_text(prompt)
        if prompt_hash in holdout_prompt_hashes or prompt_hash in global_seen_prompt_hashes:
            continue
        if "\n" in response:
            raise ValueError("weak-domain oracle contains reserved newline terminator")
        response_ids = tokenizer.encode(response + "\n")
        if not response_ids or len(response_ids) > context_length:
            continue
        base = {
            "format_version": "1.0",
            "curriculum": CURRICULUM_VERSION,
            "plan_sha256": plan_sha256,
            "role": "focus",
            "domain": domain,
            "difficulty": difficulty,
            "prompt": prompt,
            "response": response,
            "source_task_id": task["id"],
            "provenance": {
                "kind": "procedural_oracle_decomposition",
                "generator": task["generator"],
                "variant_id": variant_id,
                "decomposition": decomposition,
                "stage": stage,
                "ordinal": len(records),
                "attempt": attempts,
            },
        }
        records.append({"id": f"weak-{_sha256_object(base)[:20]}", **base})
        global_seen_prompt_hashes.add(prompt_hash)
        response_tokens += len(response_ids)
        decomposition_counts[decomposition] = decomposition_counts.get(decomposition, 0) + 1
    if len(records) != count:
        raise RuntimeError(f"weak-domain focus generation exhausted after {attempts} attempts")
    return records, {
        "examples": len(records),
        "attempts": attempts,
        "terminated_response_tokens": response_tokens,
        "decomposition_counts": dict(sorted(decomposition_counts.items())),
    }


def _holdout_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        _, prompt_hashes, _ = frozen_holdouts(path)
        hashes.update(prompt_hashes)
    return hashes


def build_plan(
    *,
    incumbent_sha256: str,
    suite_path: Path,
    variant_id: str,
    stage: str,
) -> dict[str, Any]:
    variant = variant_config(variant_id)
    config = stage_config(stage)
    suite = load_terminated_suite(suite_path)
    focus = str(variant["focus_domain"])
    replay = [domain for domain in DOMAINS if domain != focus]
    plan: dict[str, Any] = {
        "format_version": "1.0",
        "controller_version": CONTROLLER_VERSION,
        "funnel_version": FUNNEL_VERSION,
        "input": {
            "cycle_index": 0,
            "incumbent_checkpoint_sha256": incumbent_sha256,
            "suite_version": suite["suite_version"],
            "suite_sha256": sha256_file(suite_path),
            "difficulty": int(suite["difficulty"]),
            "research_only": stage != "full",
        },
        "evaluation_transition": {
            "new_suite_required": False,
            "target_difficulty": int(suite["difficulty"]),
            "incumbent_must_be_scored_on_target_suite_before_training": False,
            "cross_difficulty_improvement_comparison_forbidden": True,
        },
        "decision": {
            "mode": "weak-domain-successive-halving",
            "variant_id": variant_id,
            "curriculum_mode": variant["curriculum_mode"],
            "stage": stage,
            "focus_domain": focus,
            "target_difficulty": int(suite["difficulty"]),
            "target_training_tokens": config.target_training_tokens,
            "focus_examples": config.focus_examples,
            "replay_domains": replay,
            "replay_examples_per_domain": config.replay_examples_per_domain,
            "continuation_update_weights": {"focus": 0.70, "each_replay_domain": 0.15},
            "mandatory_first_and_terminator_coverage": True,
            "unique_target_contexts_only": True,
            "procedural_fraction": 0.80,
            "public_fraction": 0.20,
            "public_min_chars": 0,
            "cash_compute_cost_usd": 0.0,
        },
        "promotion_contract": {
            "same_suite_comparison_required": True,
            "minimum_focus_absolute_gain": 0.10,
            "minimum_gci_absolute_gain": 3.0,
            "maximum_nonfocus_absolute_regression": 0.05,
            "maximum_m3_loss_regression_fraction": 0.02,
            "zero_holdout_overlap_required": True,
            "semantic_reproduction_required": True,
            "zero_cash_compute_required": True,
            "live_incumbent_weight_mutation_forbidden": True,
        },
        "screening_contract": {
            "screening_only": stage != "full",
            "promotion_eligible": stage == "full",
            "promotion_authority": False if stage != "full" else "immutable-gate-only",
            "early_stop_enabled": stage != "full",
        },
    }
    plan["plan_sha256"] = _sha256_object(plan)
    return plan


def build_curriculum(
    *,
    parent_checkpoint: str | Path,
    suite_path: str | Path,
    holdout_suites: list[str | Path],
    tokenizer_path: str | Path,
    public_data: str | Path,
    variant_id: str,
    stage: str,
    records_path: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    parent_checkpoint = Path(parent_checkpoint)
    suite_path = Path(suite_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    records_path = Path(records_path)
    plan_path = Path(plan_path)
    variant = variant_config(variant_id)
    config = stage_config(stage)
    parent_sha = sha256_file(parent_checkpoint)
    plan = build_plan(incumbent_sha256=parent_sha, suite_path=suite_path, variant_id=variant_id, stage=stage)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    suite = load_terminated_suite(suite_path)
    difficulty = int(suite["difficulty"])
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    holdout_paths = [Path(path) for path in holdout_suites]
    if suite_path not in holdout_paths:
        holdout_paths.append(suite_path)
    holdout_prompt_hashes = _holdout_hashes(holdout_paths)
    seen: set[str] = set()
    focus = str(variant["focus_domain"])
    replay = [domain for domain in DOMAINS if domain != focus]

    focus_records, focus_summary = _generate_focus_records(
        tokenizer=tokenizer,
        domain=focus,
        count=config.focus_examples,
        difficulty=difficulty,
        variant_id=variant_id,
        mode=str(variant["curriculum_mode"]),
        stage=stage,
        holdout_prompt_hashes=holdout_prompt_hashes,
        global_seen_prompt_hashes=seen,
        context_length=128,
        plan_sha256=plan["plan_sha256"],
    )
    all_records = list(focus_records)
    summary: dict[str, Any] = {focus: {"role": "focus", **focus_summary}}
    for domain in replay:
        records, metrics = generate_domain_records(
            tokenizer=tokenizer,
            domain=domain,
            role="replay",
            count=config.replay_examples_per_domain,
            difficulty=difficulty,
            seed=_seed(variant_id, stage, domain),
            holdout_prompt_hashes=holdout_prompt_hashes,
            global_seen_prompt_hashes=seen,
            context_length=128,
            plan_sha256=plan["plan_sha256"],
        )
        all_records.extend(records)
        summary[domain] = {"role": "replay", **metrics}

    if seen & holdout_prompt_hashes:
        raise ValueError("blocking weak-domain curriculum overlap with frozen/dev holdout")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(_canonical(record) + "\n")

    curriculum = {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "funnel_version": FUNNEL_VERSION,
        "variant_id": variant_id,
        "stage": stage,
        "screening_only": stage != "full",
        "promotion_eligible": stage == "full",
        "promotion_authority": False if stage != "full" else "immutable-gate-only",
        "plan_sha256": plan["plan_sha256"],
        "incumbent_checkpoint_sha256": parent_sha,
        "focus_domain": focus,
        "focus_examples": config.focus_examples,
        "replay_examples_per_domain": config.replay_examples_per_domain,
        "target_difficulty": difficulty,
        "target_training_tokens": config.target_training_tokens,
        "procedural_fraction": 0.80,
        "public_fraction": 0.20,
        "public_min_chars": 0,
        "continuation_update_weights": {"focus": 0.70, "each_replay_domain": 0.15},
        "domain_records": summary,
        "record_count": len(all_records),
        "record_set_sha256": _sha256_object(all_records),
        "records_file_sha256": sha256_file(records_path),
        "target_suite_version": suite["suite_version"],
        "target_suite_sha256": sha256_file(suite_path),
        "exact_holdout_prompt_overlap_count": 0,
        "holdout_suite_sha256": {path.as_posix(): sha256_file(path) for path in holdout_paths},
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "cash_compute_cost_usd": 0.0,
    }
    return curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic weak-domain decomposition curricula.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--holdout-suites", nargs="+", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--stage", choices=("tiny", "medium", "full"), required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_curriculum(
        parent_checkpoint=args.parent,
        suite_path=args.suite,
        holdout_suites=list(args.holdout_suites),
        tokenizer_path=args.tokenizer,
        public_data=args.public_data,
        variant_id=args.variant,
        stage=args.stage,
        records_path=args.records,
        plan_path=args.plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
