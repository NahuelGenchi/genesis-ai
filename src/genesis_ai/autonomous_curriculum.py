from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from .challenger import build_task
from .domain_selection import oracle_response
from .improvement_controller import CONTROLLER_VERSION, PUBLIC_MIN_CHARS_VARIANTS
from .ingest import sha256_file
from .multidomain_curriculum import frozen_holdouts
from .terminated_eval import load_terminated_suite
from .tokenizer import ByteBPETokenizer

CURRICULUM_VERSION = "autonomous-curriculum-v1"
DOMAINS = {"code", "math", "structured"}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_text(_canonical(value))


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("format_version") != "1.0" or plan.get("controller_version") != CONTROLLER_VERSION:
        raise ValueError("unsupported autonomous plan")
    stored = plan.get("plan_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        raise ValueError("plan_sha256 is required")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256", None)
    if _sha256_object(unhashed) != stored:
        raise ValueError("autonomous plan hash mismatch")
    decision = plan.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("plan decision is required")
    if decision.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("autonomous plan violates zero-cash contract")
    if decision.get("mandatory_first_and_terminator_coverage") is not True:
        raise ValueError("autonomous plan must require first/terminator coverage")
    if decision.get("unique_target_contexts_only") is not True:
        raise ValueError("autonomous plan must require unique target contexts")
    public_min_chars = decision.get("public_min_chars")
    allowed = {0, *PUBLIC_MIN_CHARS_VARIANTS.values()}
    if not isinstance(public_min_chars, int) or isinstance(public_min_chars, bool) or public_min_chars not in allowed:
        raise ValueError("autonomous plan public_min_chars is outside the screened allow-list")
    focus_examples = decision.get("focus_examples")
    if not isinstance(focus_examples, int) or isinstance(focus_examples, bool) or not 1 <= focus_examples <= 4096:
        raise ValueError("autonomous focus example count is outside bounded research contract")
    replay_examples = decision.get("replay_examples_per_domain")
    if not isinstance(replay_examples, int) or isinstance(replay_examples, bool) or replay_examples <= 0:
        raise ValueError("autonomous replay example count is invalid")
    weights = decision.get("continuation_update_weights")
    if not isinstance(weights, dict):
        raise ValueError("autonomous continuation weights are required")
    if set(weights) == {"focus", "each_replay_domain"}:
        total = float(weights["focus"]) + 2.0 * float(weights["each_replay_domain"])
    elif set(weights) == DOMAINS:
        total = sum(float(weights[domain]) for domain in DOMAINS)
    else:
        raise ValueError("autonomous continuation weights have an unsupported shape")
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("autonomous continuation weights must sum to 1")


def _domain_seed(plan_sha256: str, domain: str) -> int:
    digest = hashlib.sha256(f"{plan_sha256}:{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_000_000_000


def generate_domain_records(
    *,
    tokenizer: ByteBPETokenizer,
    domain: str,
    role: str,
    count: int,
    difficulty: int,
    seed: int,
    holdout_prompt_hashes: set[str],
    global_seen_prompt_hashes: set[str],
    context_length: int,
    plan_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if role not in {"focus", "replay"}:
        raise ValueError("role must be focus or replay")
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 300
    response_tokens = 0
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        task = build_task(rng, domain, difficulty)
        prompt = str(task["prompt"])
        prompt_hash = _sha256_text(prompt)
        if prompt_hash in holdout_prompt_hashes or prompt_hash in global_seen_prompt_hashes:
            continue
        response = oracle_response(task)
        if "\n" in response:
            raise ValueError("oracle contains reserved newline terminator")
        prompt_ids = tokenizer.encode(prompt + "\nAnswer:")
        response_ids = tokenizer.encode(response + "\n")
        if not prompt_ids or not response_ids:
            raise ValueError("autonomous example tokenized empty")
        if len(response_ids) > context_length:
            raise ValueError("autonomous response exceeds context length")
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
                "ordinal": len(records),
                "attempt": attempts,
            },
        }
        records.append({"id": f"auto-{_sha256_object(base)[:20]}", **base})
        global_seen_prompt_hashes.add(prompt_hash)
        response_tokens += len(response_ids)
    if len(records) != count:
        raise RuntimeError(f"{domain}/{role} exhausted after {attempts} attempts")
    return records, {"examples": len(records), "attempts": attempts, "terminated_response_tokens": response_tokens}


def build_curriculum(
    *,
    plan_path: str | Path,
    suite_path: str | Path,
    tokenizer_path: str | Path,
    public_data: str | Path,
    records_path: str | Path,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    suite_path = Path(suite_path)
    tokenizer_path = Path(tokenizer_path)
    public_data = Path(public_data)
    records_path = Path(records_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("autonomous plan must be a JSON object")
    _validate_plan(plan)
    decision = plan["decision"]
    transition = plan["evaluation_transition"]
    suite = load_terminated_suite(suite_path)
    target_difficulty = int(decision["target_difficulty"])
    if int(suite["difficulty"]) != target_difficulty:
        raise ValueError("target suite difficulty does not match autonomous plan")
    actual_suite_sha = sha256_file(suite_path)
    if transition.get("new_suite_required") is not True and actual_suite_sha != plan["input"]["suite_sha256"]:
        raise ValueError("repair cycle must use the same frozen suite as the planner input")
    if transition.get("new_suite_required") is True and target_difficulty == int(plan["input"]["difficulty"]):
        raise ValueError("difficulty transition claims a new suite without changing difficulty")

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    _, holdout_prompt_hashes, _ = frozen_holdouts(suite_path)
    seen: set[str] = set()
    focus = str(decision["focus_domain"])
    replay_domains = list(decision["replay_domains"])
    counts = {focus: int(decision["focus_examples"])}
    for domain in replay_domains:
        counts[str(domain)] = int(decision["replay_examples_per_domain"])
    if set(counts) != DOMAINS:
        raise ValueError("autonomous curriculum must cover all three domains")

    all_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for domain in (focus, *replay_domains):
        role = "focus" if domain == focus else "replay"
        records, metrics = generate_domain_records(
            tokenizer=tokenizer,
            domain=domain,
            role=role,
            count=counts[domain],
            difficulty=target_difficulty,
            seed=_domain_seed(plan["plan_sha256"], domain),
            holdout_prompt_hashes=holdout_prompt_hashes,
            global_seen_prompt_hashes=seen,
            context_length=128,
            plan_sha256=plan["plan_sha256"],
        )
        all_records.extend(records)
        summary[domain] = {"role": role, **metrics}

    if seen & holdout_prompt_hashes:
        raise ValueError("blocking autonomous curriculum overlap with target holdout")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(_canonical(record) + "\n")

    result = {
        "format_version": "1.0",
        "curriculum_version": CURRICULUM_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "incumbent_checkpoint_sha256": plan["input"]["incumbent_checkpoint_sha256"],
        "focus_domain": focus,
        "focus_examples": int(decision["focus_examples"]),
        "replay_examples_per_domain": int(decision["replay_examples_per_domain"]),
        "target_difficulty": target_difficulty,
        "target_training_tokens": int(decision["target_training_tokens"]),
        "procedural_fraction": float(decision["procedural_fraction"]),
        "public_fraction": float(decision["public_fraction"]),
        "public_min_chars": int(decision["public_min_chars"]),
        "continuation_update_weights": decision["continuation_update_weights"],
        "domain_records": summary,
        "record_count": len(all_records),
        "record_set_sha256": _sha256_object(all_records),
        "records_file_sha256": sha256_file(records_path),
        "target_suite_version": suite["suite_version"],
        "target_suite_sha256": actual_suite_sha,
        "exact_holdout_prompt_overlap_count": 0,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "cash_compute_cost_usd": 0.0,
    }
    for key in ("research_strategy", "history_summary", "research_escalation", "research_evidence"):
        if key in plan:
            result[key] = plan[key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verifier-backed curriculum from an autonomous Genesis plan.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_curriculum(
        plan_path=args.plan,
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
