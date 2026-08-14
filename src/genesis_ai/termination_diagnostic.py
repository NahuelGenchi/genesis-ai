from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_model, tokenizer_from_payload
from .domain_selection import generate_domain_tasks, load_suite, oracle_response
from .ingest import sha256_file
from .verifiers import verify_task

DIAGNOSTIC_VERSION = "m6-termination-diagnostic-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def analyze_response(
    *,
    task: dict[str, Any],
    oracle: str,
    oracle_ids: list[int],
    generated_ids: list[int],
    decoded_response: str,
    tokenizer,
) -> dict[str, Any]:
    evaluator_response = decoded_response.strip()
    verification = verify_task(task, evaluator_response)
    token_prefix = generated_ids[: len(oracle_ids)] == oracle_ids
    text_prefix = evaluator_response.startswith(oracle)
    extra_ids = generated_ids[len(oracle_ids) :] if token_prefix else []
    extra_text = tokenizer.decode(extra_ids, errors="replace") if extra_ids else ""
    prefix_then_extra = token_prefix and bool(extra_ids)
    return {
        "strict_pass": bool(verification.passed),
        "strict_reason": verification.reason,
        "oracle_token_prefix": token_prefix,
        "oracle_text_prefix": text_prefix,
        "prefix_then_extra": prefix_then_extra,
        "first_extra_is_newline": prefix_then_extra and extra_text.startswith("\n"),
        "generated_token_count": len(generated_ids),
        "oracle_token_count": len(oracle_ids),
        "response_sha256": hashlib.sha256(evaluator_response.encode("utf-8")).hexdigest(),
        "extra_sha256": hashlib.sha256(extra_text.encode("utf-8")).hexdigest() if extra_text else None,
        "evaluator_response": evaluator_response,
        "extra_text": extra_text,
    }


def diagnose_checkpoint(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    device: str = "cpu",
    sample_limit: int = 8,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    suite = load_suite(suite_path)
    if "code" not in suite["domains"]:
        raise ValueError("frozen suite does not contain code domain")
    domain_ordinal = list(suite["domains"]).index("code")
    tasks = generate_domain_tasks(
        domain="code",
        seed=int(suite["base_seed"]) + domain_ordinal,
        count=int(suite["tasks_per_domain"]),
        difficulty=int(suite["difficulty"]),
    )
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    generation = suite["generation"]
    model.eval()

    strict_pass = 0
    token_prefix = 0
    text_prefix = 0
    prefix_then_extra = 0
    newline_after_prefix = 0
    responses: list[str] = []
    samples: list[dict[str, Any]] = []

    for ordinal, task in enumerate(tasks):
        oracle = oracle_response(task)
        oracle_ids = tokenizer.encode(oracle)
        if not oracle_ids:
            raise ValueError("oracle encoded to zero tokens")
        prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens")
        run_seed = int(generation["seed"]) + domain_ordinal * 10000 + ordinal
        torch.manual_seed(run_seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            tokens,
            int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_k=int(generation["top_k"]),
        )[0].tolist()
        generated_ids = output[len(prompt_ids) :]
        decoded = tokenizer.decode(generated_ids, errors="replace")
        analysis = analyze_response(
            task=task,
            oracle=oracle,
            oracle_ids=oracle_ids,
            generated_ids=generated_ids,
            decoded_response=decoded,
            tokenizer=tokenizer,
        )
        strict_pass += int(analysis["strict_pass"])
        token_prefix += int(analysis["oracle_token_prefix"])
        text_prefix += int(analysis["oracle_text_prefix"])
        prefix_then_extra += int(analysis["prefix_then_extra"])
        newline_after_prefix += int(analysis["first_extra_is_newline"])
        responses.append(analysis["evaluator_response"])
        if len(samples) < sample_limit:
            samples.append(
                {
                    "task_id": task["id"],
                    "oracle": oracle,
                    "generated": analysis["evaluator_response"],
                    "extra_after_oracle": analysis["extra_text"] if analysis["oracle_token_prefix"] else None,
                    "strict_pass": analysis["strict_pass"],
                    "strict_reason": analysis["strict_reason"],
                    "oracle_token_prefix": analysis["oracle_token_prefix"],
                    "oracle_text_prefix": analysis["oracle_text_prefix"],
                    "generated_token_count": analysis["generated_token_count"],
                    "oracle_token_count": analysis["oracle_token_count"],
                }
            )

    count = len(tasks)
    if count <= 0:
        raise ValueError("diagnostic task set is empty")
    return {
        "format_version": "1.0",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(suite_path),
        "domain": "code",
        "task_count": count,
        "task_set_sha256": _sha256_object(tasks),
        "generation": dict(generation),
        "strict_exact": {"count": strict_pass, "rate": strict_pass / count},
        "oracle_token_prefix": {"count": token_prefix, "rate": token_prefix / count},
        "oracle_text_prefix": {"count": text_prefix, "rate": text_prefix / count},
        "oracle_prefix_then_extra": {"count": prefix_then_extra, "rate": prefix_then_extra / count},
        "newline_immediately_after_oracle": {"count": newline_after_prefix, "rate": newline_after_prefix / count},
        "response_set_sha256": _sha256_object(responses),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose M6 code answer termination without changing the frozen v1 suite.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = diagnose_checkpoint(
        checkpoint=args.checkpoint,
        suite_path=args.suite,
        device=args.device,
        sample_limit=args.sample_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
