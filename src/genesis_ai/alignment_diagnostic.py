from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import load_model, tokenizer_from_payload
from .domain_selection import generate_domain_tasks, load_suite, oracle_response
from .ingest import sha256_file

ALIGNMENT_DIAGNOSTIC_VERSION = "m6-answer-alignment-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def end_anchored_logit_positions(prompt_length: int, response_length: int, context_length: int) -> list[int]:
    if prompt_length <= 0 or response_length <= 0 or context_length <= 0:
        raise ValueError("lengths must be positive")
    if response_length >= context_length:
        raise ValueError("response must fit context")
    sequence_length = prompt_length + response_length
    window_start = max(0, sequence_length - (context_length + 1))
    positions: list[int] = []
    for response_index in range(response_length):
        global_target_index = prompt_length + response_index
        local_logit_position = global_target_index - window_start - 1
        if not 0 <= local_logit_position < context_length:
            raise ValueError("end-anchored target position is outside model context")
        positions.append(local_logit_position)
    return positions


def _token_measurement(next_logits: torch.Tensor, target: int) -> tuple[bool, float, float]:
    log_probs = F.log_softmax(next_logits.float(), dim=-1)
    target_log_prob = float(log_probs[target].detach().cpu())
    probability = math.exp(target_log_prob)
    predicted = int(torch.argmax(next_logits).item())
    return predicted == target, -target_log_prob, probability


def _measure_task(
    *,
    model,
    tokenizer,
    task: dict[str, Any],
    generation: dict[str, Any],
    ordinal: int,
    domain_ordinal: int,
    device: str,
) -> dict[str, Any]:
    oracle = oracle_response(task)
    prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
    oracle_ids = tokenizer.encode(oracle)
    if not prompt_ids or not oracle_ids:
        raise ValueError("empty prompt/oracle tokenization")
    context_length = model.config.context_length
    if len(oracle_ids) >= context_length:
        raise ValueError("oracle response does not fit context")

    # Current training/oracle-loss layout: one fixed window is anchored to the
    # end of the complete oracle response. Causality hides future answer tokens,
    # but learned absolute positions still differ from free-running generation.
    sequence = prompt_ids + oracle_ids
    window_start = max(0, len(sequence) - (context_length + 1))
    window = sequence[window_start:]
    x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
    with torch.no_grad():
        end_logits, _ = model(x)
    end_positions = end_anchored_logit_positions(len(prompt_ids), len(oracle_ids), context_length)
    end_metrics: list[dict[str, Any]] = []
    for response_index, (target, logit_position) in enumerate(zip(oracle_ids, end_positions)):
        correct, nll, probability = _token_measurement(end_logits[0, logit_position], target)
        end_metrics.append(
            {
                "response_index": response_index,
                "model_position": logit_position,
                "top1_correct": correct,
                "nll": nll,
                "probability": probability,
            }
        )

    # Generation-aligned teacher forcing: each next oracle token is scored from
    # exactly the same last-context layout that model.generate uses at that step.
    aligned_metrics: list[dict[str, Any]] = []
    for response_index, target in enumerate(oracle_ids):
        history = prompt_ids + oracle_ids[:response_index]
        context = history[-context_length:]
        x_aligned = torch.tensor([context], dtype=torch.long, device=device)
        with torch.no_grad():
            aligned_logits, _ = model(x_aligned)
        correct, nll, probability = _token_measurement(aligned_logits[0, -1], target)
        aligned_metrics.append(
            {
                "response_index": response_index,
                "model_position": len(context) - 1,
                "top1_correct": correct,
                "nll": nll,
                "probability": probability,
            }
        )

    run_seed = int(generation["seed"]) + domain_ordinal * 10000 + ordinal
    torch.manual_seed(run_seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_tensor,
        int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        top_k=int(generation["top_k"]),
    )[0].tolist()[len(prompt_ids) :]
    prefix_length = longest_common_prefix(generated, oracle_ids)

    return {
        "task_id": task["id"],
        "oracle_token_count": len(oracle_ids),
        "end_anchored": end_metrics,
        "generation_aligned": aligned_metrics,
        "free_running_correct_prefix_tokens": prefix_length,
        "free_running_first_token_correct": prefix_length >= 1,
        "generated_token_count": len(generated),
    }


def _summarize_position(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("position summary requires rows")
    return {
        "count": len(rows),
        "top1_correct": sum(int(row["top1_correct"]) for row in rows),
        "top1_accuracy": sum(int(row["top1_correct"]) for row in rows) / len(rows),
        "mean_nll": sum(float(row["nll"]) for row in rows) / len(rows),
        "mean_probability": sum(float(row["probability"]) for row in rows) / len(rows),
        "model_position_min": min(int(row["model_position"]) for row in rows),
        "model_position_max": max(int(row["model_position"]) for row in rows),
        "model_position_mean": sum(int(row["model_position"]) for row in rows) / len(rows),
    }


def diagnose_alignment(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    device: str = "cpu",
    sample_limit: int = 8,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    suite = load_suite(suite_path)
    if "code" not in suite["domains"]:
        raise ValueError("suite does not contain code domain")
    if int(suite["generation"]["top_k"]) != 1:
        raise ValueError("alignment diagnostic requires greedy top_k=1 generation")
    domain_ordinal = list(suite["domains"]).index("code")
    tasks = generate_domain_tasks(
        domain="code",
        seed=int(suite["base_seed"]) + domain_ordinal,
        count=int(suite["tasks_per_domain"]),
        difficulty=int(suite["difficulty"]),
    )
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    model.eval()

    task_results = [
        _measure_task(
            model=model,
            tokenizer=tokenizer,
            task=task,
            generation=suite["generation"],
            ordinal=ordinal,
            domain_ordinal=domain_ordinal,
            device=device,
        )
        for ordinal, task in enumerate(tasks)
    ]

    end_all = [metric for task in task_results for metric in task["end_anchored"]]
    aligned_all = [metric for task in task_results for metric in task["generation_aligned"]]
    end_first = [task["end_anchored"][0] for task in task_results]
    aligned_first = [task["generation_aligned"][0] for task in task_results]
    max_response_length = max(task["oracle_token_count"] for task in task_results)
    per_position: list[dict[str, Any]] = []
    for response_index in range(max_response_length):
        end_rows = [task["end_anchored"][response_index] for task in task_results if task["oracle_token_count"] > response_index]
        aligned_rows = [task["generation_aligned"][response_index] for task in task_results if task["oracle_token_count"] > response_index]
        per_position.append(
            {
                "response_index": response_index,
                "end_anchored": _summarize_position(end_rows),
                "generation_aligned": _summarize_position(aligned_rows),
            }
        )

    prefix_lengths = [int(task["free_running_correct_prefix_tokens"]) for task in task_results]
    histogram = {str(length): prefix_lengths.count(length) for length in sorted(set(prefix_lengths))}
    samples = [
        {
            "task_id": task["task_id"],
            "oracle_token_count": task["oracle_token_count"],
            "free_running_correct_prefix_tokens": task["free_running_correct_prefix_tokens"],
            "end_anchored_first": task["end_anchored"][0],
            "generation_aligned_first": task["generation_aligned"][0],
        }
        for task in task_results[:sample_limit]
    ]

    return {
        "format_version": "1.0",
        "diagnostic_version": ALIGNMENT_DIAGNOSTIC_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(suite_path),
        "domain": "code",
        "task_count": len(tasks),
        "task_set_sha256": _sha256_object(tasks),
        "context_length": model.config.context_length,
        "first_response_token": {
            "end_anchored": _summarize_position(end_first),
            "generation_aligned": _summarize_position(aligned_first),
        },
        "all_response_tokens": {
            "end_anchored": _summarize_position(end_all),
            "generation_aligned": _summarize_position(aligned_all),
        },
        "free_running_prefix": {
            "first_token_correct": sum(int(length >= 1) for length in prefix_lengths),
            "first_token_accuracy": sum(int(length >= 1) for length in prefix_lengths) / len(prefix_lengths),
            "mean_correct_prefix_tokens": sum(prefix_lengths) / len(prefix_lengths),
            "max_correct_prefix_tokens": max(prefix_lengths),
            "histogram": histogram,
        },
        "per_response_position": per_position,
        "task_measurement_sha256": _sha256_object(task_results),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure absolute-position mismatch between M6 training and free generation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-limit", type=int, default=8)
    args = parser.parse_args()
    result = diagnose_alignment(
        checkpoint=args.checkpoint,
        suite_path=args.suite,
        device=args.device,
        sample_limit=args.sample_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
