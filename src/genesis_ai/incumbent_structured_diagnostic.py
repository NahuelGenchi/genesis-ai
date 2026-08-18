from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_model, tokenizer_from_payload
from .domain_selection import generate_domain_tasks, oracle_response
from .ingest import sha256_file
from .position_alignment import rolling_target_diagnostics
from .terminated_eval import generate_until_terminated, load_terminated_suite
from .verifiers import verify_task

DIAGNOSTIC_VERSION = "m6-incumbent-structured-diagnostic-v1"
EXPECTED_INCUMBENT_SHA256 = "0ba16a931451ffbdc369aa685845b0ddb9b6ed03910ac773b50e837fd9886d7e"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _multiset_overlap(actual: list[Any], expected: list[Any]) -> int:
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in actual + expected):
        return 0
    actual_counts = collections.Counter(actual)
    expected_counts = collections.Counter(expected)
    return sum(min(actual_counts[value], expected_counts[value]) for value in expected_counts)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_diagnostic(*, checkpoint: str | Path, suite_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_INCUMBENT_SHA256:
        raise ValueError(
            "incumbent diagnostic requires the exact promoted checkpoint: "
            f"expected {EXPECTED_INCUMBENT_SHA256}, got {checkpoint_sha}"
        )

    suite = load_terminated_suite(suite_path)
    if suite.get("suite_version") != "m6-domain-selection-v2" or int(suite.get("difficulty", -1)) != 1:
        raise ValueError("diagnostic is frozen to the original difficulty-1 v2 suite")
    if suite.get("termination") != {"delimiter": "\n", "required": True}:
        raise ValueError("diagnostic requires the original newline termination contract")

    structured_ordinal = list(suite["domains"]).index("structured")
    tasks = generate_domain_tasks(
        domain="structured",
        seed=int(suite["base_seed"]) + structured_ordinal,
        count=int(suite["tasks_per_domain"]),
        difficulty=int(suite["difficulty"]),
    )

    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    delimiter = str(suite["termination"]["delimiter"])

    oracle_lengths: list[int] = []
    teacher_correct = 0
    teacher_total = 0
    teacher_full = 0
    quartile_correct = [0, 0, 0, 0]
    quartile_total = [0, 0, 0, 0]
    strict_correct = 0
    terminated = 0
    generated_lengths: list[int] = []
    prefix_lengths: list[int] = []
    normalized_prefixes: list[float] = []
    valid_json = 0
    json_lists = 0
    correct_lengths = 0
    positional_correct = 0
    positional_total = 0
    multiset_overlap = 0
    multiset_total = 0
    task_hashes: list[str] = []

    for task_ordinal, task in enumerate(tasks):
        expected = task.get("verifier", {}).get("expected")
        if not isinstance(expected, list):
            raise ValueError("structured diagnostic expected list-valued oracle")

        oracle = oracle_response(task)
        teacher = rolling_target_diagnostics(
            model=model,
            tokenizer=tokenizer,
            task=task,
            response=oracle + delimiter,
            device=device,
        )
        targets = [int(value) for value in teacher["target_ids"]]
        predictions = [int(value) for value in teacher["predicted_ids"]]
        matches = [left == right for left, right in zip(predictions, targets)]
        if not targets or len(matches) != len(targets):
            raise AssertionError("invalid rolling diagnostic token trace")
        oracle_lengths.append(len(targets))
        teacher_correct += sum(matches)
        teacher_total += len(matches)
        teacher_full += int(all(matches))
        for position, matched in enumerate(matches):
            quartile = min(3, (position * 4) // len(matches))
            quartile_correct[quartile] += int(matched)
            quartile_total[quartile] += 1

        prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
        _, answer, did_terminate, generated_count = generate_until_terminated(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            generation=suite["generation"],
            delimiter=delimiter,
            domain_ordinal=structured_ordinal,
            task_ordinal=task_ordinal,
            device=device,
        )
        terminated += int(did_terminate)
        generated_lengths.append(generated_count)
        strict = did_terminate and verify_task(task, answer).passed
        strict_correct += int(strict)

        expected_ids = tokenizer.encode(oracle)
        generated_ids = tokenizer.encode(answer)
        prefix = _common_prefix_length(expected_ids, generated_ids)
        prefix_lengths.append(prefix)
        normalized_prefixes.append(prefix / max(1, len(expected_ids)))

        parsed: Any = None
        parsed_valid = False
        try:
            parsed = json.loads(answer)
            parsed_valid = True
            valid_json += 1
        except (json.JSONDecodeError, TypeError):
            pass
        parsed_list = parsed if parsed_valid and isinstance(parsed, list) else None
        if parsed_list is not None:
            json_lists += 1
            correct_lengths += int(len(parsed_list) == len(expected))
            positional_total += len(expected)
            positional_correct += sum(
                int(index < len(parsed_list) and parsed_list[index] == target)
                for index, target in enumerate(expected)
            )
            multiset_total += len(expected)
            multiset_overlap += _multiset_overlap(parsed_list, expected)

        task_hashes.append(
            _sha256_object({
                "task_id": task["id"],
                "teacher_correct": sum(matches),
                "teacher_total": len(matches),
                "free_prefix": prefix,
                "generated_tokens": generated_count,
                "terminated": did_terminate,
                "strict": strict,
                "valid_json": parsed_valid,
                "json_list": parsed_list is not None,
                "correct_length": parsed_list is not None and len(parsed_list) == len(expected),
                "position_correct": sum(
                    int(index < len(parsed_list) and parsed_list[index] == target)
                    for index, target in enumerate(expected)
                ) if parsed_list is not None else 0,
                "multiset_overlap": _multiset_overlap(parsed_list, expected) if parsed_list is not None else 0,
            })
        )

    task_count = len(tasks)
    return {
        "format_version": "1.0",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(suite_path),
        "domain": "structured",
        "difficulty": 1,
        "task_count": task_count,
        "task_set_sha256": _sha256_object(tasks),
        "task_outcome_hash_set_sha256": _sha256_object(sorted(task_hashes)),
        "oracle_tokens": {
            "total": teacher_total,
            "mean": _mean([float(value) for value in oracle_lengths]),
            "median": float(statistics.median(oracle_lengths)),
            "min": min(oracle_lengths),
            "max": max(oracle_lengths),
        },
        "oracle_context_greedy": {
            "token_correct": teacher_correct,
            "token_total": teacher_total,
            "token_accuracy": teacher_correct / teacher_total,
            "full_sequence_correct": teacher_full,
            "full_sequence_accuracy": teacher_full / task_count,
            "quartile_position_accuracy": [
                correct / total if total else 0.0
                for correct, total in zip(quartile_correct, quartile_total)
            ],
            "quartile_token_totals": quartile_total,
        },
        "free_generation": {
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / task_count,
            "terminated": terminated,
            "termination_rate": terminated / task_count,
            "mean_generated_tokens": _mean([float(value) for value in generated_lengths]),
            "mean_oracle_prefix_tokens": _mean([float(value) for value in prefix_lengths]),
            "mean_normalized_oracle_prefix": _mean(normalized_prefixes),
            "median_normalized_oracle_prefix": float(statistics.median(normalized_prefixes)),
        },
        "structured_semantics": {
            "valid_json_count": valid_json,
            "valid_json_rate": valid_json / task_count,
            "json_list_count": json_lists,
            "json_list_rate": json_lists / task_count,
            "correct_list_length_count": correct_lengths,
            "correct_list_length_rate": correct_lengths / task_count,
            "positional_element_correct": positional_correct,
            "positional_element_total": positional_total,
            "positional_element_accuracy": positional_correct / max(1, positional_total),
            "multiset_overlap": multiset_overlap,
            "multiset_total": multiset_total,
            "multiset_overlap_rate": multiset_overlap / max(1, multiset_total),
        },
        "runtime_contract": {
            "torch": torch.__version__,
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "training_performed": False,
        "cash_compute_cost_usd": 0.0,
        "promotion": {"allowed": False, "reason": "diagnostic-only; frozen evaluation and promotion semantics unchanged"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the exact promoted incumbent on the frozen structured suite.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_diagnostic(checkpoint=args.checkpoint, suite_path=args.suite, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
