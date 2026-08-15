from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .checkpoint import load_model, tokenizer_from_payload
from .domain_selection import generate_domain_tasks, oracle_response
from .ingest import sha256_file
from .position_alignment import rolling_target_diagnostics
from .terminated_eval import generate_until_terminated, load_terminated_suite
from .verifiers import verify_task

DIAGNOSTIC_VERSION = "m6-structured-diagnostic-v1"
EXPECTED_V2_CANDIDATE_SHA256 = "7f95f31d2fa8a1a4e20c2d4fc673df5ffe224bf8e6efcf07a4208cc893532407"


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


def _quartile_index(position: int, length: int) -> int:
    if length <= 0 or not 0 <= position < length:
        raise ValueError("position must be inside a positive-length sequence")
    return min(3, (position * 4) // length)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_diagnostic(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_V2_CANDIDATE_SHA256:
        raise ValueError(
            "diagnostic requires the exact rejected v2 candidate: "
            f"expected {EXPECTED_V2_CANDIDATE_SHA256}, got {checkpoint_sha}"
        )

    suite = load_terminated_suite(suite_path)
    if suite.get("suite_version") != "m6-domain-selection-v2" or int(suite.get("difficulty", -1)) != 1:
        raise ValueError("diagnostic is frozen to the original difficulty-1 v2 suite")
    if suite.get("termination") != {"delimiter": "\n", "required": True}:
        raise ValueError("diagnostic requires the original newline termination contract")

    try:
        structured_ordinal = list(suite["domains"]).index("structured")
    except ValueError as exc:
        raise ValueError("v2 suite is missing structured domain") from exc
    tasks = generate_domain_tasks(
        domain="structured",
        seed=int(suite["base_seed"]) + structured_ordinal,
        count=int(suite["tasks_per_domain"]),
        difficulty=int(suite["difficulty"]),
    )

    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    delimiter = str(suite["termination"]["delimiter"])

    total_oracle_tokens = 0
    total_teacher_correct = 0
    teacher_full_sequences = 0
    teacher_quartile_correct = [0, 0, 0, 0]
    teacher_quartile_total = [0, 0, 0, 0]
    total_free_positional_correct = 0
    total_expected_free_positions = 0
    strict_correct = 0
    terminated = 0
    valid_json = 0
    json_list = 0
    correct_length = 0
    positional_element_correct = 0
    positional_element_total = 0
    multiset_overlap = 0
    multiset_total = 0
    oracle_lengths: list[int] = []
    generated_lengths: list[int] = []
    prefix_lengths: list[int] = []
    normalized_prefixes: list[float] = []
    first_error_positions: list[int] = []
    normalized_first_error_positions: list[float] = []
    teacher_position_correct: list[int] = []
    teacher_position_total: list[int] = []
    task_outcome_hashes: list[str] = []

    for task_ordinal, task in enumerate(tasks):
        expected = task.get("verifier", {}).get("expected")
        if not isinstance(expected, list) or not all(isinstance(value, int) and not isinstance(value, bool) for value in expected):
            raise ValueError("diagnostic expected difficulty-1 integer-list structured task")

        oracle = oracle_response(task)
        terminated_oracle = oracle + delimiter
        teacher = rolling_target_diagnostics(
            model=model,
            tokenizer=tokenizer,
            task=task,
            response=terminated_oracle,
            device=device,
        )
        teacher_targets = [int(value) for value in teacher["target_ids"]]
        teacher_predictions = [int(value) for value in teacher["predicted_ids"]]
        if len(teacher_targets) != len(teacher_predictions) or not teacher_targets:
            raise AssertionError("rolling diagnostic returned invalid token traces")
        oracle_lengths.append(len(teacher_targets))
        total_oracle_tokens += len(teacher_targets)
        teacher_matches = [prediction == target for prediction, target in zip(teacher_predictions, teacher_targets)]
        total_teacher_correct += sum(teacher_matches)
        teacher_full_sequences += int(all(teacher_matches))
        while len(teacher_position_correct) < len(teacher_matches):
            teacher_position_correct.append(0)
            teacher_position_total.append(0)
        for position, matched in enumerate(teacher_matches):
            teacher_position_correct[position] += int(matched)
            teacher_position_total[position] += 1
            quartile = _quartile_index(position, len(teacher_matches))
            teacher_quartile_correct[quartile] += int(matched)
            teacher_quartile_total[quartile] += 1

        prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
        raw_text, answer, did_terminate, generated_count = generate_until_terminated(
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
        strict_pass = did_terminate and verify_task(task, answer).passed
        strict_correct += int(strict_pass)

        expected_answer_ids = tokenizer.encode(oracle)
        generated_answer_ids = tokenizer.encode(answer)
        prefix = _common_prefix_length(expected_answer_ids, generated_answer_ids)
        prefix_lengths.append(prefix)
        normalized_prefixes.append(prefix / max(1, len(expected_answer_ids)))
        first_error_positions.append(min(prefix, len(expected_answer_ids)))
        normalized_first_error_positions.append(min(prefix, len(expected_answer_ids)) / max(1, len(expected_answer_ids)))
        total_expected_free_positions += len(expected_answer_ids)
        total_free_positional_correct += sum(
            int(index < len(generated_answer_ids) and generated_answer_ids[index] == target)
            for index, target in enumerate(expected_answer_ids)
        )

        parsed_valid = False
        parsed_any: Any = None
        try:
            parsed_any = json.loads(answer)
            parsed_valid = True
            valid_json += 1
        except (json.JSONDecodeError, TypeError):
            pass
        parsed_list = parsed_any if parsed_valid and isinstance(parsed_any, list) else None
        if parsed_list is not None:
            json_list += 1
            correct_length += int(len(parsed_list) == len(expected))
            positional_element_total += len(expected)
            positional_element_correct += sum(
                int(index < len(parsed_list) and parsed_list[index] == target)
                for index, target in enumerate(expected)
            )
            multiset_total += len(expected)
            multiset_overlap += _multiset_overlap(parsed_list, expected)

        task_outcome_hashes.append(
            _sha256_object(
                {
                    "task_id": task["id"],
                    "teacher_correct": sum(teacher_matches),
                    "teacher_total": len(teacher_matches),
                    "free_prefix": prefix,
                    "free_expected_tokens": len(expected_answer_ids),
                    "generated_tokens": generated_count,
                    "terminated": did_terminate,
                    "strict": strict_pass,
                    "valid_json": parsed_valid,
                    "json_list": parsed_list is not None,
                    "correct_length": parsed_list is not None and len(parsed_list) == len(expected),
                    "position_correct": (
                        sum(
                            int(index < len(parsed_list) and parsed_list[index] == target)
                            for index, target in enumerate(expected)
                        )
                        if parsed_list is not None
                        else 0
                    ),
                    "multiset_overlap": _multiset_overlap(parsed_list, expected) if parsed_list is not None else 0,
                }
            )
        )

    task_count = len(tasks)
    position_accuracy = [
        correct / total if total else 0.0
        for correct, total in zip(teacher_position_correct, teacher_position_total)
    ]
    quartile_accuracy = [
        correct / total if total else 0.0
        for correct, total in zip(teacher_quartile_correct, teacher_quartile_total)
    ]

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
        "task_outcome_hash_set_sha256": _sha256_object(sorted(task_outcome_hashes)),
        "oracle_tokens": {
            "total": total_oracle_tokens,
            "mean": _mean([float(value) for value in oracle_lengths]),
            "median": float(statistics.median(oracle_lengths)),
            "min": min(oracle_lengths),
            "max": max(oracle_lengths),
        },
        "oracle_context_greedy": {
            "token_correct": total_teacher_correct,
            "token_total": total_oracle_tokens,
            "token_accuracy": total_teacher_correct / total_oracle_tokens,
            "full_sequence_correct": teacher_full_sequences,
            "full_sequence_accuracy": teacher_full_sequences / task_count,
            "position_accuracy": position_accuracy,
            "quartile_position_accuracy": quartile_accuracy,
            "quartile_token_totals": teacher_quartile_total,
        },
        "free_generation": {
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / task_count,
            "terminated": terminated,
            "termination_rate": terminated / task_count,
            "mean_generated_tokens": _mean([float(value) for value in generated_lengths]),
            "positional_token_correct": total_free_positional_correct,
            "expected_token_positions": total_expected_free_positions,
            "positional_token_accuracy": total_free_positional_correct / max(1, total_expected_free_positions),
            "mean_oracle_prefix_tokens": _mean([float(value) for value in prefix_lengths]),
            "mean_normalized_oracle_prefix": _mean(normalized_prefixes),
            "median_normalized_oracle_prefix": float(statistics.median(normalized_prefixes)),
            "mean_first_error_token_position": _mean([float(value) for value in first_error_positions]),
            "mean_normalized_first_error_position": _mean(normalized_first_error_positions),
        },
        "structured_semantics": {
            "valid_json_count": valid_json,
            "valid_json_rate": valid_json / task_count,
            "json_list_count": json_list,
            "json_list_rate": json_list / task_count,
            "correct_list_length_count": correct_length,
            "correct_list_length_rate": correct_length / task_count,
            "positional_element_correct": positional_element_correct,
            "positional_element_total": positional_element_total,
            "positional_element_accuracy": positional_element_correct / max(1, positional_element_total),
            "multiset_overlap": multiset_overlap,
            "multiset_total": multiset_total,
            "multiset_overlap_rate": multiset_overlap / max(1, multiset_total),
        },
        "promotion": {"allowed": False, "reason": "diagnostic-only; no benchmark or promotion semantics changed"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the exact rejected M6 v2 structured candidate.")
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
