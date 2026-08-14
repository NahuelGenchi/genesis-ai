from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from .challenger import DOMAINS
from .checkpoint import load_model, tokenizer_from_payload
from .domain_selection import generate_domain_tasks, oracle_response
from .ingest import sha256_file
from .position_alignment import rolling_target_loss
from .verifiers import verify_task

SUITE_VERSION = "m6-domain-selection-v2"
LADDER_SUITE_PATTERN = re.compile(r"^m6-domain-ladder-d([1-5])-v1$")
SELECTION_RULE = "highest_exact_accuracy_then_lowest_terminated_oracle_loss_then_domain_name"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_terminated_suite(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format_version",
        "suite_version",
        "base_seed",
        "tasks_per_domain",
        "difficulty",
        "domains",
        "generation",
        "termination",
        "selection_rule",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid terminated domain suite")
    if raw.get("format_version") != "1.0":
        raise ValueError("unsupported terminated domain suite")
    suite_version = raw.get("suite_version")
    ladder_match = LADDER_SUITE_PATTERN.fullmatch(str(suite_version))
    if suite_version != SUITE_VERSION and ladder_match is None:
        raise ValueError("unsupported terminated domain suite")
    if raw.get("selection_rule") != SELECTION_RULE:
        raise ValueError("unsupported terminated selection rule")
    if not isinstance(raw["base_seed"], int) or isinstance(raw["base_seed"], bool):
        raise ValueError("base_seed must be an integer")
    if not isinstance(raw["tasks_per_domain"], int) or isinstance(raw["tasks_per_domain"], bool) or raw["tasks_per_domain"] <= 0:
        raise ValueError("tasks_per_domain must be positive")
    if not isinstance(raw["difficulty"], int) or isinstance(raw["difficulty"], bool) or not 1 <= raw["difficulty"] <= 5:
        raise ValueError("difficulty must be in [1,5]")
    if ladder_match is not None and int(ladder_match.group(1)) != int(raw["difficulty"]):
        raise ValueError("ladder suite version difficulty does not match suite difficulty")
    domains = raw["domains"]
    if not isinstance(domains, list) or sorted(domains) != sorted(DOMAINS) or len(set(domains)) != len(domains):
        raise ValueError("suite must contain each supported domain exactly once")

    generation = raw["generation"]
    if not isinstance(generation, dict) or set(generation) != {"max_new_tokens", "temperature", "top_k", "seed"}:
        raise ValueError("invalid generation policy")
    if not isinstance(generation["max_new_tokens"], int) or isinstance(generation["max_new_tokens"], bool) or generation["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    if generation.get("top_k") != 1:
        raise ValueError("terminated suite requires deterministic top_k=1")
    if not isinstance(generation["temperature"], (int, float)) or isinstance(generation["temperature"], bool) or generation["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if not isinstance(generation["seed"], int) or isinstance(generation["seed"], bool):
        raise ValueError("generation seed must be an integer")

    termination = raw["termination"]
    if not isinstance(termination, dict) or set(termination) != {"delimiter", "required"}:
        raise ValueError("invalid termination policy")
    if termination.get("delimiter") != "\n" or termination.get("required") is not True:
        raise ValueError("v2 requires a generated newline terminator")
    return raw


def split_generated_answer(raw_text: str, delimiter: str) -> tuple[str, bool]:
    if not delimiter:
        raise ValueError("delimiter must be non-empty")
    if delimiter not in raw_text:
        return raw_text.strip(), False
    answer, _ = raw_text.split(delimiter, 1)
    return answer.strip(), True


def generate_until_terminated(
    *,
    model,
    tokenizer,
    prompt_ids: list[int],
    generation: dict[str, Any],
    delimiter: str,
    domain_ordinal: int,
    task_ordinal: int,
    device: str,
) -> tuple[str, str, bool, int]:
    run_seed = int(generation["seed"]) + domain_ordinal * 10000 + task_ordinal
    torch.manual_seed(run_seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    raw_text = ""
    generated_count = 0
    for _ in range(int(generation["max_new_tokens"])):
        tokens = model.generate(
            tokens,
            1,
            temperature=float(generation["temperature"]),
            top_k=int(generation["top_k"]),
        )
        generated_count += 1
        generated_ids = tokens[0, len(prompt_ids) :].tolist()
        raw_text = tokenizer.decode(generated_ids, errors="replace")
        if delimiter in raw_text:
            break
    answer, terminated = split_generated_answer(raw_text, delimiter)
    return raw_text, answer, terminated, generated_count


def evaluate_terminated_domain(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    generation: dict[str, Any],
    termination: dict[str, Any],
    domain_ordinal: int,
    device: str,
) -> dict[str, Any]:
    delimiter = str(termination["delimiter"])
    exact = 0
    terminated_count = 0
    loss_sum = 0.0
    target_tokens = 0
    responses: list[str] = []
    raw_responses: list[str] = []
    oracles: list[str] = []
    terminated_oracles: list[str] = []
    generated_token_counts: list[int] = []

    model.eval()
    for task_ordinal, task in enumerate(tasks):
        oracle = oracle_response(task)
        terminated_oracle = oracle + delimiter
        oracles.append(oracle)
        terminated_oracles.append(terminated_oracle)
        prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
        if not prompt_ids:
            raise ValueError("task prompt encoded to zero tokens")
        raw, answer, terminated, generated_count = generate_until_terminated(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            generation=generation,
            delimiter=delimiter,
            domain_ordinal=domain_ordinal,
            task_ordinal=task_ordinal,
            device=device,
        )
        passed = terminated and verify_task(task, answer).passed
        exact += int(passed)
        terminated_count += int(terminated)
        responses.append(answer)
        raw_responses.append(raw)
        generated_token_counts.append(generated_count)

        rolling = rolling_target_loss(
            model=model,
            tokenizer=tokenizer,
            task=task,
            response=terminated_oracle,
            device=device,
        )
        loss_sum += float(rolling["loss_sum"])
        target_tokens += int(rolling["target_tokens"])

    task_count = len(tasks)
    if task_count <= 0 or target_tokens <= 0:
        raise ValueError("terminated domain produced no tasks/targets")
    return {
        "task_count": task_count,
        "task_set_sha256": _sha256_object(tasks),
        "oracle_set_sha256": _sha256_object(oracles),
        "terminated_oracle_set_sha256": _sha256_object(terminated_oracles),
        "response_set_sha256": _sha256_object(responses),
        "raw_response_set_sha256": _sha256_object(raw_responses),
        "exact_correct": exact,
        "exact_accuracy": exact / task_count,
        "terminated_count": terminated_count,
        "termination_rate": terminated_count / task_count,
        "unterminated_count": task_count - terminated_count,
        "mean_generated_tokens": sum(generated_token_counts) / task_count,
        "terminated_oracle_target_tokens": target_tokens,
        "terminated_oracle_loss": loss_sum / target_tokens,
    }


def run_terminated_selection(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    suite = load_terminated_suite(suite_path)
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    domains: dict[str, dict[str, Any]] = {}
    for domain_ordinal, domain in enumerate(suite["domains"]):
        tasks = generate_domain_tasks(
            domain=domain,
            seed=int(suite["base_seed"]) + domain_ordinal,
            count=int(suite["tasks_per_domain"]),
            difficulty=int(suite["difficulty"]),
        )
        domains[domain] = evaluate_terminated_domain(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            generation=suite["generation"],
            termination=suite["termination"],
            domain_ordinal=domain_ordinal,
            device=device,
        )
    selected = min(
        suite["domains"],
        key=lambda domain: (
            -float(domains[domain]["exact_accuracy"]),
            float(domains[domain]["terminated_oracle_loss"]),
            domain,
        ),
    )
    return {
        "format_version": "1.0",
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(suite_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "difficulty": suite["difficulty"],
        "tasks_per_domain": suite["tasks_per_domain"],
        "generation": suite["generation"],
        "termination": suite["termination"],
        "selection_rule": suite["selection_rule"],
        "domains": domains,
        "selected_domain": selected,
        "selected_reason": {
            "exact_accuracy": domains[selected]["exact_accuracy"],
            "terminated_oracle_loss": domains[selected]["terminated_oracle_loss"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Genesis with a frozen stop-aware domain suite.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_terminated_selection(checkpoint=args.checkpoint, suite_path=args.suite, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
