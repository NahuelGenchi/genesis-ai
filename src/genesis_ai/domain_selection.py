from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .challenger import DOMAINS, build_task
from .checkpoint import load_model, tokenizer_from_payload
from .ingest import sha256_file
from .verifiers import verify_task

SUITE_VERSION = "m6-domain-selection-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_suite(path: str | Path) -> dict[str, Any]:
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
        "selection_rule",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid domain-selection suite")
    if raw.get("format_version") != "1.0" or raw.get("suite_version") != SUITE_VERSION:
        raise ValueError("unsupported domain-selection suite")
    if raw.get("selection_rule") != "highest_exact_accuracy_then_lowest_oracle_target_loss_then_domain_name":
        raise ValueError("unsupported domain-selection rule")
    if not isinstance(raw["base_seed"], int) or isinstance(raw["base_seed"], bool):
        raise ValueError("base_seed must be an integer")
    if not isinstance(raw["tasks_per_domain"], int) or isinstance(raw["tasks_per_domain"], bool) or raw["tasks_per_domain"] <= 0:
        raise ValueError("tasks_per_domain must be positive")
    if not isinstance(raw["difficulty"], int) or isinstance(raw["difficulty"], bool) or not 1 <= raw["difficulty"] <= 5:
        raise ValueError("difficulty must be in [1,5]")
    domains = raw["domains"]
    if not isinstance(domains, list) or sorted(domains) != sorted(DOMAINS) or len(set(domains)) != len(domains):
        raise ValueError("suite must contain each supported domain exactly once")
    generation = raw["generation"]
    if not isinstance(generation, dict) or set(generation) != {"max_new_tokens", "temperature", "top_k", "seed"}:
        raise ValueError("invalid generation policy")
    if not isinstance(generation["max_new_tokens"], int) or isinstance(generation["max_new_tokens"], bool) or generation["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not isinstance(generation["top_k"], int) or isinstance(generation["top_k"], bool) or generation["top_k"] != 1:
        raise ValueError("selection suite requires greedy top_k=1")
    if not isinstance(generation["temperature"], (int, float)) or isinstance(generation["temperature"], bool) or generation["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if not isinstance(generation["seed"], int) or isinstance(generation["seed"], bool):
        raise ValueError("generation seed must be an integer")
    return raw


def generate_domain_tasks(*, domain: str, seed: int, count: int, difficulty: int) -> list[dict[str, Any]]:
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain}")
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(tasks) < count and attempts < count * 100:
        attempts += 1
        task = build_task(rng, domain, difficulty)
        if task["id"] in seen:
            continue
        seen.add(task["id"])
        task["selection_generation"] = {"seed": seed, "ordinal": len(tasks), "attempt": attempts}
        tasks.append(task)
    if len(tasks) != count:
        raise RuntimeError("domain task generation exhausted duplicate budget")
    return tasks


def oracle_response(task: dict[str, Any]) -> str:
    verifier = task.get("verifier")
    if not isinstance(verifier, dict):
        raise ValueError("task verifier missing")
    kind = verifier.get("kind")
    if kind == "integer_exact":
        response = str(verifier["expected"])
    elif kind == "json_exact":
        response = _canonical(verifier["expected"])
    elif kind == "restricted_expression":
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or "compute: " not in prompt:
            raise ValueError("restricted-expression prompt is not canonical")
        response = prompt.rsplit("compute: ", 1)[1]
        if response.endswith("."):
            response = response[:-1]
        response = response.replace("^", "**")
    else:
        raise ValueError(f"unsupported verifier kind: {kind}")
    result = verify_task(task, response)
    if not result.passed:
        raise ValueError(f"derived oracle does not pass verifier for {task.get('id')}: {result.reason}")
    return response


def _target_loss_sum(model, tokenizer, task: dict[str, Any], response: str, device: str) -> tuple[float, int]:
    prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
    response_ids = tokenizer.encode(response)
    if not prompt_ids or not response_ids:
        raise ValueError("empty prompt/response tokenization")
    sequence = prompt_ids + response_ids
    if len(response_ids) >= model.config.context_length:
        raise ValueError("oracle response does not fit model context")
    window_start = max(0, len(sequence) - (model.config.context_length + 1))
    window = sequence[window_start:]
    if len(window) < 2:
        raise ValueError("oracle scoring window too short")
    x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
    target_values = window[1:]
    labels: list[int] = []
    supervised = 0
    for local_target_index, token in enumerate(target_values, start=1):
        global_target_index = window_start + local_target_index
        if global_target_index < len(prompt_ids):
            labels.append(-100)
        else:
            labels.append(token)
            supervised += 1
    if supervised != len(response_ids):
        raise ValueError("oracle response was truncated during scoring")
    y = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
        total = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
    return float(total.detach().cpu()), supervised


def evaluate_domain(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    generation: dict[str, Any],
    domain_ordinal: int,
    device: str,
) -> dict[str, Any]:
    exact = 0
    loss_sum = 0.0
    target_tokens = 0
    responses: list[str] = []
    oracles: list[str] = []
    model.eval()
    for ordinal, task in enumerate(tasks):
        oracle = oracle_response(task)
        oracles.append(oracle)
        prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
        if not prompt_ids:
            raise ValueError("task prompt encoded to zero tokens")
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
        response = tokenizer.decode(output[len(prompt_ids):], errors="replace").strip()
        responses.append(response)
        exact += int(verify_task(task, response).passed)
        task_loss, task_tokens = _target_loss_sum(model, tokenizer, task, oracle, device)
        loss_sum += task_loss
        target_tokens += task_tokens
    if target_tokens <= 0:
        raise ValueError("domain produced no target tokens")
    return {
        "task_count": len(tasks),
        "task_set_sha256": _sha256_object(tasks),
        "oracle_set_sha256": _sha256_object(oracles),
        "response_set_sha256": _sha256_object(responses),
        "exact_correct": exact,
        "exact_accuracy": exact / len(tasks),
        "oracle_target_tokens": target_tokens,
        "oracle_target_loss": loss_sum / target_tokens,
    }


def run_selection(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    suite = load_suite(suite_path)
    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    per_domain: dict[str, dict[str, Any]] = {}
    for domain_ordinal, domain in enumerate(suite["domains"]):
        tasks = generate_domain_tasks(
            domain=domain,
            seed=int(suite["base_seed"]) + domain_ordinal,
            count=int(suite["tasks_per_domain"]),
            difficulty=int(suite["difficulty"]),
        )
        per_domain[domain] = evaluate_domain(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            generation=suite["generation"],
            domain_ordinal=domain_ordinal,
            device=device,
        )
    selected = min(
        suite["domains"],
        key=lambda domain: (
            -float(per_domain[domain]["exact_accuracy"]),
            float(per_domain[domain]["oracle_target_loss"]),
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
        "selection_rule": suite["selection_rule"],
        "domains": per_domain,
        "selected_domain": selected,
        "selected_reason": {
            "exact_accuracy": per_domain[selected]["exact_accuracy"],
            "oracle_target_loss": per_domain[selected]["oracle_target_loss"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Genesis's first useful domain from a frozen verifier-backed suite.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run_selection(checkpoint=args.checkpoint, suite_path=args.suite, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
