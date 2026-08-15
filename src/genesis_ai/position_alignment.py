from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import load_model, tokenizer_from_payload
from .domain_selection import _target_loss_sum, generate_domain_tasks, load_suite, oracle_response
from .ingest import sha256_file

DIAGNOSTIC_VERSION = "m6-position-alignment-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def predictor_positions(*, prompt_tokens: int, response_tokens: int, context_length: int) -> dict[str, int]:
    if prompt_tokens <= 0 or response_tokens <= 0 or context_length <= 0:
        raise ValueError("token counts and context length must be positive")
    sequence_tokens = prompt_tokens + response_tokens
    static_window_start = max(0, sequence_tokens - (context_length + 1))
    static_first_predictor = prompt_tokens - 1 - static_window_start
    generation_first_predictor = min(prompt_tokens, context_length) - 1
    if static_first_predictor < 0:
        raise ValueError("static window removed the first response predictor")
    return {
        "static_first_predictor_position": static_first_predictor,
        "generation_first_predictor_position": generation_first_predictor,
        "first_predictor_position_shift": generation_first_predictor - static_first_predictor,
    }


def rolling_target_loss(
    *,
    model,
    tokenizer,
    task: dict[str, Any],
    response: str,
    device: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
    response_ids = tokenizer.encode(response)
    if not prompt_ids or not response_ids:
        raise ValueError("empty prompt/response tokenization")
    if len(response_ids) >= model.config.context_length:
        raise ValueError("response does not fit model context")

    history = list(prompt_ids)
    total_loss = 0.0
    greedy_correct = 0
    first_mismatch: int | None = None
    with torch.no_grad():
        for ordinal, target in enumerate(response_ids):
            context = history[-model.config.context_length :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            next_logits = logits[:, -1, :]
            y = torch.tensor([target], dtype=torch.long, device=device)
            loss = F.cross_entropy(next_logits, y, reduction="sum")
            total_loss += float(loss.detach().cpu())
            predicted = int(torch.argmax(next_logits, dim=-1).item())
            if predicted == target:
                greedy_correct += 1
            elif first_mismatch is None:
                first_mismatch = ordinal
            history.append(target)

    positions = predictor_positions(
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
        context_length=model.config.context_length,
    )
    return {
        "loss_sum": total_loss,
        "target_tokens": len(response_ids),
        "mean_loss": total_loss / len(response_ids),
        "greedy_correct_tokens": greedy_correct,
        "greedy_token_accuracy": greedy_correct / len(response_ids),
        "all_greedy_tokens_correct": greedy_correct == len(response_ids),
        "first_greedy_mismatch_ordinal": first_mismatch,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        **positions,
    }


def rolling_target_diagnostics(
    *,
    model,
    tokenizer,
    task: dict[str, Any],
    response: str,
    device: str,
) -> dict[str, Any]:
    """Return deterministic teacher-forced greedy token traces for one response."""
    prompt_ids = tokenizer.encode(task["prompt"] + "\nAnswer:")
    response_ids = tokenizer.encode(response)
    if not prompt_ids or not response_ids:
        raise ValueError("empty prompt/response tokenization")
    if len(response_ids) >= model.config.context_length:
        raise ValueError("response does not fit model context")

    history = list(prompt_ids)
    predicted_ids: list[int] = []
    with torch.no_grad():
        for target in response_ids:
            context = history[-model.config.context_length :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = model(x)
            predicted_ids.append(int(torch.argmax(logits[:, -1, :], dim=-1).item()))
            history.append(target)

    return {
        "target_ids": [int(value) for value in response_ids],
        "predicted_ids": predicted_ids,
    }


def diagnose_checkpoint(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    domain: str = "code",
    device: str = "cpu",
    sample_limit: int = 8,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    suite_path = Path(suite_path)
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")
    suite = load_suite(suite_path)
    if suite.get("suite_version") != "m6-domain-selection-v1":
        raise ValueError("diagnostic requires immutable m6-domain-selection-v1")
    if domain not in suite["domains"]:
        raise ValueError(f"domain not present in suite: {domain}")
    domain_ordinal = list(suite["domains"]).index(domain)
    tasks = generate_domain_tasks(
        domain=domain,
        seed=int(suite["base_seed"]) + domain_ordinal,
        count=int(suite["tasks_per_domain"]),
        difficulty=int(suite["difficulty"]),
    )

    model, payload = load_model(checkpoint, device)
    tokenizer = tokenizer_from_payload(payload)
    model.eval()

    static_loss_sum = 0.0
    static_tokens = 0
    rolling_loss_sum = 0.0
    rolling_tokens = 0
    rolling_greedy_correct = 0
    rolling_all_correct = 0
    first_token_greedy_correct = 0
    shifted_tasks = 0
    position_shift_sum = 0
    samples: list[dict[str, Any]] = []

    for task in tasks:
        oracle = oracle_response(task)
        legacy_loss, legacy_tokens = _target_loss_sum(model, tokenizer, task, oracle, device)
        rolling = rolling_target_loss(
            model=model,
            tokenizer=tokenizer,
            task=task,
            response=oracle,
            device=device,
        )
        static_loss_sum += legacy_loss
        static_tokens += legacy_tokens
        rolling_loss_sum += float(rolling["loss_sum"])
        rolling_tokens += int(rolling["target_tokens"])
        rolling_greedy_correct += int(rolling["greedy_correct_tokens"])
        rolling_all_correct += int(bool(rolling["all_greedy_tokens_correct"]))
        first_token_greedy_correct += int(rolling["first_greedy_mismatch_ordinal"] != 0)
        shift = int(rolling["first_predictor_position_shift"])
        shifted_tasks += int(shift != 0)
        position_shift_sum += shift
        if len(samples) < sample_limit:
            samples.append(
                {
                    "task_id": task["id"],
                    "oracle": oracle,
                    "legacy_static_mean_loss": legacy_loss / legacy_tokens,
                    **rolling,
                }
            )

    if static_tokens <= 0 or rolling_tokens <= 0 or len(tasks) <= 0:
        raise ValueError("diagnostic produced no target tokens")
    return {
        "format_version": "1.0",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "suite_version": suite["suite_version"],
        "suite_sha256": sha256_file(suite_path),
        "domain": domain,
        "task_count": len(tasks),
        "task_set_sha256": _sha256_object(tasks),
        "legacy_static": {
            "target_tokens": static_tokens,
            "mean_loss": static_loss_sum / static_tokens,
        },
        "generation_aligned_rolling": {
            "target_tokens": rolling_tokens,
            "mean_loss": rolling_loss_sum / rolling_tokens,
            "greedy_correct_tokens": rolling_greedy_correct,
            "greedy_token_accuracy": rolling_greedy_correct / rolling_tokens,
            "all_greedy_tokens_correct_tasks": rolling_all_correct,
            "all_greedy_tokens_correct_rate": rolling_all_correct / len(tasks),
            "first_token_greedy_correct_tasks": first_token_greedy_correct,
            "first_token_greedy_correct_rate": first_token_greedy_correct / len(tasks),
        },
        "position_alignment": {
            "shifted_tasks": shifted_tasks,
            "shifted_task_rate": shifted_tasks / len(tasks),
            "mean_first_predictor_position_shift": position_shift_sum / len(tasks),
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy static oracle loss with generation-aligned rolling loss.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", default="code")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = diagnose_checkpoint(
        checkpoint=args.checkpoint,
        suite_path=args.suite,
        domain=args.domain,
        device=args.device,
        sample_limit=args.sample_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
