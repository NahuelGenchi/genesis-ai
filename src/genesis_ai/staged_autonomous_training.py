from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from .aligned_training import batch_from_indices
from .autonomous_training import (
    TARGET_TOKENS_PER_STEP,
    TRAINING_POLICY_VERSION,
    _hash_indices,
    _seed_from_plan,
    build_focus_schedule,
    validate_inputs,
)
from .checkpoint import export_inference_checkpoint, load_model, restore_training_state, save_checkpoint
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .scale_training import BASE_LR, CPU_THREADS, GRAD_CLIP, _lr
from .terminated_training import TerminatedGenerationAlignedDataset

STAGED_IMPLEMENTATION_VERSION = "staged-autonomous-training-v1"


def _resume_metadata(payload: dict[str, Any], *, policy: dict[str, Any], total_steps: int) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("resume checkpoint metadata is missing")
    staged = metadata.get("staged_training")
    if not isinstance(staged, dict):
        raise ValueError("resume checkpoint is not a staged autonomous checkpoint")
    if staged.get("implementation_version") != STAGED_IMPLEMENTATION_VERSION:
        raise ValueError("resume checkpoint staged implementation version mismatch")
    if metadata.get("training_policy") != TRAINING_POLICY_VERSION:
        raise ValueError("resume checkpoint training policy mismatch")
    if metadata.get("plan_sha256") != policy["plan_sha256"]:
        raise ValueError("resume checkpoint plan mismatch")
    if metadata.get("parent_checkpoint_sha256") != policy["parent_checkpoint_sha256"]:
        raise ValueError("resume checkpoint parent mismatch")
    if int(staged.get("total_steps", -1)) != total_steps:
        raise ValueError("resume checkpoint total-step contract mismatch")
    return staged


def train_staged_continuation(
    *,
    parent_checkpoint: str | Path,
    curriculum_lock: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    export_path: str | Path | None = None,
    run_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    stop_after_steps: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("staged autonomous continuation is frozen to deterministic CPU")
    if stop_after_steps is not None and stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)

    model, tokenizer, records, curriculum, policy, parent_payload = validate_inputs(
        parent_checkpoint=parent_checkpoint,
        curriculum_lock=curriculum_lock,
        records_path=records_path,
        public_data=public_data,
        tokenizer_path=tokenizer_path,
    )
    seed = _seed_from_plan(policy["plan_sha256"])
    random.seed(seed)
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, foreach=False, fused=False)
    dataset = TerminatedGenerationAlignedDataset(records, tokenizer, model.config.context_length)
    public_dataset = TokenDataset(
        public_data,
        tokenizer,
        model.config.context_length,
        split="train",
        min_chars=policy["public_min_chars"],
    )

    batch_size = TARGET_TOKENS_PER_STEP // model.config.context_length
    raw_steps = math.ceil(policy["target_training_tokens"] / TARGET_TOKENS_PER_STEP)
    total_steps = math.ceil(raw_steps / 5) * 5
    procedural_steps = total_steps * 4 // 5
    public_steps = total_steps // 5
    procedural_updates = procedural_steps * batch_size
    schedule, accounting = build_focus_schedule(
        dataset,
        records,
        focus_domain=policy["focus_domain"],
        total_samples=procedural_updates,
        seed=seed + 10,
        continuation_weights=policy["continuation_update_weights"],
    )
    public_generator = torch.Generator(device="cpu").manual_seed(seed + 20)

    completed_steps = 0
    schedule_cursor = 0
    procedural_loss_sum = 0.0
    public_loss_sum = 0.0
    if resume_checkpoint is not None:
        resumed_model, resume_payload = load_model(resume_checkpoint, device)
        if resumed_model.config.to_dict() != model.config.to_dict():
            raise ValueError("resume checkpoint model config mismatch")
        model = resumed_model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, foreach=False, fused=False)
        staged = _resume_metadata(resume_payload, policy=policy, total_steps=total_steps)
        completed_steps = int(staged["completed_steps"])
        schedule_cursor = int(staged["schedule_cursor"])
        procedural_loss_sum = float(staged["procedural_loss_sum"])
        public_loss_sum = float(staged["public_loss_sum"])
        if completed_steps < 0 or completed_steps >= total_steps:
            raise ValueError("resume checkpoint completed-step count is invalid")
        expected_cursor = sum(batch_size for step in range(1, completed_steps + 1) if (step - 1) % 5 < 4)
        if schedule_cursor != expected_cursor:
            raise ValueError("resume checkpoint schedule cursor mismatch")
        restore_training_state(resume_payload, optimizer, public_generator)

    end_step = total_steps if stop_after_steps is None else min(total_steps, completed_steps + stop_after_steps)
    model.train()
    for step in range(completed_steps + 1, end_step + 1):
        learning_rate = _lr(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        procedural_step = (step - 1) % 5 < 4
        if procedural_step:
            indices = schedule[schedule_cursor : schedule_cursor + batch_size]
            schedule_cursor += batch_size
            x, y = batch_from_indices(dataset, indices)
        else:
            x, y = sample_batch(public_dataset, batch_size, public_generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        observed = float(loss.detach().cpu())
        if procedural_step:
            procedural_loss_sum += observed
        else:
            public_loss_sum += observed
        if step == completed_steps + 1 or step % 250 == 0 or step == end_step:
            kind = "procedural-autonomous" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")

    if end_step == total_steps and schedule_cursor != len(schedule):
        raise AssertionError("autonomous schedule was not consumed exactly")

    parent_step = int(parent_payload.get("step", 0))
    metadata = {
        "training_policy": TRAINING_POLICY_VERSION,
        "plan_sha256": policy["plan_sha256"],
        "parent_checkpoint_sha256": policy["parent_checkpoint_sha256"],
        "parent_parameter_count": policy["parent_parameter_count"],
        "curriculum_lock_sha256": policy["curriculum_lock_sha256"],
        "public_min_chars": policy["public_min_chars"],
        "staged_training": {
            "implementation_version": STAGED_IMPLEMENTATION_VERSION,
            "completed_steps": end_step,
            "total_steps": total_steps,
            "schedule_cursor": schedule_cursor,
            "procedural_loss_sum": procedural_loss_sum,
            "public_loss_sum": public_loss_sum,
        },
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=parent_step + end_step,
        metadata=metadata,
        tokenizer=tokenizer,
        batch_generator=public_generator,
    )

    complete = end_step == total_steps
    result: dict[str, Any] = {
        "format_version": "1.0",
        "implementation_version": STAGED_IMPLEMENTATION_VERSION,
        "training_policy": TRAINING_POLICY_VERSION,
        "plan_sha256": policy["plan_sha256"],
        "seed": seed,
        "parent_checkpoint_sha256": policy["parent_checkpoint_sha256"],
        "completed_steps": end_step,
        "steps": total_steps,
        "complete": complete,
        "schedule_cursor": schedule_cursor,
        "schedule_sha256": _hash_indices(schedule),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cash_compute_cost_usd": 0.0,
    }
    if not complete:
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    if export_path is None or run_path is None:
        raise ValueError("complete staged training requires --export and --run")
    export_inference_checkpoint(checkpoint_path, export_path)
    result.update(
        {
            "focus_domain": policy["focus_domain"],
            "focus_examples": policy["focus_examples"],
            "replay_examples_per_domain": policy["replay_examples_per_domain"],
            "continuation_update_weights": policy["continuation_update_weights"],
            "parent_parameter_count": policy["parent_parameter_count"],
            "parent_step": parent_step,
            "final_step": parent_step + total_steps,
            "parameter_count": model.parameter_count(),
            "target_training_tokens": policy["target_training_tokens"],
            "processed_tokens": total_steps * TARGET_TOKENS_PER_STEP,
            "batch_size": batch_size,
            "procedural_steps": procedural_steps,
            "public_steps": public_steps,
            "procedural_updates": procedural_updates,
            "public_min_chars": policy["public_min_chars"],
            "public_document_count": public_dataset.document_count,
            "public_token_count": public_dataset.token_count,
            "schedule_unique_updates": int(len(torch.unique(schedule))),
            "schedule_accounting": accounting,
            "mean_procedural_training_loss": procedural_loss_sum / procedural_steps,
            "mean_public_training_loss": public_loss_sum / public_steps,
            "curriculum_lock_sha256": policy["curriculum_lock_sha256"],
            "records_sha256": policy["records_sha256"],
            "public_manifest_sha256": policy["public_manifest_sha256"],
            "tokenizer_sha256": policy["tokenizer_sha256"],
            "inference_checkpoint_sha256": sha256_file(export_path),
            "determinism": {
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": True,
                "adamw_foreach": False,
                "adamw_fused": False,
            },
        }
    )
    Path(run_path).parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(run_path).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return result


def _equal_nested(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_equal_nested(a, b) for a, b in zip(left, right))
    return left == right


def compare_training_checkpoints(left: str | Path, right: str | Path, output: str | Path) -> dict[str, Any]:
    a = torch.load(left, map_location="cpu", weights_only=False)
    b = torch.load(right, map_location="cpu", weights_only=False)
    checks = {
        "model_equal": _equal_nested(a.get("model"), b.get("model")),
        "optimizer_equal": _equal_nested(a.get("optimizer"), b.get("optimizer")),
        "step_equal": a.get("step") == b.get("step"),
        "rng_equal": _equal_nested(a.get("rng"), b.get("rng")),
    }
    result = {
        "format_version": "1.0",
        "implementation_version": STAGED_IMPLEMENTATION_VERSION,
        "checks": checks,
        "trajectory_identity": all(checks.values()),
        "cash_compute_cost_usd": 0.0,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["trajectory_identity"]:
        raise SystemExit(2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify trajectory-preserving staged autonomous continuation training.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--parent", type=Path, required=True)
    train.add_argument("--curriculum-lock", type=Path, required=True)
    train.add_argument("--records", type=Path, required=True)
    train.add_argument("--public-data", type=Path, required=True)
    train.add_argument("--tokenizer", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--export", type=Path)
    train.add_argument("--run", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after-steps", type=int)
    train.add_argument("--device", default="cpu")
    compare = sub.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        compare_training_checkpoints(args.left, args.right, args.output)
        return
    train_staged_continuation(
        parent_checkpoint=args.parent,
        curriculum_lock=args.curriculum_lock,
        records_path=args.records,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        export_path=args.export,
        run_path=args.run,
        resume_checkpoint=args.resume,
        stop_after_steps=args.stop_after_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
