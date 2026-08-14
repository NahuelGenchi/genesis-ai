from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from .candidate import IGNORE_INDEX
from .checkpoint import export_inference_checkpoint, save_checkpoint
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .scale_training import (
    BASE_LR,
    CPU_THREADS,
    GRAD_CLIP,
    MIN_LR,
    SEED,
    WARMUP_STEPS,
    _lr,
    validate_training_inputs,
)
from .tokenizer import ByteBPETokenizer

ALIGNED_DATASET_VERSION = "generation-aligned-response-v1"
ALIGNED_TRAINING_POLICY_VERSION = "m6-micro-2m-aligned-training-v1"


class GenerationAlignedExperienceDataset:
    """One response target per context using the exact sliding window used by generation."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: ByteBPETokenizer, context_length: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        encoded: list[tuple[list[int], list[int]]] = []
        total_targets = 0
        for record in records:
            prompt_ids = tokenizer.encode(record["prompt"] + "\nAnswer:")
            response_ids = tokenizer.encode(record["response"])
            if not prompt_ids:
                raise ValueError(f"experience {record.get('id')} has zero prompt tokens")
            if not response_ids:
                raise ValueError(f"experience {record.get('id')} has zero response tokens")
            encoded.append((prompt_ids, response_ids))
            total_targets += len(response_ids)
        if total_targets <= 0:
            raise ValueError("generation-aligned dataset has no response targets")

        contexts = torch.zeros((total_targets, context_length), dtype=torch.long)
        targets = torch.empty(total_targets, dtype=torch.long)
        predictor_positions = torch.empty(total_targets, dtype=torch.long)
        response_ordinals = torch.empty(total_targets, dtype=torch.long)
        record_ordinals = torch.empty(total_targets, dtype=torch.long)
        first_target_indices: list[int] = []
        record_ranges: list[tuple[int, int]] = []

        cursor = 0
        for record_ordinal, (prompt_ids, response_ids) in enumerate(encoded):
            start = cursor
            history = list(prompt_ids)
            for response_ordinal, target in enumerate(response_ids):
                context = history[-context_length:]
                if not context:
                    raise ValueError("generation context is empty")
                contexts[cursor, : len(context)] = torch.tensor(context, dtype=torch.long)
                targets[cursor] = int(target)
                predictor_positions[cursor] = len(context) - 1
                response_ordinals[cursor] = response_ordinal
                record_ordinals[cursor] = record_ordinal
                if response_ordinal == 0:
                    first_target_indices.append(cursor)
                history.append(int(target))
                cursor += 1
            record_ranges.append((start, cursor))

        if cursor != total_targets:
            raise AssertionError("generation-aligned dataset target count mismatch")
        self.contexts = contexts
        self.targets = targets
        self.predictor_positions = predictor_positions
        self.response_ordinals = response_ordinals
        self.record_ordinals = record_ordinals
        self.first_target_indices = tuple(first_target_indices)
        first_set = set(first_target_indices)
        self.continuation_indices = tuple(index for index in range(total_targets) if index not in first_set)
        self.record_ranges = tuple(record_ranges)
        self.experience_count = len(records)
        self.supervised_tokens = total_targets
        self.context_length = context_length
        self.dataset_version = ALIGNED_DATASET_VERSION

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        y = torch.full((self.context_length,), IGNORE_INDEX, dtype=torch.long)
        y[int(self.predictor_positions[index])] = self.targets[index]
        return self.contexts[index], y


def _sha256_indices(indices: torch.Tensor) -> str:
    rendered = ",".join(str(int(value)) for value in indices.tolist()).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def build_aligned_schedule(
    dataset: GenerationAlignedExperienceDataset,
    *,
    total_samples: int,
    seed: int,
) -> torch.Tensor:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    first = torch.tensor(dataset.first_target_indices, dtype=torch.long)
    continuation = torch.tensor(dataset.continuation_indices, dtype=torch.long)
    if total_samples < len(first):
        raise ValueError("aligned schedule must cover every first response target")
    if total_samples > len(dataset):
        raise ValueError("aligned schedule v1 forbids duplicate target contexts")
    remaining = total_samples - len(first)
    if remaining > len(continuation):
        raise ValueError("insufficient unique continuation targets")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = first[torch.randperm(len(first), generator=generator)]
    if remaining:
        continuation = continuation[torch.randperm(len(continuation), generator=generator)[:remaining]]
        selected = torch.cat((first, continuation))
    else:
        selected = first
    selected = selected[torch.randperm(len(selected), generator=generator)]
    if len(torch.unique(selected)) != len(selected):
        raise AssertionError("aligned schedule unexpectedly contains duplicate contexts")
    selected_first = int((dataset.response_ordinals[selected] == 0).sum().item())
    if selected_first != dataset.experience_count:
        raise AssertionError("aligned schedule did not preserve first-target coverage")
    return selected


def batch_from_indices(
    dataset: GenerationAlignedExperienceDataset,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [dataset[int(index)] for index in indices]
    if not pairs:
        raise ValueError("aligned batch is empty")
    return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs])


def _loss_on_indices(
    model: GenesisLM,
    dataset: GenerationAlignedExperienceDataset,
    indices: torch.Tensor,
    device: str,
) -> float:
    if len(indices) <= 0:
        raise ValueError("probe indices are empty")
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for index in indices.tolist():
            x, y = dataset[int(index)]
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
            assert loss is not None
            total += float(loss.detach().cpu())
    if was_training:
        model.train()
    return total / len(indices)


def train_aligned_micro_2m(
    *,
    ladder_definition_path: str | Path,
    ladder_result_path: str | Path,
    curriculum_lock_path: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    export_path: str | Path,
    run_path: str | Path,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("M6 aligned training v1 is frozen to CPU for exact reproducibility")
    torch.set_num_threads(CPU_THREADS)
    config, tokenizer, records, policy = validate_training_inputs(
        ladder_definition_path=ladder_definition_path,
        ladder_result_path=ladder_result_path,
        curriculum_lock_path=curriculum_lock_path,
        records_path=records_path,
        public_data=public_data,
        tokenizer_path=tokenizer_path,
    )
    if seed != SEED:
        raise ValueError(f"M6 aligned training v1 seed is frozen to {SEED}")

    batch_size = policy["target_tokens_per_step"] // config.context_length
    if batch_size * config.context_length != policy["target_tokens_per_step"]:
        raise ValueError("target tokens/step is not divisible by context length")
    raw_steps = math.ceil(policy["target_training_tokens"] / policy["target_tokens_per_step"])
    total_steps = math.ceil(raw_steps / 5) * 5
    processed_tokens = total_steps * policy["target_tokens_per_step"]
    procedural_steps = total_steps * 4 // 5
    public_steps = total_steps // 5
    procedural_updates = procedural_steps * batch_size

    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, foreach=False, fused=False)
    procedural_dataset = GenerationAlignedExperienceDataset(records, tokenizer, config.context_length)
    expected_response_tokens = int(
        json.loads(Path(curriculum_lock_path).read_text(encoding="utf-8"))["training"]["procedural"]["response_tokens"]
    )
    if procedural_dataset.supervised_tokens != expected_response_tokens:
        raise ValueError("aligned dataset does not preserve frozen response-token count")
    schedule = build_aligned_schedule(
        procedural_dataset,
        total_samples=procedural_updates,
        seed=seed + 1,
    )
    public_dataset = TokenDataset(public_data, tokenizer, config.context_length, split="train")
    public_generator = torch.Generator(device="cpu").manual_seed(seed + 2)

    probe_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    probe_count = min(256, len(procedural_dataset))
    probe_indices = torch.randperm(len(procedural_dataset), generator=probe_generator)[:probe_count]
    public_probe_generator = torch.Generator(device="cpu").manual_seed(seed + 4)
    public_probe_x, public_probe_y = sample_batch(public_dataset, min(8, batch_size), public_probe_generator)
    public_probe_items = [(x, y) for x, y in zip(public_probe_x, public_probe_y)]

    aligned_loss_before = _loss_on_indices(model, procedural_dataset, probe_indices, device)
    was_training = model.training
    model.eval()
    public_losses: list[float] = []
    with torch.no_grad():
        for x, y in public_probe_items:
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
            assert loss is not None
            public_losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    public_loss_before = sum(public_losses) / len(public_losses)

    proc_loss_sum = 0.0
    public_loss_sum = 0.0
    proc_steps_observed = 0
    public_steps_observed = 0
    schedule_cursor = 0
    model.train()
    for step in range(1, total_steps + 1):
        learning_rate = _lr(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        procedural_step = (step - 1) % 5 < 4
        if procedural_step:
            indices = schedule[schedule_cursor : schedule_cursor + batch_size]
            if len(indices) != batch_size:
                raise AssertionError("aligned schedule exhausted unexpectedly")
            schedule_cursor += batch_size
            x, y = batch_from_indices(procedural_dataset, indices)
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
            proc_loss_sum += observed
            proc_steps_observed += 1
        else:
            public_loss_sum += observed
            public_steps_observed += 1
        if step == 1 or step % 100 == 0 or step == total_steps:
            kind = "procedural-aligned" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")
    if schedule_cursor != len(schedule):
        raise AssertionError("aligned schedule was not consumed exactly")

    aligned_loss_after = _loss_on_indices(model, procedural_dataset, probe_indices, device)
    was_training = model.training
    model.eval()
    public_losses = []
    with torch.no_grad():
        for x, y in public_probe_items:
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
            assert loss is not None
            public_losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    public_loss_after = sum(public_losses) / len(public_losses)

    selected_response_ordinals = procedural_dataset.response_ordinals[schedule]
    first_updates = int((selected_response_ordinals == 0).sum().item())
    continuation_updates = len(schedule) - first_updates
    result: dict[str, Any] = {
        "format_version": "1.0",
        "training_policy": ALIGNED_TRAINING_POLICY_VERSION,
        "dataset_version": ALIGNED_DATASET_VERSION,
        "seed": seed,
        "architecture": config.to_dict(),
        "parameter_count": model.parameter_count(),
        "steps": total_steps,
        "batch_size": batch_size,
        "context_length": config.context_length,
        "processed_tokens": processed_tokens,
        "target_training_tokens": policy["target_training_tokens"],
        "processed_token_overhead": processed_tokens - policy["target_training_tokens"],
        "procedural_steps": procedural_steps,
        "public_steps": public_steps,
        "procedural_step_fraction": procedural_steps / total_steps,
        "public_step_fraction": public_steps / total_steps,
        "base_learning_rate": BASE_LR,
        "minimum_learning_rate": MIN_LR,
        "warmup_steps": WARMUP_STEPS,
        "gradient_clip": GRAD_CLIP,
        "determinism": {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "adamw_foreach": False,
            "adamw_fused": False,
        },
        "alignment": {
            "dataset_response_targets": len(procedural_dataset),
            "schedule_target_updates": len(schedule),
            "schedule_unique_updates": int(len(torch.unique(schedule))),
            "schedule_sha256": _sha256_indices(schedule),
            "first_response_targets": len(procedural_dataset.first_target_indices),
            "first_response_target_updates": first_updates,
            "first_response_target_coverage": first_updates / len(procedural_dataset.first_target_indices),
            "continuation_target_updates": continuation_updates,
            "one_target_per_generation_context": True,
            "right_padding_after_predictor_only": True,
        },
        "mean_procedural_training_loss": proc_loss_sum / proc_steps_observed,
        "mean_public_training_loss": public_loss_sum / public_steps_observed,
        "aligned_probe_loss_before": aligned_loss_before,
        "aligned_probe_loss_after": aligned_loss_after,
        "public_probe_loss_before": public_loss_before,
        "public_probe_loss_after": public_loss_after,
        "procedural_supervised_response_tokens": procedural_dataset.supervised_tokens,
        "procedural_examples": procedural_dataset.experience_count,
        "public_train_documents": public_dataset.document_count,
        "public_train_tokens": public_dataset.token_count,
        "cash_compute_cost_usd": 0.0,
        "inputs": policy,
    }
    stable_metadata = {"m6_aligned_scale_training": result}
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        metadata=stable_metadata,
        tokenizer=tokenizer,
        batch_generator=public_generator,
    )
    export_inference_checkpoint(checkpoint_path, export_path)
    result["inference_checkpoint_sha256"] = sha256_file(Path(export_path))
    result["inference_checkpoint_size_bytes"] = Path(export_path).stat().st_size
    run_path = Path(run_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the M6 generation-aligned micro-2m candidate.")
    parser.add_argument("--ladder-definition", type=Path, required=True)
    parser.add_argument("--ladder-result", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_aligned_micro_2m(
        ladder_definition_path=args.ladder_definition,
        ladder_result_path=args.ladder_result,
        curriculum_lock_path=args.curriculum_lock,
        records_path=args.records,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        export_path=args.export,
        run_path=args.run,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
