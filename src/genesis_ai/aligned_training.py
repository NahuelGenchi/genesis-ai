from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

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
    _loss_on_items,
    _lr,
    validate_training_inputs,
)
from .tokenizer import ByteBPETokenizer

ALIGNED_TRAINING_POLICY_VERSION = "m6-micro-2m-generation-aligned-v1"


class GenerationAlignedExperienceDataset(Dataset):
    """One response target per rolling context, exactly matching free generation."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: ByteBPETokenizer,
        context_length: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        contexts: list[torch.Tensor] = []
        targets: list[int] = []
        record_item_counts: list[int] = []
        prompt_lengths: list[int] = []
        response_lengths: list[int] = []

        for record in records:
            prompt = record.get("prompt")
            response = record.get("response")
            if not isinstance(prompt, str) or not prompt or not isinstance(response, str) or not response:
                raise ValueError("aligned experience requires non-empty prompt/response")
            prompt_ids = tokenizer.encode(prompt + "\nAnswer:")
            response_ids = tokenizer.encode(response)
            if not prompt_ids or not response_ids:
                raise ValueError("aligned experience tokenized to an empty sequence")
            prompt_lengths.append(len(prompt_ids))
            response_lengths.append(len(response_ids))
            count_before = len(contexts)
            for response_index, target in enumerate(response_ids):
                history = prompt_ids + response_ids[:response_index]
                if len(history) < context_length:
                    raise ValueError(
                        "generation-aligned v1 requires every rolling history to fill the learned-position context"
                    )
                context = history[-context_length:]
                if len(context) != context_length:
                    raise AssertionError("rolling context length mismatch")
                contexts.append(torch.tensor(context, dtype=torch.long))
                targets.append(int(target))
            record_item_counts.append(len(contexts) - count_before)

        if not contexts:
            raise ValueError("generation-aligned experience is empty")
        self.contexts = tuple(contexts)
        self.targets = tuple(targets)
        self.context_length = context_length
        self.experience_count = len(records)
        self.supervised_response_tokens = len(targets)
        self.record_item_counts = tuple(record_item_counts)
        self.prompt_length_min = min(prompt_lengths)
        self.prompt_length_max = max(prompt_lengths)
        self.response_length_min = min(response_lengths)
        self.response_length_max = max(response_lengths)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        x = self.contexts[index]
        y = torch.full((self.context_length,), IGNORE_INDEX, dtype=torch.long)
        y[-1] = self.targets[index]
        return x, y


def _aligned_probe_items(dataset: GenerationAlignedExperienceDataset, limit: int = 64) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [dataset[index] for index in range(min(limit, len(dataset)))]


def train_generation_aligned_micro_2m(
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
        raise ValueError("generation-aligned training v1 is frozen to CPU for exact reproducibility")
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
        raise ValueError(f"generation-aligned training v1 seed is frozen to {SEED}")

    batch_size = policy["target_tokens_per_step"] // config.context_length
    if batch_size * config.context_length != policy["target_tokens_per_step"]:
        raise ValueError("target tokens/step is not divisible by context length")
    raw_steps = math.ceil(policy["target_training_tokens"] / policy["target_tokens_per_step"])
    total_steps = math.ceil(raw_steps / 5) * 5
    processed_tokens = total_steps * policy["target_tokens_per_step"]
    procedural_steps = total_steps * 4 // 5
    public_steps = total_steps // 5
    if procedural_steps / total_steps != 0.8 or public_steps / total_steps != 0.2:
        raise ValueError("aligned training schedule does not preserve exact 80/20 mix")

    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, foreach=False, fused=False)
    procedural_dataset = GenerationAlignedExperienceDataset(records, tokenizer, config.context_length)
    public_dataset = TokenDataset(public_data, tokenizer, config.context_length, split="train")
    procedural_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    public_generator = torch.Generator(device="cpu").manual_seed(seed + 2)

    procedural_probe_items = _aligned_probe_items(procedural_dataset)
    public_probe_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    public_probe_x, public_probe_y = sample_batch(public_dataset, min(8, batch_size), public_probe_generator)
    public_probe_items = [(x, y) for x, y in zip(public_probe_x, public_probe_y)]
    procedural_loss_before = _loss_on_items(model, procedural_probe_items, device)
    public_loss_before = _loss_on_items(model, public_probe_items, device)

    procedural_loss_sum = 0.0
    public_loss_sum = 0.0
    procedural_updates = 0
    public_updates = 0
    model.train()
    for step in range(1, total_steps + 1):
        learning_rate = _lr(step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        procedural_step = (step - 1) % 5 < 4
        if procedural_step:
            x, y = sample_batch(procedural_dataset, batch_size, procedural_generator)
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
            procedural_updates += 1
        else:
            public_loss_sum += observed
            public_updates += 1
        if step == 1 or step % 100 == 0 or step == total_steps:
            kind = "procedural-aligned" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")

    procedural_loss_after = _loss_on_items(model, procedural_probe_items, device)
    public_loss_after = _loss_on_items(model, public_probe_items, device)
    sampled_procedural_targets = procedural_steps * batch_size
    result: dict[str, Any] = {
        "format_version": "1.0",
        "training_policy": ALIGNED_TRAINING_POLICY_VERSION,
        "scientific_parent_policy": "m6-micro-2m-training-v2",
        "changed_variable": "procedural_response_target_positioning",
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
            "policy": "rolling_last_context_predict_final_position",
            "target_model_position": config.context_length - 1,
            "dataset_items": len(procedural_dataset),
            "frozen_oracle_response_tokens": procedural_dataset.supervised_response_tokens,
            "sampled_procedural_targets": sampled_procedural_targets,
            "sampled_target_fraction_of_dataset": sampled_procedural_targets / len(procedural_dataset),
            "prompt_length_min": procedural_dataset.prompt_length_min,
            "prompt_length_max": procedural_dataset.prompt_length_max,
            "response_length_min": procedural_dataset.response_length_min,
            "response_length_max": procedural_dataset.response_length_max,
        },
        "mean_procedural_training_loss": procedural_loss_sum / procedural_updates,
        "mean_public_training_loss": public_loss_sum / public_updates,
        "procedural_probe_loss_before": procedural_loss_before,
        "procedural_probe_loss_after": procedural_loss_after,
        "public_probe_loss_before": public_loss_before,
        "public_probe_loss_after": public_loss_after,
        "procedural_examples": procedural_dataset.experience_count,
        "public_train_documents": public_dataset.document_count,
        "public_train_tokens": public_dataset.token_count,
        "cash_compute_cost_usd": 0.0,
        "inputs": policy,
    }
    stable_metadata = {"m6_generation_aligned_training": result}
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        metadata=stable_metadata,
        tokenizer=tokenizer,
        batch_generator=procedural_generator,
    )
    export_inference_checkpoint(checkpoint_path, export_path)
    result["inference_checkpoint_sha256"] = sha256_file(Path(export_path))
    result["inference_checkpoint_size_bytes"] = Path(export_path).stat().st_size
    run_path = Path(run_path)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the generation-aligned Genesis micro-2m candidate.")
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
    result = train_generation_aligned_micro_2m(
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
