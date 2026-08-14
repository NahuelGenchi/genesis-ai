from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from .candidate import ExperienceDataset
from .checkpoint import export_inference_checkpoint, save_checkpoint
from .config import ModelConfig
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer

TRAINING_POLICY_VERSION = "m6-micro-2m-training-v1"
SEED = 97001
BASE_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 50
GRAD_CLIP = 1.0


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            if not isinstance(value.get("prompt"), str) or not isinstance(value.get("response"), str):
                raise ValueError(f"{path.name}:{line_number} requires prompt/response")
            if value.get("curriculum") != "m6-code-curriculum-v1":
                raise ValueError("unexpected procedural curriculum version")
            provenance = value.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("kind") != "procedural_oracle":
                raise ValueError("procedural record provenance is invalid")
            records.append(value)
    if not records:
        raise ValueError("procedural records are empty")
    return records


def _stage_config(ladder: dict[str, Any], tokenizer: ByteBPETokenizer) -> tuple[ModelConfig, int]:
    stages = ladder.get("stages")
    if not isinstance(stages, list):
        raise ValueError("scaling ladder stages missing")
    stage = next((item for item in stages if isinstance(item, dict) and item.get("name") == "micro-2m"), None)
    if not isinstance(stage, dict) or stage.get("role") != "candidate":
        raise ValueError("micro-2m stage missing from scaling ladder")
    raw = stage.get("config")
    if not isinstance(raw, dict):
        raise ValueError("micro-2m config missing")
    config = ModelConfig(vocab_size=tokenizer.vocab_size, **raw)
    config.validate()
    target_tokens = stage.get("training_tokens")
    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool) or target_tokens <= 0:
        raise ValueError("micro-2m target training tokens invalid")
    return config, target_tokens


def validate_training_inputs(
    *,
    ladder_definition_path: str | Path,
    ladder_result_path: str | Path,
    curriculum_lock_path: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
) -> tuple[ModelConfig, ByteBPETokenizer, list[dict[str, Any]], dict[str, Any]]:
    ladder_definition_path = Path(ladder_definition_path)
    ladder_result_path = Path(ladder_result_path)
    curriculum_lock_path = Path(curriculum_lock_path)
    records_path = Path(records_path)
    public_data = Path(public_data)
    tokenizer_path = Path(tokenizer_path)

    ladder = _load_json(ladder_definition_path)
    ladder_result = _load_json(ladder_result_path)
    curriculum = _load_json(curriculum_lock_path)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)

    if ladder.get("name") != "m6-scaling-ladder-v1" or ladder_result.get("experiment") != "m6-scaling-ladder-v1":
        raise ValueError("unsupported scaling ladder")
    if ladder_result.get("definition_sha256") != sha256_file(ladder_definition_path):
        raise ValueError("scaling ladder definition does not match measured result")
    if ladder_result.get("next_stage") != "micro-2m":
        raise ValueError("micro-2m is not the currently authorized stage")
    if curriculum.get("curriculum_version") != "m6-code-curriculum-v1" or curriculum.get("selected_domain") != "code":
        raise ValueError("unsupported curriculum lock")
    if curriculum.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("curriculum requires non-zero cash compute")
    if curriculum.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("curriculum tokenizer does not match supplied tokenizer")
    separation = curriculum.get("evaluation_separation")
    if not isinstance(separation, dict) or separation.get("exact_prompt_overlap_count") != 0:
        raise ValueError("curriculum/evaluation separation is not clean")
    training = curriculum.get("training")
    if not isinstance(training, dict):
        raise ValueError("curriculum training block missing")
    procedural = training.get("procedural")
    if not isinstance(procedural, dict):
        raise ValueError("curriculum procedural block missing")
    if procedural.get("truncation_policy") != "left_prefix_only_preserve_all_response_tokens":
        raise ValueError("unsupported curriculum truncation policy")
    if procedural.get("records_file_sha256") != sha256_file(records_path):
        raise ValueError("procedural records do not match frozen curriculum")
    if curriculum.get("public_text", {}).get("manifest_sha256") != sha256_file(public_data / "manifest.json"):
        raise ValueError("public corpus does not match frozen curriculum")

    records = _read_records(records_path)
    if len(records) != procedural.get("examples"):
        raise ValueError("procedural record count does not match frozen curriculum")

    config, stage_target_tokens = _stage_config(ladder, tokenizer)
    if stage_target_tokens != training.get("target_training_tokens"):
        raise ValueError("ladder/curriculum target token budgets differ")
    measured_stage = next(
        (item for item in ladder_result.get("stages", []) if isinstance(item, dict) and item.get("name") == "micro-2m"),
        None,
    )
    if not isinstance(measured_stage, dict) or measured_stage.get("local_cpu_feasible") is not True:
        raise ValueError("micro-2m was not measured locally feasible")
    expected_parameters = measured_stage.get("parameter_count")
    if expected_parameters != 1_895_808:
        raise ValueError("measured micro-2m parameter count changed")

    model = GenesisLM(config)
    if model.parameter_count() != expected_parameters:
        raise ValueError("constructed micro-2m model parameter count mismatch")

    policy = {
        "target_training_tokens": stage_target_tokens,
        "target_tokens_per_step": int(ladder["target_tokens_per_step"]),
        "procedural_fraction": float(training["procedural_batch_fraction"]),
        "public_fraction": float(training["public_text_batch_fraction"]),
        "expected_parameters": expected_parameters,
        "ladder_definition_sha256": sha256_file(ladder_definition_path),
        "ladder_result_sha256": sha256_file(ladder_result_path),
        "curriculum_lock_sha256": sha256_file(curriculum_lock_path),
        "procedural_records_sha256": sha256_file(records_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
    }
    if not math.isclose(policy["procedural_fraction"], 0.8) or not math.isclose(policy["public_fraction"], 0.2):
        raise ValueError("M6 training v1 requires frozen 80/20 batch mix")
    if policy["target_tokens_per_step"] != 1024:
        raise ValueError("M6 training v1 requires 1024 processed tokens per step")
    return config, tokenizer, records, policy


def _lr(step: int, total_steps: int) -> float:
    if step <= WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def _loss_on_items(model: GenesisLM, items: list[tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in items:
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_micro_2m(
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
    config, tokenizer, records, policy = validate_training_inputs(
        ladder_definition_path=ladder_definition_path,
        ladder_result_path=ladder_result_path,
        curriculum_lock_path=curriculum_lock_path,
        records_path=records_path,
        public_data=public_data,
        tokenizer_path=tokenizer_path,
    )
    if seed != SEED:
        raise ValueError(f"M6 training v1 seed is frozen to {SEED}")

    batch_size = policy["target_tokens_per_step"] // config.context_length
    if batch_size * config.context_length != policy["target_tokens_per_step"]:
        raise ValueError("target tokens/step is not divisible by context length")
    raw_steps = math.ceil(policy["target_training_tokens"] / policy["target_tokens_per_step"])
    total_steps = math.ceil(raw_steps / 5) * 5
    processed_tokens = total_steps * policy["target_tokens_per_step"]
    procedural_steps = total_steps * 4 // 5
    public_steps = total_steps // 5

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    procedural_dataset = ExperienceDataset(records, tokenizer, config.context_length)
    public_dataset = TokenDataset(public_data, tokenizer, config.context_length, split="train")
    procedural_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    public_generator = torch.Generator(device="cpu").manual_seed(seed + 2)

    procedural_probe_items = list(procedural_dataset.items[: min(32, len(procedural_dataset.items))])
    public_probe_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    public_probe_x, public_probe_y = sample_batch(public_dataset, min(8, batch_size), public_probe_generator)
    public_probe_items = [(x, y) for x, y in zip(public_probe_x, public_probe_y)]
    procedural_loss_before = _loss_on_items(model, procedural_probe_items, device)
    public_loss_before = _loss_on_items(model, public_probe_items, device)

    proc_loss_sum = 0.0
    public_loss_sum = 0.0
    proc_updates = 0
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
            proc_loss_sum += observed
            proc_updates += 1
        else:
            public_loss_sum += observed
            public_updates += 1
        if step == 1 or step % 100 == 0 or step == total_steps:
            kind = "procedural" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")

    procedural_loss_after = _loss_on_items(model, procedural_probe_items, device)
    public_loss_after = _loss_on_items(model, public_probe_items, device)
    result: dict[str, Any] = {
        "format_version": "1.0",
        "training_policy": TRAINING_POLICY_VERSION,
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
        "mean_procedural_training_loss": proc_loss_sum / proc_updates,
        "mean_public_training_loss": public_loss_sum / public_updates,
        "procedural_probe_loss_before": procedural_loss_before,
        "procedural_probe_loss_after": procedural_loss_after,
        "public_probe_loss_before": public_loss_before,
        "public_probe_loss_after": public_loss_after,
        "procedural_supervised_response_tokens": procedural_dataset.supervised_tokens,
        "procedural_examples": procedural_dataset.experience_count,
        "public_train_documents": public_dataset.document_count,
        "public_train_tokens": public_dataset.token_count,
        "cash_compute_cost_usd": 0.0,
        "inputs": policy,
    }
    stable_metadata = {"m6_scale_training": result}
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
    parser = argparse.ArgumentParser(description="Train the authorized deterministic Genesis micro-2m candidate.")
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
    result = train_micro_2m(
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
