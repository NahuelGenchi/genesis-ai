from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .aligned_training import batch_from_indices
from .checkpoint import export_inference_checkpoint, save_checkpoint
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .model import GenesisLM
from .scale_5m_contract import evidence_hashes, load_scale_contract
from .scale_5m_curriculum import CURRICULUM_VERSION
from .terminated_training import TerminatedGenerationAlignedDataset
from .tokenizer import ByteBPETokenizer

TRAINING_POLICY_VERSION = "m6-scale-5m-rope-training-v1"
DOMAINS = ("code", "math", "structured")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"records:{line_number} must be an object")
            if value.get("curriculum") != CURRICULUM_VERSION or value.get("domain") not in DOMAINS:
                raise ValueError("unexpected ~5M curriculum record")
            provenance = value.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("kind") != "procedural_oracle":
                raise ValueError("~5M record provenance must be procedural_oracle")
            records.append(value)
    if not records:
        raise ValueError("~5M records are empty")
    return records


def validate_inputs(
    *,
    experiment_path: str | Path,
    finalist_path: str | Path,
    preflight_path: str | Path,
    curriculum_path: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
) -> tuple[dict[str, Any], Any, ByteBPETokenizer, list[dict[str, Any]], dict[str, Any]]:
    experiment, _, _, config = load_scale_contract(
        experiment_path=experiment_path,
        finalist_path=finalist_path,
        preflight_path=preflight_path,
    )
    curriculum = _load_json(curriculum_path)
    if curriculum.get("format_version") != "1.0" or curriculum.get("curriculum_version") != CURRICULUM_VERSION:
        raise ValueError("unsupported ~5M curriculum lock")
    if curriculum.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("~5M curriculum violates zero-cash contract")
    hashes = evidence_hashes(
        experiment_path=experiment_path,
        finalist_path=finalist_path,
        preflight_path=preflight_path,
    )
    for key, value in hashes.items():
        if curriculum.get(key) != value:
            raise ValueError(f"~5M curriculum evidence hash drifted: {key}")
    if curriculum.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("~5M curriculum tokenizer hash mismatch")
    if curriculum.get("public_text", {}).get("manifest_sha256") != sha256_file(Path(public_data) / "manifest.json"):
        raise ValueError("~5M public corpus manifest mismatch")
    separation = curriculum.get("ladder_separation")
    if not isinstance(separation, dict) or separation.get("exact_training_prompt_overlap_count") != 0:
        raise ValueError("~5M training data overlaps frozen GCI-Ladder prompts")
    procedural = curriculum.get("procedural")
    if not isinstance(procedural, dict) or procedural.get("records_file_sha256") != sha256_file(records_path):
        raise ValueError("~5M records do not match curriculum lock")

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("~5M tokenizer vocabulary mismatch")
    records = _read_records(records_path)
    training = experiment["training"]
    expected_per_domain = int(training["examples_per_domain"])
    counts = {domain: 0 for domain in DOMAINS}
    for record in records:
        counts[str(record["domain"])] += 1
    if counts != {domain: expected_per_domain for domain in DOMAINS}:
        raise ValueError(f"~5M curriculum is not exactly balanced: {counts}")
    if int(procedural.get("examples", -1)) != len(records):
        raise ValueError("~5M curriculum record count mismatch")
    return experiment, config, tokenizer, records, curriculum


def _hash_indices(indices: torch.Tensor) -> str:
    payload = ",".join(str(int(value)) for value in indices.tolist()).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_balanced_unique_schedule(
    dataset: TerminatedGenerationAlignedDataset,
    records: list[dict[str, Any]],
    *,
    total_samples: int,
    seed: int,
    examples_per_domain: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    anchors = list(dataset.anchor_indices)
    expected_anchor_per_domain = examples_per_domain * 2
    expected_anchors = expected_anchor_per_domain * len(DOMAINS)
    if len(anchors) != expected_anchors:
        raise ValueError(f"~5M expected {expected_anchors} first/terminator anchors, got {len(anchors)}")
    if total_samples < expected_anchors:
        raise ValueError("~5M procedural budget cannot cover mandatory first/terminator anchors")
    if total_samples > len(dataset):
        raise ValueError("~5M training forbids duplicate procedural target contexts")

    domain_by_record = [str(record["domain"]) for record in records]
    anchor_by_domain = {domain: 0 for domain in DOMAINS}
    for index in anchors:
        record_ordinal = int(dataset.record_ordinals[index])
        anchor_by_domain[domain_by_record[record_ordinal]] += 1
    if anchor_by_domain != {domain: expected_anchor_per_domain for domain in DOMAINS}:
        raise ValueError(f"~5M anchor accounting drifted: {anchor_by_domain}")

    continuation_by_domain: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    for index in dataset.continuation_indices:
        record_ordinal = int(dataset.record_ordinals[index])
        continuation_by_domain[domain_by_record[record_ordinal]].append(int(index))
    remaining = total_samples - expected_anchors
    base = remaining // len(DOMAINS)
    remainder = remaining % len(DOMAINS)
    quotas = {domain: base for domain in DOMAINS}
    for domain in DOMAINS[:remainder]:
        quotas[domain] += 1

    rng = random.Random(seed)
    selected_extra: list[int] = []
    available = {domain: len(values) for domain, values in continuation_by_domain.items()}
    for domain in DOMAINS:
        pool = continuation_by_domain[domain]
        rng.shuffle(pool)
        if len(pool) < quotas[domain]:
            raise ValueError(
                f"~5M requires {quotas[domain]} unique {domain} continuation contexts; only {len(pool)} available"
            )
        selected_extra.extend(pool[: quotas[domain]])
    selected = torch.tensor(anchors + selected_extra, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    selected = selected[torch.randperm(len(selected), generator=generator)]
    if len(torch.unique(selected)) != len(selected):
        raise AssertionError("~5M procedural schedule contains duplicate target contexts")
    return selected, {
        "anchor_updates_by_domain": anchor_by_domain,
        "continuation_available_by_domain": available,
        "continuation_updates_by_domain": quotas,
        "total_updates_by_domain": {domain: anchor_by_domain[domain] + quotas[domain] for domain in DOMAINS},
        "total_anchor_updates": expected_anchors,
        "total_continuation_updates": remaining,
        "total_updates": total_samples,
    }


def _learning_rate(step: int, total_steps: int, *, base: float, minimum: float, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return base * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def train_scale_5m(
    *,
    experiment_path: str | Path,
    finalist_path: str | Path,
    preflight_path: str | Path,
    curriculum_path: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    export_path: str | Path,
    run_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("~5M v1 full training is frozen to deterministic CPU")
    experiment, config, tokenizer, records, curriculum = validate_inputs(
        experiment_path=experiment_path,
        finalist_path=finalist_path,
        preflight_path=preflight_path,
        curriculum_path=curriculum_path,
        records_path=records_path,
        public_data=public_data,
        tokenizer_path=tokenizer_path,
    )
    training = experiment["training"]
    threads = int(training["torch_threads"])
    seed = int(training["seed"])
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)

    model = GenesisLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        foreach=False,
        fused=False,
    )
    procedural_dataset = TerminatedGenerationAlignedDataset(records, tokenizer, config.context_length)
    public_dataset = TokenDataset(public_data, tokenizer, config.context_length, split="train")
    target_tokens_per_step = int(training["target_tokens_per_step"])
    batch_size = target_tokens_per_step // config.context_length
    if batch_size * config.context_length != target_tokens_per_step:
        raise ValueError("~5M target tokens/step is not divisible by context length")
    raw_steps = math.ceil(int(training["target_training_tokens"]) / target_tokens_per_step)
    total_steps = math.ceil(raw_steps / 5) * 5
    processed_tokens = total_steps * target_tokens_per_step
    procedural_steps = total_steps * 2 // 5
    public_steps = total_steps * 3 // 5
    if procedural_steps + public_steps != total_steps:
        raise AssertionError("~5M 40/60 step accounting drifted")
    procedural_updates = procedural_steps * batch_size
    schedule, accounting = build_balanced_unique_schedule(
        procedural_dataset,
        records,
        total_samples=procedural_updates,
        seed=seed + 10,
        examples_per_domain=int(training["examples_per_domain"]),
    )
    public_generator = torch.Generator(device="cpu").manual_seed(seed + 20)

    base_lr = float(training["learning_rate"])
    minimum_lr = float(training["minimum_learning_rate"])
    warmup_steps = int(training["warmup_steps"])
    grad_clip = float(training["gradient_clip"])
    schedule_cursor = 0
    procedural_loss_sum = 0.0
    public_loss_sum = 0.0
    model.train()
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        learning_rate = _learning_rate(
            step,
            total_steps,
            base=base_lr,
            minimum=minimum_lr,
            warmup_steps=warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        procedural_step = (step - 1) % 5 < 2
        if procedural_step:
            indices = schedule[schedule_cursor : schedule_cursor + batch_size]
            schedule_cursor += batch_size
            x, y = batch_from_indices(procedural_dataset, indices)
        else:
            x, y = sample_batch(public_dataset, batch_size, public_generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        observed = float(loss.detach().cpu())
        if procedural_step:
            procedural_loss_sum += observed
        else:
            public_loss_sum += observed
        if step == 1 or step % 500 == 0 or step == total_steps:
            kind = "procedural-scale5m" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")
    elapsed = time.perf_counter() - started
    if schedule_cursor != len(schedule):
        raise AssertionError("~5M procedural schedule was not consumed exactly")

    metadata = {
        "training_policy": TRAINING_POLICY_VERSION,
        **evidence_hashes(
            experiment_path=experiment_path,
            finalist_path=finalist_path,
            preflight_path=preflight_path,
        ),
        "curriculum_sha256": sha256_file(curriculum_path),
        "records_sha256": sha256_file(records_path),
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        metadata=metadata,
        tokenizer=tokenizer,
        batch_generator=public_generator,
    )
    export_inference_checkpoint(checkpoint_path, export_path)
    result = {
        "format_version": "1.0",
        "training_policy": TRAINING_POLICY_VERSION,
        "initialization": "genesis-random-from-scratch",
        "seed": seed,
        "parameter_count": model.parameter_count(),
        "config": config.to_dict(),
        "target_training_tokens": int(training["target_training_tokens"]),
        "processed_tokens": processed_tokens,
        "steps": total_steps,
        "batch_size": batch_size,
        "procedural_steps": procedural_steps,
        "public_steps": public_steps,
        "procedural_updates": procedural_updates,
        "schedule_sha256": _hash_indices(schedule),
        "schedule_unique_updates": int(len(torch.unique(schedule))),
        "schedule_accounting": accounting,
        "mean_procedural_training_loss": procedural_loss_sum / procedural_steps,
        "mean_public_training_loss": public_loss_sum / public_steps,
        "wall_seconds": elapsed,
        "training_tokens_per_second": processed_tokens / elapsed,
        "curriculum_sha256": sha256_file(curriculum_path),
        "records_sha256": sha256_file(records_path),
        "public_manifest_sha256": sha256_file(Path(public_data) / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        **evidence_hashes(
            experiment_path=experiment_path,
            finalist_path=finalist_path,
            preflight_path=preflight_path,
        ),
        "inference_checkpoint_sha256": sha256_file(export_path),
        "cash_compute_cost_usd": 0.0,
        "determinism": {
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "adamw_foreach": False,
            "adamw_fused": False,
        },
    }
    Path(run_path).parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(run_path).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen from-scratch ~5M RoPE Genesis candidate.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--finalist", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_scale_5m(
        experiment_path=args.experiment,
        finalist_path=args.finalist,
        preflight_path=args.preflight,
        curriculum_path=args.curriculum,
        records_path=args.records,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        checkpoint_path=args.checkpoint,
        export_path=args.export,
        run_path=args.run,
        device=args.device,
    )


if __name__ == "__main__":
    main()
