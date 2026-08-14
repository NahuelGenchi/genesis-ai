from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from .aligned_training import batch_from_indices
from .checkpoint import export_inference_checkpoint, load_model, save_checkpoint, tokenizer_from_payload
from .data import TokenDataset, sample_batch
from .ingest import sha256_file
from .multidomain_curriculum_v2 import CURRICULUM_VERSION, EXAMPLES_PER_DOMAIN, TARGET_TRAINING_TOKENS
from .scale_training import BASE_LR, CPU_THREADS, GRAD_CLIP, _lr
from .terminated_training import TerminatedGenerationAlignedDataset
from .tokenizer import ByteBPETokenizer

TRAINING_POLICY_VERSION = "m6-multidomain-continuation-v2"
SEED = 101_001
TARGET_TOKENS_PER_STEP = 1024
EXPECTED_PARAMETERS = 1_895_808
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
            if value.get("curriculum") != CURRICULUM_VERSION:
                raise ValueError("unexpected v2 curriculum version")
            if value.get("domain") not in DOMAINS:
                raise ValueError("invalid v2 record domain")
            provenance = value.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("kind") != "procedural_oracle":
                raise ValueError("v2 provenance must be procedural_oracle")
            records.append(value)
    if not records:
        raise ValueError("v2 records are empty")
    return records


def validate_inputs(
    *,
    parent_checkpoint: str | Path,
    curriculum_lock: str | Path,
    records_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
) -> tuple[Any, ByteBPETokenizer, list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent_checkpoint = Path(parent_checkpoint)
    curriculum_lock = Path(curriculum_lock)
    records_path = Path(records_path)
    public_data = Path(public_data)
    tokenizer_path = Path(tokenizer_path)
    curriculum = _load_json(curriculum_lock)
    if curriculum.get("curriculum_version") != CURRICULUM_VERSION:
        raise ValueError("unsupported v2 curriculum lock")
    if int(curriculum.get("target_training_tokens", -1)) != TARGET_TRAINING_TOKENS:
        raise ValueError("v2 training budget drifted")
    if curriculum.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("curriculum violates zero-cash contract")
    if curriculum.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("curriculum tokenizer hash mismatch")
    if curriculum.get("public_text", {}).get("manifest_sha256") != sha256_file(public_data / "manifest.json"):
        raise ValueError("public corpus manifest mismatch")
    procedural = curriculum.get("procedural")
    if not isinstance(procedural, dict) or procedural.get("records_file_sha256") != sha256_file(records_path):
        raise ValueError("v2 records do not match curriculum lock")
    if int(curriculum.get("evaluation", {}).get("exact_prompt_overlap_count", -1)) != 0:
        raise ValueError("v2 curriculum overlaps frozen holdout")

    model, parent_payload = load_model(parent_checkpoint, "cpu")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    parent_tokenizer = tokenizer_from_payload(parent_payload)
    if parent_tokenizer.to_dict() != tokenizer.to_dict():
        raise ValueError("parent checkpoint tokenizer differs from project tokenizer")
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise ValueError("parent is not the promoted micro-2m architecture")
    if model.config.context_length != int(curriculum.get("context_length", -1)):
        raise ValueError("parent context length differs from v2 curriculum")

    records = _read_records(records_path)
    expected = EXAMPLES_PER_DOMAIN * len(DOMAINS)
    if len(records) != expected or int(procedural.get("examples", -1)) != expected:
        raise ValueError("v2 record count mismatch")
    counts = {domain: 0 for domain in DOMAINS}
    for record in records:
        counts[str(record["domain"])] += 1
    if counts != {domain: EXAMPLES_PER_DOMAIN for domain in DOMAINS}:
        raise ValueError("v2 records are not exactly 4096/domain")

    policy = {
        "target_training_tokens": int(curriculum["target_training_tokens"]),
        "procedural_fraction": float(curriculum["procedural_fraction"]),
        "public_fraction": float(curriculum["public_fraction"]),
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "curriculum_lock_sha256": sha256_file(curriculum_lock),
        "records_sha256": sha256_file(records_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "domain_examples": counts,
    }
    if not math.isclose(policy["procedural_fraction"], 0.8) or not math.isclose(policy["public_fraction"], 0.2):
        raise ValueError("v2 requires exact 80/20 procedural/public mix")
    return model, tokenizer, records, curriculum, policy, parent_payload


def _hash_indices(indices: torch.Tensor) -> str:
    payload = ",".join(str(int(value)) for value in indices.tolist()).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_balanced_schedule(
    dataset: TerminatedGenerationAlignedDataset,
    records: list[dict[str, Any]],
    *,
    total_samples: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    anchors = list(dataset.anchor_indices)
    expected_anchors = EXAMPLES_PER_DOMAIN * len(DOMAINS) * 2
    if len(anchors) != expected_anchors:
        raise ValueError(f"expected {expected_anchors} first/terminator anchors, got {len(anchors)}")
    if total_samples < len(anchors):
        raise ValueError("budget cannot cover every v2 anchor")
    if total_samples > len(dataset):
        raise ValueError("v2 forbids duplicate target contexts")

    domain_by_record = [str(record["domain"]) for record in records]
    anchor_by_domain = {domain: 0 for domain in DOMAINS}
    for index in anchors:
        record_ordinal = int(dataset.record_ordinals[index])
        anchor_by_domain[domain_by_record[record_ordinal]] += 1
    expected_per_domain_anchors = EXAMPLES_PER_DOMAIN * 2
    if anchor_by_domain != {domain: expected_per_domain_anchors for domain in DOMAINS}:
        raise ValueError(f"v2 anchor imbalance: {anchor_by_domain}")

    continuation_by_domain: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    for index in dataset.continuation_indices:
        record_ordinal = int(dataset.record_ordinals[index])
        continuation_by_domain[domain_by_record[record_ordinal]].append(int(index))

    remaining = total_samples - len(anchors)
    base = remaining // len(DOMAINS)
    remainder = remaining % len(DOMAINS)
    # Give any indivisible extra replay to incumbent code first, then math.
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
                f"v2 requires {quotas[domain]} unique {domain} continuation contexts; only {len(pool)} available"
            )
        selected_extra.extend(pool[: quotas[domain]])

    selected = torch.tensor(anchors + selected_extra, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    selected = selected[torch.randperm(len(selected), generator=generator)]
    if len(torch.unique(selected)) != len(selected):
        raise AssertionError("v2 schedule contains duplicate target contexts")

    totals = {domain: anchor_by_domain[domain] + quotas[domain] for domain in DOMAINS}
    return selected, {
        "anchor_updates_by_domain": anchor_by_domain,
        "continuation_available_by_domain": available,
        "continuation_updates_by_domain": quotas,
        "total_updates_by_domain": totals,
        "total_anchor_updates": len(anchors),
        "total_continuation_updates": remaining,
        "total_updates": total_samples,
    }


def _loss_probe(model, dataset: TerminatedGenerationAlignedDataset, indices: torch.Tensor, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for index in indices.tolist():
            x, y = dataset[int(index)]
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
            assert loss is not None
            total += float(loss.detach().cpu())
    return total / len(indices)


def train_continuation(
    *,
    parent_checkpoint: str | Path,
    curriculum_lock: str | Path,
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
        raise ValueError("v2 is frozen to deterministic CPU")
    if seed != SEED:
        raise ValueError(f"v2 seed is frozen to {SEED}")
    torch.set_num_threads(CPU_THREADS)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer, records, curriculum, policy, parent_payload = validate_inputs(
        parent_checkpoint=parent_checkpoint,
        curriculum_lock=curriculum_lock,
        records_path=records_path,
        public_data=public_data,
        tokenizer_path=tokenizer_path,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, foreach=False, fused=False)
    dataset = TerminatedGenerationAlignedDataset(records, tokenizer, model.config.context_length)
    public_dataset = TokenDataset(public_data, tokenizer, model.config.context_length, split="train")

    batch_size = TARGET_TOKENS_PER_STEP // model.config.context_length
    raw_steps = math.ceil(policy["target_training_tokens"] / TARGET_TOKENS_PER_STEP)
    total_steps = math.ceil(raw_steps / 5) * 5
    procedural_steps = total_steps * 4 // 5
    public_steps = total_steps // 5
    procedural_updates = procedural_steps * batch_size
    schedule, schedule_accounting = build_balanced_schedule(
        dataset, records, total_samples=procedural_updates, seed=seed + 10
    )
    expected_updates = {"code": 12502, "math": 12501, "structured": 12501}
    if schedule_accounting["total_updates_by_domain"] != expected_updates:
        raise ValueError(f"v2 supervision accounting drifted: {schedule_accounting}")

    public_generator = torch.Generator(device="cpu").manual_seed(seed + 20)
    probe_generator = torch.Generator(device="cpu").manual_seed(seed + 30)
    probe_indices = torch.randperm(len(dataset), generator=probe_generator)[: min(384, len(dataset))]
    probe_before = _loss_probe(model, dataset, probe_indices, device)

    schedule_cursor = 0
    procedural_loss_sum = 0.0
    public_loss_sum = 0.0
    model.train()
    for step in range(1, total_steps + 1):
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
        if step == 1 or step % 250 == 0 or step == total_steps:
            kind = "procedural-multidomain-v2" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")
    if schedule_cursor != len(schedule):
        raise AssertionError("v2 schedule was not consumed exactly")
    probe_after = _loss_probe(model, dataset, probe_indices, device)

    parent_step = int(parent_payload.get("step", 0))
    final_step = parent_step + total_steps
    metadata = {
        "training_policy": TRAINING_POLICY_VERSION,
        "parent_checkpoint_sha256": policy["parent_checkpoint_sha256"],
        "curriculum_lock_sha256": policy["curriculum_lock_sha256"],
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=final_step,
        metadata=metadata,
        tokenizer=tokenizer,
        batch_generator=public_generator,
    )
    export_inference_checkpoint(checkpoint_path, export_path)

    result = {
        "format_version": "1.0",
        "training_policy": TRAINING_POLICY_VERSION,
        "seed": seed,
        "parent_checkpoint_sha256": policy["parent_checkpoint_sha256"],
        "parent_step": parent_step,
        "final_step": final_step,
        "parameter_count": model.parameter_count(),
        "target_training_tokens": policy["target_training_tokens"],
        "processed_tokens": total_steps * TARGET_TOKENS_PER_STEP,
        "steps": total_steps,
        "batch_size": batch_size,
        "procedural_steps": procedural_steps,
        "public_steps": public_steps,
        "procedural_updates": procedural_updates,
        "schedule_sha256": _hash_indices(schedule),
        "schedule_unique_updates": int(len(torch.unique(schedule))),
        "schedule_accounting": schedule_accounting,
        "domain_examples": policy["domain_examples"],
        "terminated_probe_loss_before": probe_before,
        "terminated_probe_loss_after": probe_after,
        "mean_procedural_training_loss": procedural_loss_sum / procedural_steps,
        "mean_public_training_loss": public_loss_sum / public_steps,
        "curriculum_lock_sha256": policy["curriculum_lock_sha256"],
        "records_sha256": policy["records_sha256"],
        "public_manifest_sha256": policy["public_manifest_sha256"],
        "tokenizer_sha256": policy["tokenizer_sha256"],
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
    parser = argparse.ArgumentParser(description="Continue Genesis micro-2m with full per-domain supervision v2.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--curriculum-lock", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_continuation(
        parent_checkpoint=args.parent,
        curriculum_lock=args.curriculum_lock,
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
