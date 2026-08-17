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
from .autonomous_curriculum import CURRICULUM_VERSION
from .checkpoint import export_inference_checkpoint, load_model, save_checkpoint, tokenizer_from_payload
from .data import TokenDataset, sample_batch
from .improvement_controller import PUBLIC_MIN_CHARS_VARIANTS
from .ingest import sha256_file
from .scale_training import BASE_LR, CPU_THREADS, GRAD_CLIP, _lr
from .terminated_training import TerminatedGenerationAlignedDataset
from .tokenizer import ByteBPETokenizer

TRAINING_POLICY_VERSION = "autonomous-continuation-v1.3"
TARGET_TOKENS_PER_STEP = 1024
EXPECTED_PARAMETERS = 1_895_808
DOMAINS = ("code", "math", "structured")
REPLAY_EXAMPLES_BY_BUDGET = {
    3_000_000: 1_024,
    2_500_000: 768,
    2_000_000: 512,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_records(path: str | Path, *, plan_sha256: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"records:{line_number} must be an object")
            if value.get("curriculum") != CURRICULUM_VERSION or value.get("plan_sha256") != plan_sha256:
                raise ValueError("record is not bound to the autonomous curriculum plan")
            if value.get("role") not in {"focus", "replay"} or value.get("domain") not in DOMAINS:
                raise ValueError("invalid autonomous record role/domain")
            records.append(value)
    if not records:
        raise ValueError("autonomous records are empty")
    return records


def _resolve_continuation_weights(raw: object, *, focus_domain: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("autonomous continuation weights must be an object")
    replay = [domain for domain in DOMAINS if domain != focus_domain]
    if set(raw) == {"focus", "each_replay_domain"}:
        weights = {
            focus_domain: float(raw["focus"]),
            replay[0]: float(raw["each_replay_domain"]),
            replay[1]: float(raw["each_replay_domain"]),
        }
    elif set(raw) == set(DOMAINS):
        weights = {domain: float(raw[domain]) for domain in DOMAINS}
    else:
        raise ValueError("autonomous continuation weights must be legacy focus/replay or exact per-domain weights")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in weights.values()):
        raise ValueError("autonomous continuation weights are outside [0,1]")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("autonomous continuation weights must sum to 1")
    if weights[focus_domain] <= 0.0:
        raise ValueError("autonomous focus domain must receive continuation weight")
    return weights


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
        raise ValueError("unsupported autonomous curriculum")
    if curriculum.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("autonomous curriculum violates zero-cash contract")
    parent_sha = sha256_file(parent_checkpoint)
    if curriculum.get("incumbent_checkpoint_sha256") != parent_sha:
        raise ValueError("autonomous curriculum is not bound to this incumbent checkpoint")
    if curriculum.get("records_file_sha256") != sha256_file(records_path):
        raise ValueError("autonomous records hash mismatch")
    if curriculum.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise ValueError("autonomous tokenizer hash mismatch")
    if curriculum.get("public_manifest_sha256") != sha256_file(public_data / "manifest.json"):
        raise ValueError("autonomous public manifest mismatch")
    if int(curriculum.get("exact_holdout_prompt_overlap_count", -1)) != 0:
        raise ValueError("autonomous curriculum overlaps target holdout")

    public_min_chars = curriculum.get("public_min_chars")
    allowed_min_chars = {0, *PUBLIC_MIN_CHARS_VARIANTS.values()}
    if not isinstance(public_min_chars, int) or isinstance(public_min_chars, bool) or public_min_chars not in allowed_min_chars:
        raise ValueError("autonomous public minimum length is outside screened allow-list")

    model, parent_payload = load_model(parent_checkpoint, "cpu")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer_from_payload(parent_payload).to_dict() != tokenizer.to_dict():
        raise ValueError("incumbent tokenizer differs from autonomous tokenizer")
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise ValueError("autonomous v1 currently requires the micro-2m architecture")
    if model.config.context_length != 128:
        raise ValueError("autonomous v1 currently requires context length 128")

    plan_sha256 = str(curriculum.get("plan_sha256"))
    records = _read_records(records_path, plan_sha256=plan_sha256)
    if len(records) != int(curriculum.get("record_count", -1)):
        raise ValueError("autonomous record count mismatch")
    focus = str(curriculum.get("focus_domain"))
    if focus not in DOMAINS:
        raise ValueError("invalid autonomous focus domain")

    target_training_tokens = int(curriculum["target_training_tokens"])
    if target_training_tokens not in REPLAY_EXAMPLES_BY_BUDGET:
        raise ValueError("autonomous v1 training budget is outside controller contract")
    expected_replay_records = int(curriculum.get("replay_examples_per_domain", REPLAY_EXAMPLES_BY_BUDGET[target_training_tokens]))
    if expected_replay_records != REPLAY_EXAMPLES_BY_BUDGET[target_training_tokens]:
        raise ValueError("autonomous replay example count is outside budget contract")
    expected_focus_records = int(curriculum.get("focus_examples", 4096))
    if expected_focus_records <= 0 or expected_focus_records > 4096:
        raise ValueError("autonomous focus example count is outside bounded research contract")
    continuation_weights = _resolve_continuation_weights(curriculum.get("continuation_update_weights"), focus_domain=focus)

    role_counts = {domain: {"focus": 0, "replay": 0} for domain in DOMAINS}
    for record in records:
        role_counts[str(record["domain"])][str(record["role"])] += 1
    if role_counts[focus]["focus"] != expected_focus_records:
        raise ValueError(f"autonomous focus domain must contain exactly {expected_focus_records} focus records")
    for domain in DOMAINS:
        if domain == focus:
            if role_counts[domain]["replay"] != 0:
                raise ValueError("focus domain may not also contain replay records")
        elif role_counts[domain]["replay"] != expected_replay_records or role_counts[domain]["focus"] != 0:
            raise ValueError(
                f"each non-focus domain must contain exactly {expected_replay_records} replay records for this budget"
            )

    domain_records = curriculum.get("domain_records")
    if not isinstance(domain_records, dict) or set(domain_records) != set(DOMAINS):
        raise ValueError("autonomous curriculum domain record summary is invalid")
    for domain in DOMAINS:
        summary = domain_records[domain]
        if not isinstance(summary, dict):
            raise ValueError(f"autonomous curriculum domain summary is invalid: {domain}")
        expected_role = "focus" if domain == focus else "replay"
        expected_examples = role_counts[domain][expected_role]
        if summary.get("role") != expected_role or int(summary.get("examples", -1)) != expected_examples:
            raise ValueError(f"autonomous curriculum record summary drifted: {domain}")

    policy = {
        "parent_checkpoint_sha256": parent_sha,
        "curriculum_lock_sha256": sha256_file(curriculum_lock),
        "records_sha256": sha256_file(records_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "target_training_tokens": target_training_tokens,
        "procedural_fraction": float(curriculum["procedural_fraction"]),
        "public_fraction": float(curriculum["public_fraction"]),
        "public_min_chars": public_min_chars,
        "focus_domain": focus,
        "focus_examples": expected_focus_records,
        "replay_examples_per_domain": expected_replay_records,
        "continuation_update_weights": continuation_weights,
        "role_counts": role_counts,
        "plan_sha256": plan_sha256,
    }
    if not math.isclose(policy["procedural_fraction"], 0.8) or not math.isclose(policy["public_fraction"], 0.2):
        raise ValueError("autonomous v1 requires exact 80/20 mix")
    return model, tokenizer, records, curriculum, policy, parent_payload


def _hash_indices(indices: torch.Tensor) -> str:
    payload = ",".join(str(int(value)) for value in indices.tolist()).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_focus_schedule(
    dataset: TerminatedGenerationAlignedDataset,
    records: list[dict[str, Any]],
    *,
    focus_domain: str,
    total_samples: int,
    seed: int,
    continuation_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if focus_domain not in DOMAINS:
        raise ValueError("invalid autonomous focus domain")
    domain_by_record = [str(record["domain"]) for record in records]
    record_counts = {domain: domain_by_record.count(domain) for domain in DOMAINS}
    if any(count <= 0 for count in record_counts.values()):
        raise ValueError("autonomous schedule requires records for every domain")

    anchors = list(dataset.anchor_indices)
    expected_anchors = {domain: record_counts[domain] * 2 for domain in DOMAINS}
    expected_anchor_total = sum(expected_anchors.values())
    if len(anchors) != expected_anchor_total:
        raise ValueError(
            f"autonomous first/terminator anchor count drifted: expected {expected_anchor_total}, got {len(anchors)}"
        )
    if total_samples < len(anchors):
        raise ValueError("autonomous budget cannot cover mandatory anchors")
    if total_samples > len(dataset):
        raise ValueError("autonomous v1 forbids duplicate target contexts")

    anchor_by_domain = {domain: 0 for domain in DOMAINS}
    for index in anchors:
        record_ordinal = int(dataset.record_ordinals[index])
        anchor_by_domain[domain_by_record[record_ordinal]] += 1
    if anchor_by_domain != expected_anchors:
        raise ValueError(f"autonomous anchor accounting drifted: {anchor_by_domain}")

    continuation_by_domain: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    for index in dataset.continuation_indices:
        record_ordinal = int(dataset.record_ordinals[index])
        continuation_by_domain[domain_by_record[record_ordinal]].append(int(index))

    remaining = total_samples - len(anchors)
    if continuation_weights is None:
        replay = [domain for domain in DOMAINS if domain != focus_domain]
        continuation_weights = {focus_domain: 0.70, replay[0]: 0.15, replay[1]: 0.15}
    else:
        continuation_weights = _resolve_continuation_weights(continuation_weights, focus_domain=focus_domain)
    quotas = {domain: int(remaining * continuation_weights[domain]) for domain in DOMAINS}
    quotas[focus_domain] += remaining - sum(quotas.values())

    rng = random.Random(seed)
    selected_extra: list[int] = []
    available = {domain: len(values) for domain, values in continuation_by_domain.items()}
    for domain in DOMAINS:
        pool = continuation_by_domain[domain]
        rng.shuffle(pool)
        if len(pool) < quotas[domain]:
            raise ValueError(f"insufficient unique {domain} continuation contexts: need {quotas[domain]}, have {len(pool)}")
        selected_extra.extend(pool[: quotas[domain]])

    selected = torch.tensor(anchors + selected_extra, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    selected = selected[torch.randperm(len(selected), generator=generator)]
    if len(torch.unique(selected)) != len(selected):
        raise AssertionError("autonomous schedule contains duplicate target contexts")
    return selected, {
        "record_counts_by_domain": record_counts,
        "anchor_updates_by_domain": anchor_by_domain,
        "continuation_available_by_domain": available,
        "continuation_weights": continuation_weights,
        "continuation_updates_by_domain": quotas,
        "total_updates_by_domain": {domain: anchor_by_domain[domain] + quotas[domain] for domain in DOMAINS},
        "total_anchor_updates": len(anchors),
        "total_continuation_updates": remaining,
        "total_updates": total_samples,
    }


def _seed_from_plan(plan_sha256: str) -> int:
    return int(plan_sha256[:16], 16) % 2_000_000_000


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
) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("autonomous continuation v1 is frozen to deterministic CPU")
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
            kind = "procedural-autonomous" if procedural_step else "public"
            print(f"step={step}/{total_steps} kind={kind} loss={observed:.6f} lr={learning_rate:.8f}")
    if schedule_cursor != len(schedule):
        raise AssertionError("autonomous schedule was not consumed exactly")

    parent_step = int(parent_payload.get("step", 0))
    final_step = parent_step + total_steps
    metadata = {
        "training_policy": TRAINING_POLICY_VERSION,
        "plan_sha256": policy["plan_sha256"],
        "parent_checkpoint_sha256": policy["parent_checkpoint_sha256"],
        "curriculum_lock_sha256": policy["curriculum_lock_sha256"],
        "public_min_chars": policy["public_min_chars"],
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
        "plan_sha256": policy["plan_sha256"],
        "seed": seed,
        "focus_domain": policy["focus_domain"],
        "focus_examples": policy["focus_examples"],
        "replay_examples_per_domain": policy["replay_examples_per_domain"],
        "continuation_update_weights": policy["continuation_update_weights"],
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
        "public_min_chars": policy["public_min_chars"],
        "public_document_count": public_dataset.document_count,
        "public_token_count": public_dataset.token_count,
        "schedule_sha256": _hash_indices(schedule),
        "schedule_unique_updates": int(len(torch.unique(schedule))),
        "schedule_accounting": accounting,
        "mean_procedural_training_loss": procedural_loss_sum / procedural_steps,
        "mean_public_training_loss": public_loss_sum / public_steps,
        "curriculum_lock_sha256": policy["curriculum_lock_sha256"],
        "records_sha256": policy["records_sha256"],
        "public_manifest_sha256": policy["public_manifest_sha256"],
        "tokenizer_sha256": policy["tokenizer_sha256"],
        "inference_checkpoint_sha256": sha256_file(export_path),
        "cash_compute_cost_usd": 0.0,
        "determinism": {"device": "cpu", "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True, "adamw_foreach": False, "adamw_fused": False},
    }
    Path(run_path).parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(run_path).write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one plan-bound autonomous Genesis continuation candidate.")
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
