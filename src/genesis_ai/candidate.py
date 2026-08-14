from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .checkpoint import export_inference_checkpoint, load_model, restore_training_state, save_checkpoint, tokenizer_from_payload
from .data import sample_batch
from .experience import EXPERIENCE_FORMAT_VERSION, POLICY_VERSION
from .ingest import sha256_file
from .model import GenesisLM
from .tokenizer import ByteBPETokenizer

CANDIDATE_POLICY_VERSION = "candidate-training-v1"
IGNORE_INDEX = -100


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            records.append(value)
    return records


def validate_experience_bundle(
    bundle_dir: str | Path,
    parent_checkpoint: str | Path,
    *,
    min_accepted: int = 1,
    required_min_score: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if min_accepted <= 0:
        raise ValueError("min_accepted must be positive")
    if not 0.0 <= required_min_score <= 1.0:
        raise ValueError("required_min_score must be within [0, 1]")
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    accepted_path = bundle_dir / "accepted.jsonl"
    audit_path = bundle_dir / "audit.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != EXPERIENCE_FORMAT_VERSION:
        raise ValueError("unsupported experience manifest")

    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION:
        raise ValueError("unsupported experience policy")
    min_score = policy.get("min_score")
    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool) or float(min_score) < required_min_score:
        raise ValueError("experience score policy is below candidate threshold")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("experience manifest missing file hashes")
    for name, path in (("accepted.jsonl", accepted_path), ("audit.jsonl", audit_path)):
        expected_hash = files.get(name)
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise ValueError(f"experience file hash mismatch: {name}")

    parent_checkpoint = Path(parent_checkpoint)
    parent_hash = sha256_file(parent_checkpoint)
    producer = manifest.get("producer")
    if (
        not isinstance(producer, dict)
        or producer.get("kind") != "genesis_checkpoint"
        or producer.get("checkpoint_sha256") != parent_hash
    ):
        raise ValueError("experience producer does not match parent checkpoint")

    accepted = _read_jsonl(accepted_path)
    audit = _read_jsonl(audit_path)
    accepted_count = manifest.get("accepted")
    attempted = manifest.get("attempted")
    rejected = manifest.get("rejected")
    if accepted_count != len(accepted) or attempted != len(audit):
        raise ValueError("experience manifest counts do not match files")
    audit_accepted = [record for record in audit if record.get("accepted") is True]
    audit_rejected = [record for record in audit if record.get("accepted") is False]
    if len(audit_accepted) != len(accepted) or rejected != len(audit_rejected):
        raise ValueError("experience audit acceptance counts do not match manifest")
    if len(accepted) < min_accepted:
        raise ValueError(f"insufficient accepted experience: {len(accepted)} < {min_accepted}")

    accepted_ids = {record.get("id") for record in accepted}
    audit_ids = {record.get("id") for record in audit_accepted}
    if None in accepted_ids or len(accepted_ids) != len(accepted) or accepted_ids != audit_ids:
        raise ValueError("accepted experience IDs do not match audit ledger")

    for record in accepted:
        if record.get("format_version") != EXPERIENCE_FORMAT_VERSION:
            raise ValueError("unsupported accepted experience record")
        prompt = record.get("prompt")
        response = record.get("response")
        score = record.get("quality_score")
        provenance = record.get("provenance")
        if not isinstance(prompt, str) or not prompt or not isinstance(response, str) or not response:
            raise ValueError("accepted experience requires non-empty prompt/response")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or float(score) < required_min_score:
            raise ValueError("accepted experience score is below candidate threshold")
        if not isinstance(provenance, dict) or provenance.get("policy") != POLICY_VERSION:
            raise ValueError("accepted experience provenance is invalid")
        record_producer = provenance.get("producer")
        if not isinstance(record_producer, dict) or record_producer.get("checkpoint_sha256") != parent_hash:
            raise ValueError("accepted record producer does not match parent checkpoint")

    return manifest, accepted


class ExperienceDataset(Dataset):
    """Fixed-size windows with loss only on verified response tokens."""

    def __init__(self, records: list[dict[str, Any]], tokenizer: ByteBPETokenizer, context_length: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        items: list[tuple[torch.Tensor, torch.Tensor]] = []
        supervised_tokens = 0
        for record in records:
            prompt = record["prompt"] + "\nAnswer:"
            response = record["response"]
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)
            if not response_ids:
                raise ValueError(f"experience {record.get('id')} has zero response tokens")
            sequence = prompt_ids + response_ids
            response_start = len(prompt_ids)
            for segment_start in range(response_start, len(sequence), context_length):
                segment_end = min(segment_start + context_length - 1, len(sequence) - 1)
                window_start = max(0, segment_end - context_length)
                chunk = sequence[window_start : segment_end + 1]
                pad = context_length + 1 - len(chunk)
                packed = [0] * pad + chunk
                x = torch.tensor(packed[:-1], dtype=torch.long)
                y_values = packed[1:]
                for position in range(context_length):
                    packed_target_index = position + 1
                    if packed_target_index < pad:
                        y_values[position] = IGNORE_INDEX
                        continue
                    global_target_index = window_start + packed_target_index - pad
                    if global_target_index < segment_start or global_target_index > segment_end:
                        y_values[position] = IGNORE_INDEX
                    else:
                        supervised_tokens += 1
                y = torch.tensor(y_values, dtype=torch.long)
                if bool((y != IGNORE_INDEX).any()):
                    items.append((x, y))
        if not items or supervised_tokens <= 0:
            raise ValueError("experience dataset has no supervised response tokens")
        self.items = tuple(items)
        self.supervised_tokens = supervised_tokens
        self.experience_count = len(records)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


def _experience_loss(model: GenesisLM, dataset: ExperienceDataset, device: str) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in dataset.items:
            _, loss = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_candidate(
    *,
    parent_checkpoint: str | Path,
    experience_dir: str | Path,
    checkpoint: str | Path,
    steps: int,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    seed: int = 12001,
    min_accepted: int = 1,
    required_min_score: float = 1.0,
    device: str = "cpu",
    resume: str | Path | None = None,
    checkpoint_every: int = 25,
    export: str | Path | None = None,
) -> dict[str, Any]:
    if steps <= 0 or batch_size <= 0 or checkpoint_every <= 0:
        raise ValueError("steps, batch_size, and checkpoint_every must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    parent_checkpoint = Path(parent_checkpoint)
    parent_hash_before = sha256_file(parent_checkpoint)
    manifest, accepted = validate_experience_bundle(
        experience_dir,
        parent_checkpoint,
        min_accepted=min_accepted,
        required_min_score=required_min_score,
    )
    manifest_hash = sha256_file(Path(experience_dir) / "manifest.json")

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)

    parent_model, parent_payload = load_model(parent_checkpoint, device)
    tokenizer = tokenizer_from_payload(parent_payload)
    parent_step = int(parent_payload.get("step", 0))
    dataset = ExperienceDataset(accepted, tokenizer, parent_model.config.context_length)
    completed_steps = 0

    if resume is not None:
        model, payload = load_model(resume, device)
        resume_tokenizer = tokenizer_from_payload(payload)
        if resume_tokenizer.merges != tokenizer.merges:
            raise ValueError("resume tokenizer does not match parent tokenizer")
        metadata = payload.get("metadata")
        self_improvement = metadata.get("self_improvement") if isinstance(metadata, dict) else None
        if (
            not isinstance(self_improvement, dict)
            or self_improvement.get("parent_checkpoint_sha256") != parent_hash_before
            or self_improvement.get("experience_manifest_sha256") != manifest_hash
            or self_improvement.get("policy") != CANDIDATE_POLICY_VERSION
        ):
            raise ValueError("resume candidate provenance does not match parent/experience")
        completed_steps = int(self_improvement.get("candidate_steps_completed", 0))
        if steps <= completed_steps:
            raise ValueError("steps must exceed resumed candidate steps")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        restore_training_state(payload, optimizer, batch_generator)
        initial_loss = float(self_improvement["experience_loss_before"])
    else:
        model = parent_model
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        initial_loss = _experience_loss(model, dataset, device)

    def metadata_for(candidate_steps_completed: int, experience_loss_after: float | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "self_improvement": {
                "policy": CANDIDATE_POLICY_VERSION,
                "parent_checkpoint_sha256": parent_hash_before,
                "parent_checkpoint_step": parent_step,
                "experience_manifest_sha256": manifest_hash,
                "experience_files": dict(manifest["files"]),
                "experience_policy": manifest["policy"],
                "experience_accepted": len(accepted),
                "candidate_steps_completed": candidate_steps_completed,
                "target_candidate_steps": steps,
                "seed": seed,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "experience_loss_before": initial_loss,
                "supervised_response_tokens": dataset.supervised_tokens,
            }
        }
        if experience_loss_after is not None:
            result["self_improvement"]["experience_loss_after"] = experience_loss_after
            result["self_improvement"]["experience_loss_decreased"] = experience_loss_after < initial_loss
        return result

    model.train()
    checkpoint = Path(checkpoint)
    for candidate_step in range(completed_steps + 1, steps + 1):
        x, y = sample_batch(dataset, batch_size, batch_generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if candidate_step % checkpoint_every == 0 and candidate_step != steps:
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=parent_step + candidate_step,
                metadata=metadata_for(candidate_step),
                tokenizer=tokenizer,
                batch_generator=batch_generator,
            )

    final_loss = _experience_loss(model, dataset, device)
    metadata = metadata_for(steps, final_loss)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=parent_step + steps,
        metadata=metadata,
        tokenizer=tokenizer,
        batch_generator=batch_generator,
    )
    if export is not None:
        export_inference_checkpoint(checkpoint, export)
    if sha256_file(parent_checkpoint) != parent_hash_before:
        raise RuntimeError("parent checkpoint changed during candidate training")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an auditable N+1 candidate from verified Genesis experience.")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=12001)
    parser.add_argument("--min-accepted", type=int, default=1)
    parser.add_argument("--required-min-score", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    metadata = train_candidate(
        parent_checkpoint=args.parent,
        experience_dir=args.experience,
        checkpoint=args.checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        min_accepted=args.min_accepted,
        required_min_score=args.required_min_score,
        device=args.device,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        export=args.export,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
