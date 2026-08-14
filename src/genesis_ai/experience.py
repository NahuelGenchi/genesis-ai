from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from .checkpoint import load_model, tokenizer_from_payload
from .ingest import sha256_file
from .verifiers import verify_task

EXPERIENCE_FORMAT_VERSION = "1.0"
POLICY_VERSION = "verified-experience-v1"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not isinstance(task, dict):
                raise ValueError(f"task line {line_number} must be an object")
            task_id = task.get("id")
            prompt = task.get("prompt")
            if not isinstance(task_id, str) or not task_id or not isinstance(prompt, str) or not prompt:
                raise ValueError(f"task line {line_number} missing id/prompt")
            if task_id in seen:
                raise ValueError(f"duplicate task id: {task_id}")
            seen.add(task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError("task file is empty")
    return tasks


class CheckpointProducer:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int | None = 1,
        seed: int = 9107,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.seed = seed
        self.model, payload = load_model(self.checkpoint, device)
        self.tokenizer = tokenizer_from_payload(payload)
        self.model.eval()
        self.metadata = {
            "kind": "genesis_checkpoint",
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "checkpoint_step": int(payload.get("step", 0)),
            "generation": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "base_seed": seed,
                "device": device,
            },
        }

    def __call__(self, task: dict[str, Any], ordinal: int) -> str:
        prompt = task["prompt"] + "\nAnswer:"
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("task prompt encoded to zero tokens")
        run_seed = self.seed + ordinal
        torch.manual_seed(run_seed)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            tokens,
            self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )[0].tolist()
        completion_ids = output[len(prompt_ids) :]
        return self.tokenizer.decode(completion_ids, errors="replace").strip()


def collect_experience(
    tasks: list[dict[str, Any]],
    producer: Callable[[dict[str, Any], int], str],
    *,
    producer_metadata: dict[str, Any],
    min_score: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be within [0, 1]")
    if producer_metadata.get("kind") != "genesis_checkpoint":
        raise ValueError("experience producer must be a Genesis checkpoint")

    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, task in enumerate(tasks):
        task_id = task.get("id")
        prompt = task.get("prompt")
        if not isinstance(task_id, str) or not task_id or not isinstance(prompt, str) or not prompt:
            raise ValueError("task missing id/prompt")
        if task_id in seen:
            raise ValueError(f"duplicate task id: {task_id}")
        seen.add(task_id)

        answer = producer(task, ordinal)
        if not isinstance(answer, str):
            raise TypeError("producer answer must be text")
        verification = verify_task(task, answer)
        accepted_by_policy = verification.score >= min_score
        verifier = task.get("verifier", {})
        audit_record = {
            "format_version": EXPERIENCE_FORMAT_VERSION,
            "policy": {"version": POLICY_VERSION, "min_score": min_score},
            "task": {
                "id": task_id,
                "domain": task.get("domain"),
                "difficulty": task.get("difficulty"),
                "generator": task.get("generator"),
                "prompt_sha256": _sha256_text(prompt),
                "provenance": task.get("provenance"),
            },
            "producer": producer_metadata,
            "response": answer,
            "response_sha256": _sha256_text(answer),
            "verification": verification.to_dict(),
            "verifier": {
                "kind": verifier.get("kind") if isinstance(verifier, dict) else None,
                "version": verifier.get("version") if isinstance(verifier, dict) else None,
            },
            "accepted": accepted_by_policy,
        }
        experience_id = "exp-" + _sha256_text(_canonical(audit_record))[:20]
        audit_record["id"] = experience_id
        audit.append(audit_record)

        if accepted_by_policy:
            # Candidate training examples intentionally contain no hidden oracle/
            # verifier payload. The response is exactly what Genesis produced.
            accepted.append(
                {
                    "format_version": EXPERIENCE_FORMAT_VERSION,
                    "id": experience_id,
                    "task_id": task_id,
                    "prompt": prompt,
                    "response": answer,
                    "quality_score": verification.score,
                    "provenance": {
                        "policy": POLICY_VERSION,
                        "task_generator": task.get("generator"),
                        "producer": producer_metadata,
                        "verifier_kind": audit_record["verifier"]["kind"],
                        "verifier_version": audit_record["verifier"]["version"],
                        "audit_sha256": _sha256_text(_canonical(audit_record)),
                    },
                }
            )
    return accepted, audit


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical(record) + "\n")


def write_experience_bundle(
    output_dir: Path,
    accepted: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    *,
    producer_metadata: dict[str, Any],
    min_score: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted.jsonl"
    audit_path = output_dir / "audit.jsonl"
    _write_jsonl(accepted_path, accepted)
    _write_jsonl(audit_path, audit)
    rejected = sum(not bool(record["accepted"]) for record in audit)
    manifest = {
        "format_version": EXPERIENCE_FORMAT_VERSION,
        "policy": {"version": POLICY_VERSION, "min_score": min_score},
        "producer": producer_metadata,
        "attempted": len(audit),
        "accepted": len(accepted),
        "rejected": rejected,
        "acceptance_rate": len(accepted) / len(audit) if audit else 0.0,
        "files": {
            "accepted.jsonl": sha256_file(accepted_path),
            "audit.jsonl": sha256_file(audit_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect auditable verified experience produced only by a Genesis checkpoint.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    producer = CheckpointProducer(
        args.checkpoint,
        device=args.device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=None if args.top_k <= 0 else args.top_k,
        seed=args.seed,
    )
    accepted, audit = collect_experience(
        tasks,
        producer,
        producer_metadata=producer.metadata,
        min_score=args.min_score,
    )
    manifest = write_experience_bundle(
        args.output_dir,
        accepted,
        audit,
        producer_metadata=producer.metadata,
        min_score=args.min_score,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
