from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

FORMAT_VERSION = "1.0"
JOB_KIND = "genesis-train-v1"
SUPPORTED_PLATFORMS = {"kaggle", "colab", "local"}
MODEL_LANES = {"architecture", "optimizer", "tiny-model"}


class AcceleratorJobError(ValueError):
    pass


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AcceleratorJobError(f"cannot read JSON file {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AcceleratorJobError(f"invalid JSON in {source}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise AcceleratorJobError(f"{source} must contain a JSON object")
    return value


def validate_job(job: dict[str, Any]) -> None:
    if job.get("format_version") != FORMAT_VERSION:
        raise AcceleratorJobError("unsupported accelerator job format_version")
    if job.get("kind") != JOB_KIND:
        raise AcceleratorJobError(f"job kind must be {JOB_KIND}")
    if job.get("enabled") is not True:
        raise AcceleratorJobError("accelerator job must be explicitly enabled")
    if not isinstance(job.get("id"), str) or not job["id"].strip():
        raise AcceleratorJobError("job id is required")
    if not isinstance(job.get("issue"), int) or isinstance(job.get("issue"), bool) or job["issue"] <= 0:
        raise AcceleratorJobError("job issue must be a positive integer")

    cpu_screen = job.get("cpu_screen")
    if not isinstance(cpu_screen, dict):
        raise AcceleratorJobError("cpu_screen is required")
    lane = cpu_screen.get("lane")
    variant = cpu_screen.get("variant")
    if lane not in MODEL_LANES:
        raise AcceleratorJobError("GPU jobs are restricted to model-side CPU-screened lanes")
    if not isinstance(variant, str) or not variant:
        raise AcceleratorJobError("cpu_screen.variant is required")

    training = job.get("training")
    if not isinstance(training, dict):
        raise AcceleratorJobError("training configuration is required")
    required_paths = ("data", "tokenizer")
    for key in required_paths:
        if not isinstance(training.get(key), str) or not training[key]:
            raise AcceleratorJobError(f"training.{key} is required")

    positive_ints = ("steps", "batch_size", "checkpoint_every")
    for key in positive_ints:
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AcceleratorJobError(f"training.{key} must be a positive integer")
    lr = training.get("lr")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
        raise AcceleratorJobError("training.lr must be positive")
    seed = training.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise AcceleratorJobError("training.seed must be a non-negative integer")

    model = training.get("model")
    if not isinstance(model, dict):
        raise AcceleratorJobError("training.model is required")
    for key in ("context_length", "d_model", "n_heads", "n_layers", "d_ff"):
        value = model.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise AcceleratorJobError(f"training.model.{key} must be a positive integer")
    if model["d_model"] % model["n_heads"] != 0:
        raise AcceleratorJobError("training.model.d_model must be divisible by n_heads")

    budget = job.get("budget")
    if not isinstance(budget, dict):
        raise AcceleratorJobError("budget is required")
    max_steps = budget.get("max_steps")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise AcceleratorJobError("budget.max_steps must be a positive integer")
    if training["steps"] > max_steps:
        raise AcceleratorJobError("training.steps exceeds budget.max_steps")
    max_checkpoint_interval = budget.get("max_checkpoint_interval", 500)
    if not isinstance(max_checkpoint_interval, int) or max_checkpoint_interval <= 0:
        raise AcceleratorJobError("budget.max_checkpoint_interval must be positive")
    if training["checkpoint_every"] > max_checkpoint_interval:
        raise AcceleratorJobError("checkpoint interval exceeds budget maximum")

    if job.get("promotion_authority") is not False:
        raise AcceleratorJobError("accelerator jobs must set promotion_authority=false")


def validate_cpu_summary(summary: dict[str, Any], job: dict[str, Any]) -> None:
    if summary.get("screening_only") is not True:
        raise AcceleratorJobError("CPU summary must be screening_only")
    if summary.get("promotion_eligible") is not False:
        raise AcceleratorJobError("CPU summary must not have promotion authority")
    if summary.get("guards_pass") is not True:
        raise AcceleratorJobError("CPU summary guards did not pass")

    cpu_screen = job["cpu_screen"]
    wanted = (cpu_screen["lane"], cpu_screen["variant"])
    eligible = summary.get("expensive_stage_eligible")
    if not isinstance(eligible, list):
        raise AcceleratorJobError("CPU summary has no expensive_stage_eligible list")
    pairs = {
        (entry.get("lane"), entry.get("variant"))
        for entry in eligible
        if isinstance(entry, dict)
    }
    if wanted not in pairs:
        raise AcceleratorJobError(
            f"{wanted[0]}/{wanted[1]} did not survive the referenced CPU screen"
        )


def detect_accelerator(platform: str) -> dict[str, Any]:
    if platform not in SUPPORTED_PLATFORMS:
        raise AcceleratorJobError(f"unsupported platform: {platform}")

    try:
        import torch
    except ImportError as exc:
        raise AcceleratorJobError("PyTorch is required to detect the accelerator") from exc

    tpu_detected = any(
        os.environ.get(name)
        for name in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_WORKER_ID")
    )
    if torch.cuda.is_available():
        return {
            "platform": platform,
            "device": "cuda",
            "accelerator": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "tpu_detected": tpu_detected,
        }
    if tpu_detected:
        return {
            "platform": platform,
            "device": "tpu",
            "accelerator": "TPU detected; Genesis PyTorch/XLA training is not enabled yet",
            "cuda_device_count": 0,
            "tpu_detected": True,
        }
    return {
        "platform": platform,
        "device": "cpu",
        "accelerator": "CPU",
        "cuda_device_count": 0,
        "tpu_detected": False,
    }


def build_train_command(
    job: dict[str, Any],
    *,
    output_dir: str | Path,
    device: str,
) -> list[str]:
    if device != "cuda":
        raise AcceleratorJobError("accelerator training requires an available CUDA GPU")
    training = job["training"]
    model = training["model"]
    output = Path(output_dir)
    checkpoint = output / "latest.pt"
    run_metadata = output / "run.json"
    export = output / "inference.pt"

    command = [
        sys.executable,
        "-m",
        "genesis_ai.train",
        "--data",
        training["data"],
        "--tokenizer",
        training["tokenizer"],
        "--steps",
        str(training["steps"]),
        "--batch-size",
        str(training["batch_size"]),
        "--lr",
        str(training["lr"]),
        "--seed",
        str(training["seed"]),
        "--device",
        "cuda",
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-every",
        str(training["checkpoint_every"]),
        "--run-metadata",
        str(run_metadata),
        "--export",
        str(export),
        "--context-length",
        str(model["context_length"]),
        "--d-model",
        str(model["d_model"]),
        "--n-heads",
        str(model["n_heads"]),
        "--n-layers",
        str(model["n_layers"]),
        "--d-ff",
        str(model["d_ff"]),
    ]
    if job.get("resume_if_present", True) and checkpoint.is_file():
        command.extend(("--resume", str(checkpoint)))
    return command


def validate_files(job: dict[str, Any], root: str | Path = ".") -> None:
    base = Path(root)
    training = job["training"]
    data = Path(training["data"])
    tokenizer = Path(training["tokenizer"])
    if not data.is_absolute():
        data = base / data
    if not tokenizer.is_absolute():
        tokenizer = base / tokenizer
    if not (data / "manifest.json").is_file():
        raise AcceleratorJobError(f"training data manifest not found: {data / 'manifest.json'}")
    if not tokenizer.is_file():
        raise AcceleratorJobError(f"tokenizer not found: {tokenizer}")


def run_job(
    job_path: str | Path,
    summary_path: str | Path,
    *,
    platform: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    job = _load_json(job_path)
    summary = _load_json(summary_path)
    validate_job(job)
    validate_cpu_summary(summary, job)
    validate_files(job)
    detected = detect_accelerator(platform)
    if detected["device"] != "cuda":
        raise AcceleratorJobError(
            f"CUDA GPU required; detected {detected['device']}. "
            "On Colab choose a GPU runtime; TPU execution is not yet supported."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = build_train_command(job, output_dir=output, device="cuda")
    subprocess.run(command, check=True)

    run_metadata_path = output / "run.json"
    run_metadata = _load_json(run_metadata_path)
    record = {
        "format_version": FORMAT_VERSION,
        "job_id": job["id"],
        "issue": job["issue"],
        "platform": platform,
        "accelerator": detected,
        "cpu_screen": job["cpu_screen"],
        "training": run_metadata,
        "checkpoint": str(output / "latest.pt"),
        "inference_export": str(output / "inference.pt"),
        "screening_only": False,
        "promotion_eligible": False,
        "promotion_authority": False,
        "requires_frozen_evaluation_before_promotion": True,
    }
    (output / "accelerator-record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or run a CPU-screen-gated Genesis accelerator job.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--job", required=True)
    validate_parser.add_argument("--cpu-summary", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--job", required=True)
    run_parser.add_argument("--cpu-summary", required=True)
    run_parser.add_argument("--platform", choices=sorted(SUPPORTED_PLATFORMS), required=True)
    run_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    try:
        job = _load_json(args.job)
        summary = _load_json(args.cpu_summary)
        validate_job(job)
        validate_cpu_summary(summary, job)
        if args.command == "validate":
            print(f"eligible accelerator job: {job['id']}")
            return
        record = run_job(
            args.job,
            args.cpu_summary,
            platform=args.platform,
            output_dir=args.output_dir,
        )
    except (AcceleratorJobError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
