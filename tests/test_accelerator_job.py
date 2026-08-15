from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.accelerator_job import (
    AcceleratorJobError,
    build_train_command,
    validate_cpu_summary,
    validate_files,
    validate_job,
)

ROOT = Path(__file__).resolve().parents[1]


def job() -> dict:
    return {
        "format_version": "1.0",
        "kind": "genesis-train-v1",
        "enabled": True,
        "id": "architecture-rmsnorm-v1",
        "issue": 175,
        "cpu_screen": {"lane": "architecture", "variant": "rmsnorm-gelu"},
        "training": {
            "data": "data/filtered",
            "tokenizer": "tokenizer.json",
            "steps": 100,
            "batch_size": 8,
            "lr": 0.001,
            "seed": 7,
            "checkpoint_every": 20,
            "model": {
                "context_length": 128,
                "d_model": 96,
                "n_heads": 4,
                "n_layers": 3,
                "d_ff": 384,
            },
        },
        "budget": {"max_steps": 100, "max_checkpoint_interval": 20},
        "resume_if_present": True,
        "promotion_authority": False,
    }


def summary() -> dict:
    return {
        "screening_only": True,
        "promotion_eligible": False,
        "guards_pass": True,
        "expensive_stage_eligible": [
            {"lane": "architecture", "variant": "rmsnorm-gelu", "source": "cpu-screen-v1"}
        ],
    }


class AcceleratorJobTests(unittest.TestCase):
    def test_valid_job_requires_exact_cpu_winner(self) -> None:
        candidate = job()
        validate_job(candidate)
        validate_cpu_summary(summary(), candidate)

    def test_non_model_lane_cannot_consume_gpu(self) -> None:
        candidate = job()
        candidate["cpu_screen"] = {"lane": "data-filtering", "variant": "min-chars-80"}
        with self.assertRaisesRegex(AcceleratorJobError, "model-side"):
            validate_job(candidate)

    def test_non_winner_is_rejected(self) -> None:
        candidate = job()
        candidate["cpu_screen"]["variant"] = "layernorm-swiglu"
        validate_job(candidate)
        with self.assertRaisesRegex(AcceleratorJobError, "did not survive"):
            validate_cpu_summary(summary(), candidate)

    def test_accelerator_has_no_promotion_authority(self) -> None:
        candidate = job()
        candidate["promotion_authority"] = True
        with self.assertRaisesRegex(AcceleratorJobError, "promotion_authority"):
            validate_job(candidate)

    def test_budget_caps_steps_and_checkpoint_interval(self) -> None:
        candidate = job()
        candidate["training"]["steps"] = 101
        with self.assertRaisesRegex(AcceleratorJobError, "max_steps"):
            validate_job(candidate)
        candidate = job()
        candidate["training"]["checkpoint_every"] = 21
        with self.assertRaisesRegex(AcceleratorJobError, "checkpoint interval"):
            validate_job(candidate)

    def test_build_command_is_fixed_genesis_train_cuda(self) -> None:
        candidate = job()
        with tempfile.TemporaryDirectory() as directory:
            command = build_train_command(candidate, output_dir=directory, device="cuda")
            self.assertIn("genesis_ai.train", command)
            self.assertIn("--device", command)
            self.assertEqual(command[command.index("--device") + 1], "cuda")
            self.assertNotIn("--resume", command)

    def test_resume_uses_existing_latest_checkpoint(self) -> None:
        candidate = job()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            command = build_train_command(candidate, output_dir=directory, device="cuda")
            self.assertIn("--resume", command)
            self.assertEqual(command[command.index("--resume") + 1], str(checkpoint))

    def test_input_files_are_checked(self) -> None:
        candidate = job()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "filtered").mkdir(parents=True)
            (root / "data" / "filtered" / "manifest.json").write_text("{}\n")
            (root / "tokenizer.json").write_text("{}\n")
            validate_files(candidate, root)

    def test_colab_notebook_is_valid_and_dry_by_default(self) -> None:
        notebook = json.loads((ROOT / "accelerators" / "colab" / "genesis_colab.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        sources = "\n".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("RUN = False", sources)
        self.assertIn("detect_accelerator('colab')", sources)

    def test_kaggle_dispatch_is_cpu_gated_and_p100_only(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "kaggle-gpu-dispatch.yml").read_text()
        self.assertIn("genesis_ai.accelerator_job validate", workflow)
        self.assertIn("NvidiaTeslaP100", workflow)
        self.assertIn("KAGGLE_USERNAME", workflow)
        self.assertIn("KAGGLE_KEY", workflow)
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("ubuntu-latest-", workflow)


if __name__ == "__main__":
    unittest.main()
