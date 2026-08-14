import json
import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.checkpoint import export_inference_checkpoint, save_checkpoint
from genesis_ai.config import ModelConfig
from genesis_ai.ingest import sha256_file
from genesis_ai.model import GenesisLM
from genesis_ai.promotion import decide_promotion
from genesis_ai.tokenizer import ByteBPETokenizer


class PromotionGateTest(unittest.TestCase):
    def _checkpoint(self, root: Path, name: str, *, metadata: dict | None = None, seed: int = 1) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        tokenizer = ByteBPETokenizer(())
        model = GenesisLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=16,
                d_model=16,
                n_heads=4,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        training = root / f"{name}-training.pt"
        inference = root / f"{name}.pt"
        save_checkpoint(
            training,
            model=model,
            optimizer=optimizer,
            step=10,
            metadata=metadata or {},
            tokenizer=tokenizer,
            batch_generator=generator,
        )
        export_inference_checkpoint(training, inference)
        return inference

    def _write_json(self, path: Path, value: dict) -> Path:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def _records(self, root: Path, parent: Path, candidate: Path, *, parent_loss=3.0, candidate_loss=2.9):
        hardware = {"machine": "x86_64", "processor": "fixture", "cpu_count": 8, "torch_version": "x", "torch_threads": 1}

        def evaluation(checkpoint: Path, loss: float) -> dict:
            return {
                "suite_version": "m3-v1",
                "suite_sha256": "s" * 64,
                "checkpoint_sha256": sha256_file(checkpoint),
                "data_manifest_sha256": "d" * 64,
                "contamination": {"blocking": False, "exact_overlap_count": 0},
                "primary_metric": {"name": "validation_loss", "value": loss, "lower_is_better": True},
            }

        def benchmark(checkpoint: Path, *, train_tps=100.0, decode_tps=100.0, rss=100.0) -> dict:
            return {
                "format_version": "1.0",
                "checkpoint_sha256": sha256_file(checkpoint),
                "device": "cpu",
                "hardware": hardware,
                "parameter_count": 12345,
                "training": {"tokens_per_second": train_tps},
                "decode": {"tokens_per_second": decode_tps},
                "peak_process_rss_mb": rss,
            }

        paths = {
            "parent_eval": self._write_json(root / "parent-eval.json", evaluation(parent, parent_loss)),
            "candidate_eval": self._write_json(root / "candidate-eval.json", evaluation(candidate, candidate_loss)),
            "parent_bench": self._write_json(root / "parent-bench.json", benchmark(parent)),
            "candidate_bench": self._write_json(root / "candidate-bench.json", benchmark(candidate)),
        }
        return paths

    def _fixture(self, root: Path):
        parent = self._checkpoint(root, "parent", seed=1)
        parent_hash = sha256_file(parent)
        candidate = self._checkpoint(
            root,
            "candidate",
            seed=2,
            metadata={
                "self_improvement": {
                    "policy": "candidate-training-v1",
                    "parent_checkpoint_sha256": parent_hash,
                    "experience_loss_before": 2.0,
                    "experience_loss_after": 1.0,
                    "experience_loss_decreased": True,
                }
            },
        )
        return parent, candidate, self._records(root, parent, candidate)

    def _decide(self, parent: Path, candidate: Path, paths: dict):
        return decide_promotion(
            parent_checkpoint=parent,
            candidate_checkpoint=candidate,
            parent_evaluation=paths["parent_eval"],
            candidate_evaluation=paths["candidate_eval"],
            parent_benchmark=paths["parent_bench"],
            candidate_benchmark=paths["candidate_bench"],
            policy_path=Path("evals/promotion-v1.json"),
        )

    def test_all_gates_pass_and_decision_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, candidate, paths = self._fixture(root)
            first = self._decide(parent, candidate, paths)
            second = self._decide(parent, candidate, paths)
            self.assertTrue(first["promoted"])
            self.assertEqual(first, second)
            self.assertEqual(first["decision"], "promote")
            self.assertTrue(all(gate["passed"] for gate in first["gates"]))

    def test_validation_and_throughput_regressions_block_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, candidate, paths = self._fixture(root)
            candidate_eval = json.loads(paths["candidate_eval"].read_text())
            candidate_eval["primary_metric"]["value"] = 2.99
            self._write_json(paths["candidate_eval"], candidate_eval)
            candidate_bench = json.loads(paths["candidate_bench"].read_text())
            candidate_bench["decode"]["tokens_per_second"] = 70.0
            self._write_json(paths["candidate_bench"], candidate_bench)
            result = self._decide(parent, candidate, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertFalse(result["promoted"])
            self.assertIn("validation_improvement", failed)
            self.assertIn("decode_throughput", failed)

    def test_contamination_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, candidate, paths = self._fixture(root)
            candidate_eval = json.loads(paths["candidate_eval"].read_text())
            candidate_eval["contamination"] = {"blocking": True, "exact_overlap_count": 1}
            self._write_json(paths["candidate_eval"], candidate_eval)
            result = self._decide(parent, candidate, paths)
            gate = next(gate for gate in result["gates"] if gate["name"] == "exact_contamination")
            self.assertFalse(gate["passed"])
            self.assertFalse(result["promoted"])

    def test_mismatched_suite_or_lineage_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, candidate, paths = self._fixture(root)
            candidate_eval = json.loads(paths["candidate_eval"].read_text())
            candidate_eval["suite_version"] = "other"
            self._write_json(paths["candidate_eval"], candidate_eval)
            with self.assertRaises(ValueError):
                self._decide(parent, candidate, paths)

            wrong_candidate = self._checkpoint(
                root,
                "wrong-candidate",
                seed=3,
                metadata={
                    "self_improvement": {
                        "policy": "candidate-training-v1",
                        "parent_checkpoint_sha256": "0" * 64,
                        "experience_loss_before": 2.0,
                        "experience_loss_after": 1.0,
                        "experience_loss_decreased": True,
                    }
                },
            )
            wrong_paths = self._records(root, parent, wrong_candidate)
            with self.assertRaises(ValueError):
                self._decide(parent, wrong_candidate, wrong_paths)


if __name__ == "__main__":
    unittest.main()
