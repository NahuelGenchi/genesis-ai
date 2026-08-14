import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.ingest import sha256_file
from genesis_ai.scale_gate import decide_scale_promotion


class ScaleGateTest(unittest.TestCase):
    def _write(self, path: Path, value: dict) -> Path:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def _fixture(self, root: Path):
        checkpoint = root / "candidate.pt"
        checkpoint.write_bytes(b"candidate-checkpoint-fixture")
        checkpoint_hash = sha256_file(checkpoint)

        training = {
            "training_policy": "m6-micro-2m-training-v1",
            "inference_checkpoint_sha256": checkpoint_hash,
            "parameter_count": 1_895_808,
            "target_training_tokens": 2_000_000,
            "processed_tokens": 2_001_920,
            "procedural_step_fraction": 0.8,
            "public_step_fraction": 0.2,
            "cash_compute_cost_usd": 0.0,
        }
        reproduction = {
            "repro_version": "m6-repro-v1",
            "primary_checkpoint_sha256": checkpoint_hash,
            "reproducible": True,
        }
        baseline_domain = json.loads(Path("research/m6-domain-selection-v1.json").read_text(encoding="utf-8"))
        candidate_domain = json.loads(json.dumps(baseline_domain))
        candidate_domain["checkpoint_sha256"] = checkpoint_hash
        candidate_domain["domains"]["code"]["exact_accuracy"] = 0.10
        candidate_domain["domains"]["code"]["exact_correct"] = 6

        base_m3 = {
            "suite_version": "m3-v1",
            "suite_sha256": "s" * 64,
            "data_manifest_sha256": "d" * 64,
            "checkpoint_sha256": "b" * 64,
            "primary_metric": {"name": "validation_loss", "value": 3.0, "lower_is_better": True},
            "contamination": {"blocking": False, "exact_overlap_count": 0},
        }
        candidate_m3 = json.loads(json.dumps(base_m3))
        candidate_m3["checkpoint_sha256"] = checkpoint_hash
        candidate_m3["primary_metric"]["value"] = 3.01

        paths = {
            "training": self._write(root / "training.json", training),
            "reproduction": self._write(root / "repro.json", reproduction),
            "baseline_domain": self._write(root / "baseline-domain.json", baseline_domain),
            "candidate_domain": self._write(root / "candidate-domain.json", candidate_domain),
            "baseline_m3": self._write(root / "baseline-m3.json", base_m3),
            "candidate_m3": self._write(root / "candidate-m3.json", candidate_m3),
        }
        return checkpoint, paths

    def _decide(self, checkpoint: Path, paths: dict):
        return decide_scale_promotion(
            candidate_checkpoint=checkpoint,
            training_run_path=paths["training"],
            reproduction_path=paths["reproduction"],
            baseline_domain_path=paths["baseline_domain"],
            candidate_domain_path=paths["candidate_domain"],
            baseline_m3_path=paths["baseline_m3"],
            candidate_m3_path=paths["candidate_m3"],
            ladder_result_path="research/m6-scaling-ladder-v1.json",
            curriculum_lock_path="research/m6-code-curriculum-v1.json",
        )

    def test_all_scale_gates_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, paths = self._fixture(Path(directory))
            result = self._decide(checkpoint, paths)
            self.assertTrue(result["promoted"])
            self.assertEqual(result["decision"], "promote")
            self.assertEqual(result["primary_capability"]["absolute_gain"], 0.10)
            self.assertTrue(all(gate["passed"] for gate in result["gates"]))
            self.assertEqual(result, self._decide(checkpoint, paths))

    def test_insufficient_capability_or_m3_regression_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, paths = self._fixture(root)
            candidate_domain = json.loads(paths["candidate_domain"].read_text())
            candidate_domain["domains"]["code"]["exact_accuracy"] = 0.01
            self._write(paths["candidate_domain"], candidate_domain)
            candidate_m3 = json.loads(paths["candidate_m3"].read_text())
            candidate_m3["primary_metric"]["value"] = 3.2
            self._write(paths["candidate_m3"], candidate_m3)
            result = self._decide(checkpoint, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertFalse(result["promoted"])
            self.assertIn("code_exact_accuracy", failed)
            self.assertIn("m3_validation_loss", failed)

    def test_reproducibility_and_contamination_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, paths = self._fixture(root)
            repro = json.loads(paths["reproduction"].read_text())
            repro["reproducible"] = False
            self._write(paths["reproduction"], repro)
            m3 = json.loads(paths["candidate_m3"].read_text())
            m3["contamination"] = {"blocking": True, "exact_overlap_count": 1}
            self._write(paths["candidate_m3"], m3)
            result = self._decide(checkpoint, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertIn("reproducibility", failed)
            self.assertIn("m3_exact_contamination", failed)


if __name__ == "__main__":
    unittest.main()
