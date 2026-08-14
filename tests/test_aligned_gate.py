import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.aligned_gate import decide_aligned_promotion
from genesis_ai.aligned_training import ALIGNED_DATASET_VERSION, ALIGNED_TRAINING_POLICY_VERSION
from genesis_ai.ingest import sha256_file


class AlignedGateTest(unittest.TestCase):
    def _write(self, path: Path, value: dict) -> Path:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def _fixture(self, root: Path):
        checkpoint = root / "candidate.pt"
        checkpoint.write_bytes(b"aligned-candidate-checkpoint-fixture")
        checkpoint_hash = sha256_file(checkpoint)
        training = {
            "training_policy": ALIGNED_TRAINING_POLICY_VERSION,
            "dataset_version": ALIGNED_DATASET_VERSION,
            "inference_checkpoint_sha256": checkpoint_hash,
            "parameter_count": 1_895_808,
            "target_training_tokens": 2_000_000,
            "processed_tokens": 2_001_920,
            "procedural_step_fraction": 0.8,
            "public_step_fraction": 0.2,
            "cash_compute_cost_usd": 0.0,
            "alignment": {
                "dataset_response_targets": 61_559,
                "schedule_target_updates": 12_512,
                "schedule_unique_updates": 12_512,
                "first_response_targets": 4_096,
                "first_response_target_updates": 4_096,
                "first_response_target_coverage": 1.0,
                "continuation_target_updates": 8_416,
                "one_target_per_generation_context": True,
                "right_padding_after_predictor_only": True,
            },
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
        alignment = {
            "diagnostic_version": "m6-position-alignment-v1",
            "checkpoint_sha256": checkpoint_hash,
            "suite_sha256": candidate_domain["suite_sha256"],
            "task_set_sha256": candidate_domain["domains"]["code"]["task_set_sha256"],
            "generation_aligned_rolling": {
                "mean_loss": 1.25,
                "greedy_token_accuracy": 0.8,
                "first_token_greedy_correct_rate": 0.9,
                "all_greedy_tokens_correct_rate": 0.1,
            },
        }
        baseline_m3 = {
            "suite_version": "m3-v1",
            "suite_sha256": "s" * 64,
            "data_manifest_sha256": "d" * 64,
            "checkpoint_sha256": "b" * 64,
            "primary_metric": {"name": "validation_loss", "value": 3.0, "lower_is_better": True},
            "contamination": {"blocking": False, "exact_overlap_count": 0},
        }
        candidate_m3 = json.loads(json.dumps(baseline_m3))
        candidate_m3["checkpoint_sha256"] = checkpoint_hash
        candidate_m3["primary_metric"]["value"] = 3.01
        paths = {
            "training": self._write(root / "training.json", training),
            "reproduction": self._write(root / "reproduction.json", reproduction),
            "baseline_domain": self._write(root / "baseline-domain.json", baseline_domain),
            "candidate_domain": self._write(root / "candidate-domain.json", candidate_domain),
            "alignment": self._write(root / "alignment.json", alignment),
            "baseline_m3": self._write(root / "baseline-m3.json", baseline_m3),
            "candidate_m3": self._write(root / "candidate-m3.json", candidate_m3),
        }
        return checkpoint, paths

    def _decide(self, checkpoint: Path, paths: dict):
        return decide_aligned_promotion(
            candidate_checkpoint=checkpoint,
            training_run_path=paths["training"],
            reproduction_path=paths["reproduction"],
            baseline_domain_path=paths["baseline_domain"],
            candidate_domain_path=paths["candidate_domain"],
            candidate_alignment_path=paths["alignment"],
            baseline_m3_path=paths["baseline_m3"],
            candidate_m3_path=paths["candidate_m3"],
            ladder_result_path="research/m6-scaling-ladder-v1.json",
            curriculum_lock_path="research/m6-code-curriculum-v1.json",
        )

    def test_all_aligned_gates_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, paths = self._fixture(Path(directory))
            result = self._decide(checkpoint, paths)
            self.assertTrue(result["promoted"])
            self.assertEqual(result["decision"], "promote")
            self.assertEqual(result["primary_capability"]["absolute_gain"], 0.10)
            self.assertEqual(result["auxiliary_alignment"]["rolling_greedy_token_accuracy"], 0.8)
            self.assertTrue(all(gate["passed"] for gate in result["gates"]))

    def test_alignment_contract_is_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, paths = self._fixture(root)
            training = json.loads(paths["training"].read_text())
            training["alignment"]["first_response_target_coverage"] = 0.99
            self._write(paths["training"], training)
            result = self._decide(checkpoint, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertFalse(result["promoted"])
            self.assertIn("generation_alignment", failed)

    def test_capability_and_m3_remain_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, paths = self._fixture(root)
            domain = json.loads(paths["candidate_domain"].read_text())
            domain["domains"]["code"]["exact_accuracy"] = 0.0
            domain["domains"]["code"]["exact_correct"] = 0
            self._write(paths["candidate_domain"], domain)
            m3 = json.loads(paths["candidate_m3"].read_text())
            m3["primary_metric"]["value"] = 3.2
            self._write(paths["candidate_m3"], m3)
            result = self._decide(checkpoint, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertIn("code_exact_accuracy", failed)
            self.assertIn("m3_validation_loss", failed)


if __name__ == "__main__":
    unittest.main()
