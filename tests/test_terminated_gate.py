import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.ingest import sha256_file
from genesis_ai.terminated_gate import decide_terminated_promotion
from genesis_ai.terminated_training import TERMINATED_DATASET_VERSION, TERMINATED_TRAINING_POLICY_VERSION


class TerminatedGateTest(unittest.TestCase):
    def _write(self, path: Path, value: dict) -> Path:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    def _fixture(self, root: Path):
        baseline_checkpoint = Path("checkpoints/genesis-tiny-v0.pt")
        baseline_hash = sha256_file(baseline_checkpoint)
        candidate = root / "candidate.pt"
        candidate.write_bytes(b"terminated-candidate-fixture")
        candidate_hash = sha256_file(candidate)

        training = {
            "training_policy": TERMINATED_TRAINING_POLICY_VERSION,
            "dataset_version": TERMINATED_DATASET_VERSION,
            "termination_delimiter": "\n",
            "inference_checkpoint_sha256": candidate_hash,
            "parameter_count": 1_895_808,
            "target_training_tokens": 2_000_000,
            "processed_tokens": 2_001_920,
            "procedural_step_fraction": 0.8,
            "public_step_fraction": 0.2,
            "cash_compute_cost_usd": 0.0,
            "termination": {
                "delimiter": "\n",
                "schedule_target_updates": 12_512,
                "schedule_unique_updates": 12_512,
                "first_response_target_coverage": 1.0,
                "terminator_target_coverage": 1.0,
                "one_target_per_generation_context": True,
                "right_padding_after_predictor_only": True,
            },
        }
        reproduction = {
            "repro_version": "m6-repro-v1",
            "primary_checkpoint_sha256": candidate_hash,
            "reproducible": True,
        }
        suite_sha = "v" * 64
        code_task_sha = "edea452777c7328fd13d550ced322bd5815eac664be5f3924047f15122cc17c8"
        baseline_domain = {
            "suite_version": "m6-domain-selection-v2",
            "suite_sha256": suite_sha,
            "checkpoint_sha256": baseline_hash,
            "termination": {"delimiter": "\n", "required": True},
            "domains": {"code": {"task_set_sha256": code_task_sha, "exact_accuracy": 0.0, "termination_rate": 0.0}},
        }
        candidate_domain = json.loads(json.dumps(baseline_domain))
        candidate_domain["checkpoint_sha256"] = candidate_hash
        candidate_domain["domains"]["code"].update({"exact_accuracy": 0.10, "termination_rate": 0.10})

        baseline_m3 = {
            "suite_version": "m3-v1",
            "suite_sha256": "s" * 64,
            "data_manifest_sha256": "d" * 64,
            "checkpoint_sha256": baseline_hash,
            "primary_metric": {"name": "validation_loss", "value": 3.0, "lower_is_better": True},
            "contamination": {"blocking": False, "exact_overlap_count": 0},
        }
        candidate_m3 = json.loads(json.dumps(baseline_m3))
        candidate_m3["checkpoint_sha256"] = candidate_hash
        candidate_m3["primary_metric"]["value"] = 3.01

        paths = {
            "training": self._write(root / "training.json", training),
            "reproduction": self._write(root / "reproduction.json", reproduction),
            "baseline_domain": self._write(root / "baseline-domain.json", baseline_domain),
            "candidate_domain": self._write(root / "candidate-domain.json", candidate_domain),
            "baseline_m3": self._write(root / "baseline-m3.json", baseline_m3),
            "candidate_m3": self._write(root / "candidate-m3.json", candidate_m3),
        }
        return baseline_checkpoint, candidate, paths

    def _decide(self, baseline: Path, candidate: Path, paths: dict):
        return decide_terminated_promotion(
            baseline_checkpoint=baseline,
            candidate_checkpoint=candidate,
            training_run_path=paths["training"],
            reproduction_path=paths["reproduction"],
            baseline_domain_path=paths["baseline_domain"],
            candidate_domain_path=paths["candidate_domain"],
            baseline_m3_path=paths["baseline_m3"],
            candidate_m3_path=paths["candidate_m3"],
            ladder_result_path="research/m6-scaling-ladder-v1.json",
            curriculum_lock_path="research/m6-code-curriculum-v1.json",
        )

    def test_all_terminated_gates_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate, paths = self._fixture(Path(directory))
            result = self._decide(baseline, candidate, paths)
            self.assertTrue(result["promoted"])
            self.assertEqual(result["decision"], "promote")
            self.assertEqual(result["primary_capability"]["absolute_gain"], 0.10)
            self.assertTrue(all(gate["passed"] for gate in result["gates"]))

    def test_termination_protocol_is_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate, paths = self._fixture(root)
            training = json.loads(paths["training"].read_text())
            training["termination"]["terminator_target_coverage"] = 0.99
            self._write(paths["training"], training)
            result = self._decide(baseline, candidate, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertIn("termination_protocol", failed)

    def test_same_suite_capability_and_m3_are_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate, paths = self._fixture(root)
            domain = json.loads(paths["candidate_domain"].read_text())
            domain["domains"]["code"]["exact_accuracy"] = 0.0
            self._write(paths["candidate_domain"], domain)
            m3 = json.loads(paths["candidate_m3"].read_text())
            m3["primary_metric"]["value"] = 3.2
            self._write(paths["candidate_m3"], m3)
            result = self._decide(baseline, candidate, paths)
            failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
            self.assertIn("code_exact_accuracy_v2", failed)
            self.assertIn("m3_validation_loss", failed)


if __name__ == "__main__":
    unittest.main()
