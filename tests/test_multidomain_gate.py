import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.ingest import sha256_file
from genesis_ai.multidomain_curriculum import CURRICULUM_VERSION
from genesis_ai.multidomain_gate import gate
from genesis_ai.multidomain_training import TRAINING_POLICY_VERSION


def domain_result(code, math, structured, checkpoint="sha"):
    return {
        "suite_version": "m6-domain-selection-v2",
        "suite_sha256": "suite",
        "checkpoint_sha256": checkpoint,
        "domains": {
            "code": {"exact_accuracy": code},
            "math": {"exact_accuracy": math},
            "structured": {"exact_accuracy": structured},
        },
    }


class MultidomainGateTest(unittest.TestCase):
    def _write(self, root: Path, name: str, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_minimum_target_doubles_gci_and_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_ckpt = root / "baseline.pt"
            candidate_ckpt = root / "candidate.pt"
            baseline_ckpt.write_bytes(b"baseline")
            candidate_ckpt.write_bytes(b"candidate")
            training = self._write(root, "training.json", {
                "training_policy": TRAINING_POLICY_VERSION,
                "parent_checkpoint_sha256": sha256_file(baseline_ckpt),
                "inference_checkpoint_sha256": sha256_file(candidate_ckpt),
                "cash_compute_cost_usd": 0.0,
            })
            reproduction = self._write(root, "repro.json", {"reproducible": True, "weights_equal": True})
            curriculum = self._write(root, "curriculum.json", {
                "curriculum_version": CURRICULUM_VERSION,
                "cash_compute_cost_usd": 0.0,
                "evaluation": {"exact_prompt_overlap_count": 0},
            })
            baseline_domain = self._write(root, "baseline-domain.json", domain_result(0.95, 0.0, 0.0))
            candidate_domain = self._write(root, "candidate-domain.json", domain_result(0.90, 0.50, 0.50))
            baseline_m3 = self._write(root, "baseline-m3.json", {"evaluation": {"loss": 3.35}})
            candidate_m3 = self._write(root, "candidate-m3.json", {"evaluation": {"loss": 3.36}})
            result = gate(
                baseline_checkpoint=baseline_ckpt,
                candidate_checkpoint=candidate_ckpt,
                training_path=training,
                reproduction_path=reproduction,
                curriculum_path=curriculum,
                baseline_domain_path=baseline_domain,
                candidate_domain_path=candidate_domain,
                baseline_m3_path=baseline_m3,
                candidate_m3_path=candidate_m3,
            )
            self.assertTrue(result["promoted"])
            self.assertAlmostEqual(result["gci_v1"]["relative_percent_change"], 100.0)

    def test_code_regression_blocks_even_when_breadth_is_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_ckpt = root / "baseline.pt"
            candidate_ckpt = root / "candidate.pt"
            baseline_ckpt.write_bytes(b"baseline")
            candidate_ckpt.write_bytes(b"candidate")
            training = self._write(root, "training.json", {
                "training_policy": TRAINING_POLICY_VERSION,
                "parent_checkpoint_sha256": sha256_file(baseline_ckpt),
                "inference_checkpoint_sha256": sha256_file(candidate_ckpt),
                "cash_compute_cost_usd": 0.0,
            })
            reproduction = self._write(root, "repro.json", {"reproducible": True, "weights_equal": True})
            curriculum = self._write(root, "curriculum.json", {
                "curriculum_version": CURRICULUM_VERSION,
                "cash_compute_cost_usd": 0.0,
                "evaluation": {"exact_prompt_overlap_count": 0},
            })
            baseline_domain = self._write(root, "baseline-domain.json", domain_result(0.95, 0.0, 0.0))
            candidate_domain = self._write(root, "candidate-domain.json", domain_result(0.80, 0.80, 0.80))
            baseline_m3 = self._write(root, "baseline-m3.json", {"evaluation": {"loss": 3.35}})
            candidate_m3 = self._write(root, "candidate-m3.json", {"evaluation": {"loss": 3.35}})
            result = gate(
                baseline_checkpoint=baseline_ckpt,
                candidate_checkpoint=candidate_ckpt,
                training_path=training,
                reproduction_path=reproduction,
                curriculum_path=curriculum,
                baseline_domain_path=baseline_domain,
                candidate_domain_path=candidate_domain,
                baseline_m3_path=baseline_m3,
                candidate_m3_path=candidate_m3,
            )
            self.assertFalse(result["promoted"])
            self.assertFalse(next(item for item in result["gates"] if item["name"] == "code_exact")["passed"])


if __name__ == "__main__":
    unittest.main()
