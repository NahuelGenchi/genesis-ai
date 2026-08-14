import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.autonomous_curriculum import CURRICULUM_VERSION
from genesis_ai.autonomous_gate import gate
from genesis_ai.autonomous_training import TRAINING_POLICY_VERSION
from genesis_ai.improvement_controller import plan_next_cycle
from genesis_ai.ingest import sha256_file


SUITE_SHA = "b" * 64


def aggregate(code, math, structured):
    return {
        "suite_version": "m6-domain-selection-v2",
        "suite_sha256": SUITE_SHA,
        "difficulty": 1,
        "domains": {
            "code": {"exact_accuracy": code, "terminated_oracle_loss": 1.0, "termination_rate": 1.0},
            "math": {"exact_accuracy": math, "terminated_oracle_loss": 2.0, "termination_rate": 1.0},
            "structured": {"exact_accuracy": structured, "terminated_oracle_loss": 1.5, "termination_rate": 1.0},
        },
    }


class AutonomousGateTest(unittest.TestCase):
    def _write(self, root: Path, name: str, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _fixture(self, root: Path, candidate_domain):
        baseline_ckpt = root / "baseline.pt"
        candidate_ckpt = root / "candidate.pt"
        baseline_ckpt.write_bytes(b"baseline")
        candidate_ckpt.write_bytes(b"candidate")
        baseline_sha = sha256_file(baseline_ckpt)
        plan = plan_next_cycle(aggregate(0.95, 0.20, 0.50), incumbent_checkpoint_sha256=baseline_sha)
        plan_path = self._write(root, "plan.json", plan)
        curriculum = self._write(root, "curriculum.json", {
            "curriculum_version": CURRICULUM_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "incumbent_checkpoint_sha256": baseline_sha,
            "target_suite_sha256": SUITE_SHA,
            "exact_holdout_prompt_overlap_count": 0,
            "cash_compute_cost_usd": 0.0,
        })
        training = self._write(root, "training.json", {
            "training_policy": TRAINING_POLICY_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "parent_checkpoint_sha256": baseline_sha,
            "inference_checkpoint_sha256": sha256_file(candidate_ckpt),
            "target_training_tokens": plan["decision"]["target_training_tokens"],
            "cash_compute_cost_usd": 0.0,
        })
        reproduction = self._write(root, "reproduction.json", {"reproducible": True, "weights_equal": True})
        baseline_domain = self._write(root, "baseline-domain.json", aggregate(0.95, 0.20, 0.50))
        candidate_domain_path = self._write(root, "candidate-domain.json", candidate_domain)
        baseline_m3 = self._write(root, "baseline-m3.json", {"evaluation": {"loss": 3.0}, "contamination": {"blocking": False, "exact_overlap_count": 0}})
        candidate_m3 = self._write(root, "candidate-m3.json", {"evaluation": {"loss": 3.02}, "contamination": {"blocking": False, "exact_overlap_count": 0}})
        return {
            "plan_path": plan_path,
            "curriculum_path": curriculum,
            "baseline_checkpoint": baseline_ckpt,
            "candidate_checkpoint": candidate_ckpt,
            "training_path": training,
            "reproduction_path": reproduction,
            "baseline_domain_path": baseline_domain,
            "candidate_domain_path": candidate_domain_path,
            "baseline_m3_path": baseline_m3,
            "candidate_m3_path": candidate_m3,
        }

    def test_focus_gain_plus_gci_gain_with_preserved_domains_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = aggregate(0.94, 0.35, 0.50)
            result = gate(**self._fixture(root, values))
            self.assertTrue(result["promoted"])
            self.assertEqual(result["focus_domain"], "math")
            self.assertAlmostEqual(result["focus_absolute_gain"], 0.15)
            self.assertGreater(result["gci_v1"]["absolute_point_change"], 3.0)

    def test_nonfocus_regression_blocks_even_when_focus_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = aggregate(0.80, 0.50, 0.50)
            result = gate(**self._fixture(root, values))
            self.assertFalse(result["promoted"])
            regression_gate = next(item for item in result["gates"] if item["name"] == "nonfocus_regressions")
            self.assertFalse(regression_gate["passed"])


if __name__ == "__main__":
    unittest.main()
