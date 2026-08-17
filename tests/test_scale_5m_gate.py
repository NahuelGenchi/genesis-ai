import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.ingest import sha256_file
from genesis_ai.scale_5m_contract import load_scale_contract
from genesis_ai.scale_5m_gate import evaluate_gate


EXPERIMENT = Path("experiments/m6-scale-5m-rope-v1.json")
FINALIST = Path("research/m6-architecture-finalist-v1.json")
PREFLIGHT = Path("research/m6-scale-5m-rope-preflight-v1.json")


def domain_result(checkpoint: str, *, code: float, math: float, structured: float):
    return {
        "suite_version": "m6-domain-selection-v2",
        "suite_sha256": "s" * 64,
        "checkpoint_sha256": checkpoint,
        "domains": {
            "code": {"exact_accuracy": code},
            "math": {"exact_accuracy": math},
            "structured": {"exact_accuracy": structured},
        },
    }


def m3(loss: float):
    return {
        "evaluation": {"loss": loss},
        "contamination": {"blocking": False, "exact_overlap_count": 0},
    }


def ladder(checkpoint: str):
    return {
        "metric_version": "gci-ladder-v1",
        "checkpoint_sha256": checkpoint,
        "difficulty_suite_sha256": {str(i): f"suite-{i}" for i in range(1, 6)},
        "ladder_score": 1.0,
        "worst_domain_exact_percent": 1.0,
    }


class Scale5MGateTest(unittest.TestCase):
    def _run(self, *, candidate_math=0.05, candidate_structured=0.05, candidate_code=0.95, candidate_loss=3.0):
        experiment, _, _, config = load_scale_contract(
            experiment_path=EXPERIMENT,
            finalist_path=FINALIST,
            preflight_path=PREFLIGHT,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curriculum = root / "curriculum.json"
            curriculum.write_text(json.dumps({
                "curriculum_version": "m6-scale-5m-curriculum-v1",
                "cash_compute_cost_usd": 0.0,
                "ladder_separation": {"exact_training_prompt_overlap_count": 0},
            }), encoding="utf-8")
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"candidate")
            training = {
                "training_policy": "m6-scale-5m-rope-training-v1",
                "cash_compute_cost_usd": 0.0,
                "parameter_count": 4_954_624,
                "config": config.to_dict(),
                "curriculum_sha256": sha256_file(curriculum),
                "schedule_sha256": "schedule",
                "processed_tokens": 20_003_840,
                "inference_checkpoint_sha256": sha256_file(checkpoint),
            }
            reproduction = {"repro_version": "m6-repro-v1", "reproducible": True}
            files = {
                "primary.json": training,
                "replica.json": training,
                "repro.json": reproduction,
                "baseline-domain.json": domain_result("base", code=0.95, math=0.0, structured=0.0),
                "candidate-domain.json": domain_result("candidate", code=candidate_code, math=candidate_math, structured=candidate_structured),
                "baseline-m3.json": m3(3.3551117),
                "candidate-m3.json": m3(candidate_loss),
                "baseline-ladder.json": ladder("base"),
                "candidate-ladder.json": ladder("candidate"),
            }
            paths = {}
            for name, payload in files.items():
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            return evaluate_gate(
                experiment_path=EXPERIMENT,
                finalist_path=FINALIST,
                preflight_path=PREFLIGHT,
                curriculum_path=curriculum,
                primary_training_path=paths["primary.json"],
                replica_training_path=paths["replica.json"],
                reproduction_path=paths["repro.json"],
                baseline_domain_path=paths["baseline-domain.json"],
                candidate_domain_path=paths["candidate-domain.json"],
                baseline_m3_path=paths["baseline-m3.json"],
                candidate_m3_path=paths["candidate-m3.json"],
                candidate_checkpoint_path=checkpoint,
                baseline_ladder_path=paths["baseline-ladder.json"],
                candidate_ladder_path=paths["candidate-ladder.json"],
            )

    def test_candidate_with_breadth_gain_and_m3_health_can_pass(self):
        result = self._run()
        self.assertTrue(result["promoted"])
        self.assertEqual(result["decision"], "promote")
        self.assertGreaterEqual(result["gci_v1"]["absolute_point_change"], 3.0)

    def test_structured_must_be_nonzero_even_when_gci_improves(self):
        result = self._run(candidate_math=0.15, candidate_structured=0.0, candidate_code=1.0)
        self.assertFalse(result["promoted"])
        failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
        self.assertIn("structured_exact_accuracy", failed)

    def test_code_floor_is_immutable(self):
        result = self._run(candidate_math=0.2, candidate_structured=0.2, candidate_code=0.85)
        self.assertFalse(result["promoted"])
        failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
        self.assertIn("code_exact_accuracy", failed)

    def test_m3_regression_above_two_percent_blocks_promotion(self):
        result = self._run(candidate_math=0.2, candidate_structured=0.2, candidate_code=1.0, candidate_loss=3.5)
        self.assertFalse(result["promoted"])
        failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
        self.assertIn("m3_validation_loss", failed)


if __name__ == "__main__":
    unittest.main()
