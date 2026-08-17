import unittest
from pathlib import Path

from genesis_ai.model import GenesisLM
from genesis_ai.scale_5m_contract import LADDER_SUITES, load_scale_contract


EXPERIMENT = Path("experiments/m6-scale-5m-rope-v1.json")
FINALIST = Path("research/m6-architecture-finalist-v1.json")
PREFLIGHT = Path("research/m6-scale-5m-rope-preflight-v1.json")


class Scale5MContractTest(unittest.TestCase):
    def test_contract_consumes_only_reproduced_preflighted_rope_architecture(self):
        experiment, finalist, preflight, config = load_scale_contract(
            experiment_path=EXPERIMENT,
            finalist_path=FINALIST,
            preflight_path=PREFLIGHT,
        )
        self.assertTrue(finalist["decision"]["passed"])
        self.assertEqual(finalist["accepted_architecture"], "rope-only")
        self.assertTrue(preflight["projection"]["passes_remote_cpu_time_budget"])
        self.assertEqual(config.position_encoding, "rotary")
        self.assertEqual(GenesisLM(config).parameter_count(), 4_954_624)
        self.assertEqual(experiment["training"]["target_training_tokens"], 20_000_000)
        self.assertEqual(experiment["training"]["examples_per_domain"], 8_192)
        self.assertEqual(experiment["training"]["procedural_step_fraction"], 0.4)
        self.assertEqual(experiment["training"]["public_step_fraction"], 0.6)

    def test_all_five_frozen_ladder_suites_exist(self):
        self.assertEqual(len(LADDER_SUITES), 5)
        self.assertTrue(all(path.is_file() for path in LADDER_SUITES))

    def test_promotion_thresholds_are_frozen_before_training(self):
        experiment, _, _, _ = load_scale_contract(
            experiment_path=EXPERIMENT,
            finalist_path=FINALIST,
            preflight_path=PREFLIGHT,
        )
        gate = experiment["promotion_gate"]
        self.assertEqual(gate["minimum_gci_absolute_gain"], 3.0)
        self.assertEqual(gate["minimum_code_exact_accuracy"], 0.9)
        self.assertEqual(gate["minimum_math_exact_accuracy"], 1 / 60)
        self.assertEqual(gate["minimum_structured_exact_accuracy"], 1 / 60)
        self.assertEqual(gate["maximum_m3_loss_regression_fraction"], 0.02)
        self.assertTrue(gate["semantic_reproduction_required"])
        self.assertTrue(gate["zero_cash_compute_required"])


if __name__ == "__main__":
    unittest.main()
