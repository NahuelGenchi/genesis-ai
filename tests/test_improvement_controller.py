import unittest

from genesis_ai.improvement_controller import plan_next_cycle


SHA = "a" * 64
SUITE_SHA = "b" * 64


def evaluation(code, math, structured, *, code_loss=1.0, math_loss=1.0, structured_loss=1.0, difficulty=1):
    return {
        "suite_version": "m6-domain-selection-v2",
        "suite_sha256": SUITE_SHA,
        "difficulty": difficulty,
        "domains": {
            "code": {"exact_accuracy": code, "terminated_oracle_loss": code_loss, "termination_rate": 1.0},
            "math": {"exact_accuracy": math, "terminated_oracle_loss": math_loss, "termination_rate": 1.0},
            "structured": {"exact_accuracy": structured, "terminated_oracle_loss": structured_loss, "termination_rate": 1.0},
        },
    }


class ImprovementControllerTest(unittest.TestCase):
    def test_selects_weakest_domain_and_preserves_replay(self):
        plan = plan_next_cycle(evaluation(0.95, 0.20, 0.50), incumbent_checkpoint_sha256=SHA)
        self.assertEqual(plan["decision"]["focus_domain"], "math")
        self.assertEqual(plan["decision"]["mode"], "repair-weakest-domain")
        self.assertEqual(plan["decision"]["focus_training_tokens"], 2_000_000)
        self.assertEqual(plan["decision"]["replay_domains"], ["code", "structured"])
        self.assertEqual(plan["decision"]["cash_compute_cost_usd"], 0.0)

    def test_tie_breaks_by_higher_aggregate_loss(self):
        plan = plan_next_cycle(
            evaluation(0.50, 0.20, 0.20, math_loss=2.0, structured_loss=4.0),
            incumbent_checkpoint_sha256=SHA,
        )
        self.assertEqual(plan["decision"]["focus_domain"], "structured")

    def test_all_mastered_advances_difficulty(self):
        plan = plan_next_cycle(evaluation(0.90, 0.85, 0.80, difficulty=2), incumbent_checkpoint_sha256=SHA)
        self.assertEqual(plan["decision"]["mode"], "raise-difficulty")
        self.assertEqual(plan["decision"]["target_difficulty"], 3)
        self.assertEqual(plan["decision"]["focus_domain"], "structured")

    def test_private_holdout_content_is_rejected(self):
        value = evaluation(0.9, 0.8, 0.7)
        value["tasks"] = [{"prompt": "private"}]
        with self.assertRaises(ValueError):
            plan_next_cycle(value, incumbent_checkpoint_sha256=SHA)

    def test_plan_is_deterministic_and_hash_bound(self):
        first = plan_next_cycle(evaluation(0.95, 0.30, 0.40), incumbent_checkpoint_sha256=SHA)
        second = plan_next_cycle(evaluation(0.95, 0.30, 0.40), incumbent_checkpoint_sha256=SHA)
        self.assertEqual(first, second)
        self.assertEqual(len(first["plan_sha256"]), 64)

    def test_invalid_checkpoint_or_suite_identity_fails_closed(self):
        with self.assertRaises(ValueError):
            plan_next_cycle(evaluation(0.9, 0.8, 0.7), incumbent_checkpoint_sha256="short")
        value = evaluation(0.9, 0.8, 0.7)
        value["suite_sha256"] = "short"
        with self.assertRaises(ValueError):
            plan_next_cycle(value, incumbent_checkpoint_sha256=SHA)


if __name__ == "__main__":
    unittest.main()
