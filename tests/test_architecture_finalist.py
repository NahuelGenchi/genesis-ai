import unittest

from genesis_ai.architecture_finalist import (
    BASELINE_NAME,
    FRESH_SEEDS,
    MAX_PER_SEED_RELATIVE_REGRESSION,
    MIN_MEAN_RELATIVE_IMPROVEMENT,
    _definition_candidate,
)


class ArchitectureFinalistTest(unittest.TestCase):
    def test_reproduction_policy_is_frozen_and_nontrivial(self):
        self.assertEqual(FRESH_SEEDS, (102002, 102003))
        self.assertEqual(MIN_MEAN_RELATIVE_IMPROVEMENT, 0.005)
        self.assertEqual(MAX_PER_SEED_RELATIVE_REGRESSION, 0.01)
        self.assertEqual(BASELINE_NAME, "baseline-learned-layernorm-gelu")

    def test_definition_lookup_is_exact(self):
        definition = {"candidates": [{"name": "a", "config": {}}, {"name": "b", "config": {}}]}
        self.assertEqual(_definition_candidate(definition, "b")["name"], "b")
        with self.assertRaises(ValueError):
            _definition_candidate(definition, "missing")


if __name__ == "__main__":
    unittest.main()
