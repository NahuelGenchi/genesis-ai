import unittest

from genesis_ai.benchmark import cost_per_million_tokens


class BenchmarkTest(unittest.TestCase):
    def test_cost_formula(self):
        self.assertAlmostEqual(cost_per_million_tokens(2.0, 1000.0), 2.0 * 1_000_000 / 3_600_000)
        self.assertEqual(cost_per_million_tokens(0.0, 1000.0), 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            cost_per_million_tokens(-1.0, 1000.0)
        with self.assertRaises(ValueError):
            cost_per_million_tokens(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
