import unittest

from genesis_ai.capability_index import compare_results, score_result


def result(*, code: float, math: float, structured: float, suite: str = "hash"):
    return {
        "suite_version": "m6-domain-selection-v2",
        "suite_sha256": suite,
        "checkpoint_sha256": "checkpoint",
        "domains": {
            "code": {"exact_accuracy": code},
            "math": {"exact_accuracy": math},
            "structured": {"exact_accuracy": structured},
        },
    }


class CapabilityIndexTest(unittest.TestCase):
    def test_current_candidate_scores_31_6667(self):
        score = score_result(result(code=0.95, math=0.0, structured=0.0))
        self.assertAlmostEqual(score["score"], 31.666666666666668)

    def test_zero_baseline_reports_absolute_gain_not_fake_zero_percent(self):
        comparison = compare_results(
            result(code=0.0, math=0.0, structured=0.0),
            result(code=0.95, math=0.0, structured=0.0),
        )
        self.assertAlmostEqual(comparison["absolute_point_change"], 31.666666666666668)
        self.assertIsNone(comparison["relative_percent_change"])
        self.assertEqual(comparison["relative_percent_note"], "N/A (zero baseline)")

    def test_target_minimum_is_100_percent_relative_gci_improvement(self):
        comparison = compare_results(
            result(code=0.95, math=0.0, structured=0.0),
            result(code=0.90, math=0.50, structured=0.50),
        )
        self.assertAlmostEqual(comparison["baseline"]["score"], 31.666666666666668)
        self.assertAlmostEqual(comparison["candidate"]["score"], 63.333333333333336)
        self.assertAlmostEqual(comparison["relative_percent_change"], 100.0)

    def test_cross_suite_comparison_fails_closed(self):
        with self.assertRaises(ValueError):
            compare_results(
                result(code=0.0, math=0.0, structured=0.0, suite="a"),
                result(code=1.0, math=1.0, structured=1.0, suite="b"),
            )


if __name__ == "__main__":
    unittest.main()
