import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.research_funnel import STAGES, VARIANTS, rank_screen_directory, score_screen


def evaluation(*, code=0.95, math=0.0, structured=0.0, code_loss=1.0, math_loss=2.0, structured_loss=3.0):
    return {
        "domains": {
            "code": {"exact_accuracy": code, "terminated_oracle_loss": code_loss},
            "math": {"exact_accuracy": math, "terminated_oracle_loss": math_loss},
            "structured": {"exact_accuracy": structured, "terminated_oracle_loss": structured_loss},
        }
    }


def training(tokens):
    return {"processed_tokens": tokens, "cash_compute_cost_usd": 0.0}


class ResearchFunnelTest(unittest.TestCase):
    def test_tiny_and_medium_are_small_fractions_of_full_budget(self):
        self.assertEqual(STAGES["tiny"].target_training_tokens, 150_000)
        self.assertEqual(STAGES["medium"].target_training_tokens, 750_000)
        self.assertEqual(STAGES["full"].target_training_tokens, 3_000_000)
        self.assertEqual(STAGES["tiny"].survivors, 3)
        self.assertEqual(STAGES["medium"].survivors, 1)

    def test_variant_catalog_is_materially_distinct_and_covers_both_weak_domains(self):
        ids = [item["id"] for item in VARIANTS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 6)
        self.assertEqual({item["focus_domain"] for item in VARIANTS}, {"math", "structured"})
        self.assertGreaterEqual(len({item["curriculum_mode"] for item in VARIANTS}), 6)

    def test_positive_weak_signal_advances_and_tracks_quality_per_token(self):
        result = score_screen(
            baseline=evaluation(),
            candidate=evaluation(structured=0.10, structured_loss=2.1),
            training=training(150_000),
            variant_id="structured-prefix-next-v1",
        )
        self.assertTrue(result["advance_eligible"])
        self.assertGreater(result["quality_per_million_tokens"], 0.0)
        self.assertFalse(result["promotion_authority"])

    def test_code_regression_kills_screen_even_with_weak_domain_gain(self):
        result = score_screen(
            baseline=evaluation(),
            candidate=evaluation(code=0.80, structured=0.20, structured_loss=2.0),
            training=training(150_000),
            variant_id="structured-full-sort-v1",
        )
        self.assertFalse(result["advance_eligible"])
        self.assertEqual(result["early_stop_reason"], "code-retention-boundary")

    def test_no_exact_gain_can_survive_tiny_screen_on_positive_oracle_loss_signal(self):
        result = score_screen(
            baseline=evaluation(),
            candidate=evaluation(structured_loss=2.4, math_loss=1.9),
            training=training(150_000),
            variant_id="structured-pairwise-rank-v1",
        )
        self.assertTrue(result["advance_eligible"])
        self.assertGreater(result["weak_oracle_loss_reduction"], 0.0)

    def test_rank_directory_keeps_only_top_three_tiny_survivors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(evaluation()), encoding="utf-8")
            candidates = root / "candidates"
            candidates.mkdir()
            variants = [item["id"] for item in VARIANTS[:5]]
            for ordinal, variant_id in enumerate(variants, 1):
                directory = candidates / variant_id
                directory.mkdir()
                candidate = evaluation(structured=min(0.05 * ordinal, 0.25), structured_loss=3.0 - 0.2 * ordinal)
                directory.joinpath("evaluation.json").write_text(json.dumps(candidate), encoding="utf-8")
                directory.joinpath("training.json").write_text(json.dumps(training(150_000)), encoding="utf-8")
            summary = rank_screen_directory(baseline_path=baseline, candidates_root=candidates, stage="tiny")
            self.assertTrue(summary["advance"])
            self.assertEqual(len(summary["survivor_ids"]), 3)
            self.assertFalse(summary["promotion_authority"])


if __name__ == "__main__":
    unittest.main()
