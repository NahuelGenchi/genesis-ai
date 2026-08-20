import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.terminated_eval import load_terminated_suite
from genesis_ai.weak_domain_funnel import FUNNEL_VERSION
from genesis_ai.weak_domain_screen import build_development_suite, rank_results


class WeakDomainScreenTests(unittest.TestCase):
    def test_development_suite_is_deterministic_and_evaluator_compatible(self):
        frozen = Path("evals/m6-domain-selection-v2.json")
        first = build_development_suite(frozen_suite_path=frozen)
        second = build_development_suite(frozen_suite_path=frozen)
        self.assertEqual(first, second)
        self.assertEqual(first["tasks_per_domain"], 20)
        self.assertNotEqual(first["base_seed"], load_terminated_suite(frozen)["base_seed"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "dev.json")
            path.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(load_terminated_suite(path), first)

    def test_rank_requires_all_variants_and_preserves_zero_authority(self):
        variants = [
            "structured-full-sort",
            "structured-pairwise-rank",
            "structured-prefix-next",
            "structured-partial-completion",
            "structured-length-progression",
            "structured-mixed-decomposition",
            "math-operation-level",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, variant in enumerate(variants):
                payload = {
                    "research_funnel_version": FUNNEL_VERSION,
                    "variant_id": variant,
                    "processed_tokens": 225280,
                    "weak_domain_gain_pp": float(index),
                    "code_retention_pp": 95.0,
                    "development_oracle_loss": 3.0 - index * 0.01,
                    "holdout_prompt_overlap_count": 0,
                    "screening_only": True,
                    "promotion_authority": False,
                }
                Path(root, f"result-{variant}.json").write_text(json.dumps(payload), encoding="utf-8")
            ranked = rank_results(results_dir=root, keep=3)
        self.assertEqual(ranked["candidate_count"], 7)
        self.assertEqual(ranked["survivors"], ["math-operation-level", "structured-mixed-decomposition", "structured-length-progression"])
        self.assertTrue(ranked["screening_only"])
        self.assertFalse(ranked["promotion_authority"])
        self.assertEqual(ranked["cash_compute_cost_usd"], 0.0)

    def test_rank_fails_closed_on_holdout_overlap(self):
        variants = [
            "structured-full-sort",
            "structured-pairwise-rank",
            "structured-prefix-next",
            "structured-partial-completion",
            "structured-length-progression",
            "structured-mixed-decomposition",
            "math-operation-level",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant in variants:
                payload = {
                    "variant_id": variant,
                    "processed_tokens": 225280,
                    "weak_domain_gain_pp": 0.0,
                    "code_retention_pp": 95.0,
                    "development_oracle_loss": 3.0,
                    "holdout_prompt_overlap_count": 1 if variant == variants[0] else 0,
                    "screening_only": True,
                    "promotion_authority": False,
                }
                Path(root, f"result-{variant}.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                rank_results(results_dir=root, keep=3)


if __name__ == "__main__":
    unittest.main()
