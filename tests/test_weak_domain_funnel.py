from __future__ import annotations

import json
import unittest

from genesis_ai.weak_domain_funnel import (
    MEDIUM_TOKEN_BUDGET,
    NORMAL_TOKEN_BUDGET,
    TINY_TOKEN_BUDGET,
    VARIANTS,
    build_catalog,
    select_survivors,
    validate_catalog,
)


class WeakDomainFunnelTests(unittest.TestCase):
    def test_catalog_is_deterministic_and_predeclared(self) -> None:
        left = build_catalog()
        right = build_catalog()
        self.assertEqual(left, right)
        validate_catalog(left)
        self.assertEqual(len(left["variants"]), 7)
        self.assertEqual({item["id"] for item in left["variants"]}, {item["id"] for item in VARIANTS})
        self.assertEqual(TINY_TOKEN_BUDGET / NORMAL_TOKEN_BUDGET, 0.075)
        self.assertEqual(MEDIUM_TOKEN_BUDGET / NORMAL_TOKEN_BUDGET, 0.25)
        self.assertTrue(all(item["screening_only"] for item in left["variants"]))
        self.assertTrue(all(item["promotion_authority"] is False for item in left["variants"]))
        self.assertEqual(left["safety_contract"]["cash_compute_cost_usd"], 0.0)

    def test_catalog_hash_fails_closed_on_mutation(self) -> None:
        catalog = build_catalog()
        catalog["stages"]["tiny"]["token_budget"] += 1
        with self.assertRaisesRegex(ValueError, "catalog hash mismatch"):
            validate_catalog(catalog)

    def test_successive_halving_is_equal_budget_and_deterministic(self) -> None:
        results = [
            {
                "variant_id": "a",
                "processed_tokens": 225_000,
                "weak_domain_gain_pp": 2.0,
                "code_retention_pp": 95.0,
                "development_oracle_loss": 2.0,
                "holdout_prompt_overlap_count": 0,
                "screening_only": True,
                "promotion_authority": False,
            },
            {
                "variant_id": "b",
                "processed_tokens": 225_000,
                "weak_domain_gain_pp": 2.0,
                "code_retention_pp": 96.0,
                "development_oracle_loss": 2.5,
                "holdout_prompt_overlap_count": 0,
                "screening_only": True,
                "promotion_authority": False,
            },
            {
                "variant_id": "c",
                "processed_tokens": 225_000,
                "weak_domain_gain_pp": 1.0,
                "code_retention_pp": 100.0,
                "development_oracle_loss": 1.0,
                "holdout_prompt_overlap_count": 0,
                "screening_only": True,
                "promotion_authority": False,
            },
        ]
        survivors = select_survivors(results, keep=2)
        self.assertEqual([item["variant_id"] for item in survivors], ["b", "a"])

    def test_successive_halving_rejects_unequal_budget_or_holdout_overlap(self) -> None:
        base = {
            "weak_domain_gain_pp": 1.0,
            "code_retention_pp": 95.0,
            "development_oracle_loss": 2.0,
            "holdout_prompt_overlap_count": 0,
            "screening_only": True,
            "promotion_authority": False,
        }
        unequal = [
            {**base, "variant_id": "a", "processed_tokens": 225_000},
            {**base, "variant_id": "b", "processed_tokens": 224_999},
        ]
        with self.assertRaisesRegex(ValueError, "equal processed tokens"):
            select_survivors(unequal, keep=1)

        overlap = [
            {**base, "variant_id": "a", "processed_tokens": 225_000},
            {**base, "variant_id": "b", "processed_tokens": 225_000, "holdout_prompt_overlap_count": 1},
        ]
        with self.assertRaisesRegex(ValueError, "holdout overlap"):
            select_survivors(overlap, keep=1)

    def test_catalog_is_json_roundtrippable(self) -> None:
        catalog = build_catalog()
        roundtrip = json.loads(json.dumps(catalog, sort_keys=True))
        validate_catalog(roundtrip)
        self.assertEqual(roundtrip, catalog)


if __name__ == "__main__":
    unittest.main()
