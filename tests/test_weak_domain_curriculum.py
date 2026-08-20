from __future__ import annotations

import random
import unittest

from genesis_ai.challenger import build_task
from genesis_ai.weak_domain_curriculum import (
    FOCUS_EXAMPLES_BY_STAGE,
    build_variant_pair,
)
from genesis_ai.weak_domain_funnel import VARIANTS, build_catalog
from genesis_ai.weak_domain_training import SCREEN_REPLAY_EXAMPLES_BY_BUDGET


class WeakDomainCurriculumTests(unittest.TestCase):
    @staticmethod
    def _sort_task() -> dict:
        rng = random.Random(246)
        for _ in range(100):
            task = build_task(rng, "structured", 5)
            if "sort this integer array ascending:" in task["prompt"]:
                return task
        raise AssertionError("failed to generate deterministic ascending-sort fixture")

    def test_every_catalog_variant_has_deterministic_supervision(self) -> None:
        structured_task = self._sort_task()
        math_task = build_task(random.Random(247), "math", 5)
        for variant in VARIANTS:
            variant_id = str(variant["id"])
            task = math_task if variant_id == "math-operation-level" else structured_task
            first = build_variant_pair(variant_id, task, ordinal=3)
            second = build_variant_pair(variant_id, task, ordinal=3)
            self.assertEqual(first, second)
            prompt, response, supervision = first
            self.assertTrue(prompt)
            self.assertTrue(response)
            self.assertTrue(supervision)
            self.assertNotIn("\n", response)

    def test_structured_families_are_materially_distinct(self) -> None:
        task = self._sort_task()
        outputs = {}
        for variant in VARIANTS:
            variant_id = str(variant["id"])
            if variant_id == "math-operation-level":
                continue
            outputs[variant_id] = build_variant_pair(variant_id, task, ordinal=3)[:2]
        self.assertEqual(len(outputs), 6)
        self.assertGreaterEqual(len(set(outputs.values())), 5)
        self.assertNotEqual(outputs["structured-full-sort"], outputs["structured-pairwise-rank"])
        self.assertNotEqual(outputs["structured-prefix-next"], outputs["structured-partial-completion"])

    def test_focus_and_replay_counts_are_predeclared_before_screening(self) -> None:
        catalog = build_catalog()
        tiny_budget = int(catalog["stages"]["tiny"]["token_budget"])
        medium_budget = int(catalog["stages"]["medium"]["token_budget"])
        self.assertEqual(FOCUS_EXAMPLES_BY_STAGE, {"tiny": 256, "medium": 1024})
        self.assertEqual(SCREEN_REPLAY_EXAMPLES_BY_BUDGET[tiny_budget], 128)
        self.assertEqual(SCREEN_REPLAY_EXAMPLES_BY_BUDGET[medium_budget], 256)
        self.assertLess(2 * FOCUS_EXAMPLES_BY_STAGE["tiny"], 225_000 // 128)

    def test_math_variant_keeps_exact_integer_target_format(self) -> None:
        task = build_task(random.Random(248), "math", 4)
        prompt, response, supervision = build_variant_pair("math-operation-level", task, ordinal=0)
        self.assertEqual(prompt, task["prompt"])
        int(response)
        self.assertIn("operation-level", supervision)


if __name__ == "__main__":
    unittest.main()
