import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from genesis_ai.multidomain_curriculum import frozen_holdouts
from genesis_ai.weak_domain_curriculum import _focus_example, _generate_focus_records, build_plan


class FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3]


def structured_task(values):
    return {
        "id": "task-structured",
        "generator": "procedural-v1",
        "domain": "structured",
        "prompt": f"Return only JSON: sort this integer array ascending: {json.dumps(values, separators=(',', ':'))}",
        "verifier": {"kind": "json_exact", "expected": sorted(values)},
    }


def math_task(a, b):
    return {
        "id": "task-math",
        "generator": "procedural-v1",
        "domain": "math",
        "prompt": f"Return only the integer result of {a} + {b}.",
        "verifier": {"kind": "integer_exact", "expected": a + b},
    }


class WeakDomainCurriculumTest(unittest.TestCase):
    def test_structured_modes_expose_distinct_intermediate_targets(self):
        task = structured_task([3, 1, 2, 2])
        modes = {
            "pairwise-rank",
            "prefix-next",
            "partial-completion",
            "short-to-long",
            "mixed-decomposition",
            "full-sort",
        }
        outputs = {_focus_example(task, mode=mode, ordinal=1)[2] for mode in modes}
        self.assertGreaterEqual(len(outputs), 5)

    def test_math_decomposition_keeps_exact_deterministic_supervision(self):
        task = math_task(-7, 12)
        prompt, response, kind = _focus_example(task, mode="operation-decomposition", ordinal=0)
        self.assertIn("[left,right,sum]", prompt)
        self.assertEqual(json.loads(response), [-7, 12, 5])
        self.assertEqual(kind, "operation-decomposition")

    def test_source_task_overlap_is_rejected_even_when_transformed_prompt_differs(self):
        blocked = structured_task([3, 2, 1])
        allowed = structured_task([8, -1, 4])
        blocked_hash = hashlib.sha256(blocked["prompt"].encode("utf-8")).hexdigest()
        with patch(
            "genesis_ai.weak_domain_curriculum.build_task",
            side_effect=[blocked, allowed],
        ):
            records, metrics = _generate_focus_records(
                tokenizer=FakeTokenizer(),
                domain="structured",
                count=1,
                difficulty=1,
                variant_id="structured-pairwise-rank-v1",
                mode="pairwise-rank",
                stage="tiny",
                holdout_prompt_hashes={blocked_hash},
                global_seen_prompt_hashes=set(),
                context_length=128,
                plan_sha256="a" * 64,
            )
        self.assertEqual(metrics["attempts"], 2)
        source_hash = records[0]["provenance"]["source_prompt_sha256"]
        self.assertNotEqual(source_hash, blocked_hash)

    def test_screen_plan_has_zero_promotion_authority_and_full_uses_immutable_gate(self):
        suite = Path("evals/m6-domain-selection-v2.json")
        tiny = build_plan(
            incumbent_sha256="a" * 64,
            suite_path=suite,
            variant_id="structured-prefix-next-v1",
            stage="tiny",
        )
        full = build_plan(
            incumbent_sha256="a" * 64,
            suite_path=suite,
            variant_id="structured-prefix-next-v1",
            stage="full",
        )
        self.assertFalse(tiny["screening_contract"]["promotion_authority"])
        self.assertFalse(tiny["screening_contract"]["promotion_eligible"])
        self.assertEqual(full["screening_contract"]["promotion_authority"], "immutable-gate-only")
        self.assertTrue(full["screening_contract"]["promotion_eligible"])

    def test_development_suite_is_prompt_disjoint_from_all_frozen_ladder_suites(self):
        _, dev_hashes, _ = frozen_holdouts("evals/m6-weak-domain-dev-v1.json")
        for path in [
            "evals/m6-domain-selection-v2.json",
            "evals/m6-domain-ladder-d2-v1.json",
            "evals/m6-domain-ladder-d3-v1.json",
            "evals/m6-domain-ladder-d4-v1.json",
            "evals/m6-domain-ladder-d5-v1.json",
        ]:
            _, frozen_hashes, _ = frozen_holdouts(path)
            self.assertFalse(dev_hashes & frozen_hashes, path)


if __name__ == "__main__":
    unittest.main()
