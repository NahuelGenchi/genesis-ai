import unittest

import torch

from genesis_ai.multidomain_training import build_domain_aware_schedule
from genesis_ai.terminated_training import TerminatedGenerationAlignedDataset
from genesis_ai.tokenizer import ByteBPETokenizer


class MultidomainTrainingTest(unittest.TestCase):
    def test_schedule_is_deterministic_unique_and_covers_all_anchors(self):
        tokenizer = ByteBPETokenizer(())
        records = [
            {"id": "c", "domain": "code", "prompt": "code", "response": "abcdefgh"},
            {"id": "m", "domain": "math", "prompt": "math", "response": "12345"},
            {"id": "s", "domain": "structured", "prompt": "structured", "response": "[5,4,3,2,1]"},
        ]
        dataset = TerminatedGenerationAlignedDataset(records, tokenizer, 32)
        total = min(len(dataset), len(dataset.anchor_indices) + 12)
        first, allocations_first = build_domain_aware_schedule(dataset, records, total_samples=total, seed=99)
        second, allocations_second = build_domain_aware_schedule(dataset, records, total_samples=total, seed=99)
        selected = set(int(value) for value in first.tolist())
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(allocations_first, allocations_second)
        self.assertEqual(len(selected), len(first))
        self.assertTrue(set(dataset.first_target_indices).issubset(selected))
        self.assertTrue(set(dataset.terminator_target_indices).issubset(selected))

    def test_schedule_rejects_budget_below_anchor_coverage(self):
        tokenizer = ByteBPETokenizer(())
        records = [
            {"id": "c", "domain": "code", "prompt": "code", "response": "abc"},
            {"id": "m", "domain": "math", "prompt": "math", "response": "123"},
            {"id": "s", "domain": "structured", "prompt": "structured", "response": "[1]"},
        ]
        dataset = TerminatedGenerationAlignedDataset(records, tokenizer, 16)
        with self.assertRaises(ValueError):
            build_domain_aware_schedule(dataset, records, total_samples=len(dataset.anchor_indices) - 1, seed=1)


if __name__ == "__main__":
    unittest.main()
