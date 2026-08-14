import unittest

import torch

from genesis_ai.autonomous_training import build_focus_schedule


class FakeDataset:
    def __init__(self, anchors, continuation, record_ordinals):
        self.anchor_indices = tuple(anchors)
        self.continuation_indices = tuple(continuation)
        self.record_ordinals = tuple(record_ordinals)

    def __len__(self):
        return len(self.record_ordinals)


def fixture(focus_domain: str, continuation_quotas: dict[str, int]):
    domains = ("code", "math", "structured")
    records = []
    ordinals = {domain: [] for domain in domains}
    for domain in domains:
        count = 4096 if domain == focus_domain else 512
        role = "focus" if domain == focus_domain else "replay"
        for _ in range(count):
            ordinals[domain].append(len(records))
            records.append({"domain": domain, "role": role})

    record_ordinals = []
    anchors = []
    continuation = []
    for ordinal in range(len(records)):
        for _ in range(2):
            anchors.append(len(record_ordinals))
            record_ordinals.append(ordinal)
    for domain in domains:
        pool = ordinals[domain]
        for offset in range(continuation_quotas[domain]):
            continuation.append(len(record_ordinals))
            record_ordinals.append(pool[offset % len(pool)])
    return FakeDataset(anchors, continuation, record_ordinals), records


class AutonomousTrainingTest(unittest.TestCase):
    def test_three_million_budget_preserves_full_anchor_coverage_and_focuses_continuations(self):
        # 3M -> 2930 steps -> 2344 procedural steps -> 18,752 target updates.
        # After 10,240 mandatory anchors: 8,512 continuation updates.
        expected_continuation = {"math": 5960, "code": 1276, "structured": 1276}
        dataset, records = fixture("math", expected_continuation)
        selected, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="math",
            total_samples=18_752,
            seed=123,
        )
        self.assertEqual(len(selected), 18_752)
        self.assertEqual(len(torch.unique(selected)), 18_752)
        self.assertEqual(accounting["total_anchor_updates"], 10_240)
        self.assertEqual(accounting["anchor_updates_by_domain"], {"code": 1024, "math": 8192, "structured": 1024})
        self.assertEqual(accounting["continuation_updates_by_domain"], expected_continuation)
        self.assertEqual(accounting["total_updates_by_domain"], {"code": 2300, "math": 14152, "structured": 2300})

    def test_two_million_budget_still_has_continuation_capacity(self):
        # 2M -> 1955 steps -> 1564 procedural steps -> 12,512 updates.
        expected_continuation = {"code": 1592, "math": 340, "structured": 340}
        dataset, records = fixture("code", expected_continuation)
        _, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="code",
            total_samples=12_512,
            seed=456,
        )
        self.assertEqual(accounting["total_continuation_updates"], 2272)
        self.assertEqual(accounting["continuation_updates_by_domain"], expected_continuation)

    def test_insufficient_continuation_pool_fails_before_training(self):
        dataset, records = fixture("math", {"math": 100, "code": 100, "structured": 100})
        with self.assertRaises(ValueError):
            build_focus_schedule(dataset, records, focus_domain="math", total_samples=18_752, seed=1)


if __name__ == "__main__":
    unittest.main()
