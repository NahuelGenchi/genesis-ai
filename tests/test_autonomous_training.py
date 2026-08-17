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


def fixture(focus_domain: str, continuation_quotas: dict[str, int], *, replay_records: int, focus_records: int = 4096):
    domains = ("code", "math", "structured")
    records = []
    ordinals = {domain: [] for domain in domains}
    for domain in domains:
        count = focus_records if domain == focus_domain else replay_records
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
    def test_three_million_budget_derives_larger_replay_anchor_coverage(self):
        # 3M -> 18,752 procedural target updates.
        # 4,096 focus + 1,024 replay/domain -> 12,288 mandatory anchors.
        # Remaining 6,464 -> 70/15/15 = 4,526 / 969 / 969.
        expected_continuation = {"math": 4526, "code": 969, "structured": 969}
        dataset, records = fixture("math", expected_continuation, replay_records=1024)
        selected, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="math",
            total_samples=18_752,
            seed=123,
        )
        self.assertEqual(len(selected), 18_752)
        self.assertEqual(len(torch.unique(selected)), 18_752)
        self.assertEqual(accounting["record_counts_by_domain"], {"code": 1024, "math": 4096, "structured": 1024})
        self.assertEqual(accounting["total_anchor_updates"], 12_288)
        self.assertEqual(accounting["anchor_updates_by_domain"], {"code": 2048, "math": 8192, "structured": 2048})
        self.assertEqual(accounting["continuation_updates_by_domain"], expected_continuation)
        self.assertEqual(accounting["total_updates_by_domain"], {"code": 3017, "math": 12718, "structured": 3017})

    def test_two_million_budget_keeps_original_minimum_replay_anchors(self):
        # 2M -> 12,512 updates; 512 replay records/domain -> 10,240 anchors.
        expected_continuation = {"code": 1592, "math": 340, "structured": 340}
        dataset, records = fixture("code", expected_continuation, replay_records=512)
        _, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="code",
            total_samples=12_512,
            seed=456,
        )
        self.assertEqual(accounting["total_anchor_updates"], 10_240)
        self.assertEqual(accounting["total_continuation_updates"], 2272)
        self.assertEqual(accounting["continuation_updates_by_domain"], expected_continuation)

    def test_short_math_pool_regression_has_capacity_after_adaptive_replay(self):
        available = {"structured": 4526, "code": 1500, "math": 1000}
        dataset, records = fixture("structured", available, replay_records=1024)
        _, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="structured",
            total_samples=18_752,
            seed=789,
        )
        self.assertEqual(accounting["continuation_updates_by_domain"]["math"], 969)
        self.assertEqual(accounting["continuation_available_by_domain"]["math"], 1000)

    def test_research_strategy_can_trade_breadth_for_sequence_depth_and_code_protection(self):
        # 1,024 focus + 1,024 replay/domain -> 6,144 mandatory anchors.
        # Remaining 12,608 uses the sequence-depth policy 65% structured,
        # 25% code protection, 10% math replay.
        expected = {"structured": 8196, "code": 3152, "math": 1260}
        dataset, records = fixture(
            "structured",
            expected,
            replay_records=1024,
            focus_records=1024,
        )
        selected, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="structured",
            total_samples=18_752,
            seed=101,
            continuation_weights={"structured": 0.65, "code": 0.25, "math": 0.10},
        )
        self.assertEqual(len(selected), 18_752)
        self.assertEqual(accounting["total_anchor_updates"], 6144)
        self.assertEqual(accounting["record_counts_by_domain"], {"code": 1024, "math": 1024, "structured": 1024})
        self.assertEqual(accounting["continuation_updates_by_domain"], expected)
        self.assertEqual(accounting["continuation_weights"], {"structured": 0.65, "code": 0.25, "math": 0.10})

    def test_insufficient_continuation_pool_still_fails_before_training(self):
        dataset, records = fixture(
            "math",
            {"math": 100, "code": 100, "structured": 100},
            replay_records=1024,
        )
        with self.assertRaises(ValueError):
            build_focus_schedule(dataset, records, focus_domain="math", total_samples=18_752, seed=1)


if __name__ == "__main__":
    unittest.main()
