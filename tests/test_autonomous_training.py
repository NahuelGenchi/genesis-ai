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


def fixture(focus_domain: str, continuation_quotas: dict[str, int], *, replay_examples: int):
    domains = ("code", "math", "structured")
    records = []
    ordinals = {domain: [] for domain in domains}
    for domain in domains:
        count = 4096 if domain == focus_domain else replay_examples
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
    def test_three_million_budget_uses_plan_derived_anchors_and_unique_replay(self):
        # 3M -> 18,752 procedural target updates. With 4,096 focus + 1,024 replay
        # records/domain, 12,288 anchors are mandatory and 6,464 continuations remain.
        expected_continuation = {"math": 4526, "code": 969, "structured": 969}
        dataset, records = fixture("math", {"math": 6000, "code": 1692, "structured": 1692}, replay_examples=1024)
        selected, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="math",
            total_samples=18_752,
            seed=123,
        )
        self.assertEqual(len(selected), 18_752)
        self.assertEqual(len(torch.unique(selected)), 18_752)
        self.assertEqual(accounting["record_count_by_domain"], {"code": 1024, "math": 4096, "structured": 1024})
        self.assertEqual(accounting["total_anchor_updates"], 12_288)
        self.assertEqual(accounting["anchor_updates_by_domain"], {"code": 2048, "math": 8192, "structured": 2048})
        self.assertEqual(accounting["continuation_updates_by_domain"], expected_continuation)
        self.assertEqual(accounting["total_updates_by_domain"], {"code": 3017, "math": 12718, "structured": 3017})

    def test_two_million_budget_uses_smaller_replay_pool(self):
        # 2M controller replay count is 768/domain. Mandatory anchors = 11,264;
        # 1,248 continuation updates remain.
        expected_continuation = {"code": 874, "math": 187, "structured": 187}
        dataset, records = fixture("code", {"code": 2000, "math": 500, "structured": 500}, replay_examples=768)
        _, accounting = build_focus_schedule(
            dataset,
            records,
            focus_domain="code",
            total_samples=12_512,
            seed=456,
        )
        self.assertEqual(accounting["total_anchor_updates"], 11_264)
        self.assertEqual(accounting["total_continuation_updates"], 1248)
        self.assertEqual(accounting["continuation_updates_by_domain"], expected_continuation)

    def test_old_512_record_three_million_shape_reproduces_failure(self):
        dataset, records = fixture("structured", {"structured": 7000, "code": 1600, "math": 846}, replay_examples=512)
        with self.assertRaisesRegex(ValueError, "insufficient unique math continuation contexts"):
            build_focus_schedule(dataset, records, focus_domain="structured", total_samples=18_752, seed=1)

    def test_insufficient_continuation_pool_still_fails_closed(self):
        dataset, records = fixture("math", {"math": 100, "code": 100, "structured": 100}, replay_examples=1024)
        with self.assertRaises(ValueError):
            build_focus_schedule(dataset, records, focus_domain="math", total_samples=18_752, seed=1)


if __name__ == "__main__":
    unittest.main()
