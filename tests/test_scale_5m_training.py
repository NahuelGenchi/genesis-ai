import unittest

import torch

from genesis_ai.scale_5m_training import build_balanced_unique_schedule


DOMAINS = ("code", "math", "structured")


class FakeDataset:
    def __init__(self, anchors, continuation, record_ordinals):
        self.anchor_indices = tuple(anchors)
        self.continuation_indices = tuple(continuation)
        self.record_ordinals = tuple(record_ordinals)

    def __len__(self):
        return len(self.record_ordinals)


def fixture(examples_per_domain: int, continuation_available: dict[str, int]):
    records = []
    record_ordinals_by_domain = {domain: [] for domain in DOMAINS}
    for domain in DOMAINS:
        for ordinal in range(examples_per_domain):
            record_ordinals_by_domain[domain].append(len(records))
            records.append({"domain": domain, "ordinal": ordinal})
    record_ordinals = []
    anchors = []
    continuation = []
    for record_ordinal in range(len(records)):
        for _ in range(2):
            anchors.append(len(record_ordinals))
            record_ordinals.append(record_ordinal)
    for domain in DOMAINS:
        pool = record_ordinals_by_domain[domain]
        for offset in range(continuation_available[domain]):
            continuation.append(len(record_ordinals))
            record_ordinals.append(pool[offset % len(pool)])
    return FakeDataset(anchors, continuation, record_ordinals), records


class Scale5MTrainingTest(unittest.TestCase):
    def test_20m_40_percent_procedural_schedule_is_balanced_and_unique(self):
        # 20M target rounds to 19,535 x 1,024-token steps.
        # 40% = 7,814 procedural steps x batch 8 = 62,512 target updates.
        # 8,192 records/domain require 49,152 mandatory first/stop anchors.
        # Remaining 13,360 continuations distribute 4,454 / 4,453 / 4,453.
        available = {"code": 5_000, "math": 5_000, "structured": 5_000}
        dataset, records = fixture(8_192, available)
        selected, accounting = build_balanced_unique_schedule(
            dataset,
            records,
            total_samples=62_512,
            seed=211011,
            examples_per_domain=8_192,
        )
        self.assertEqual(len(selected), 62_512)
        self.assertEqual(len(torch.unique(selected)), 62_512)
        self.assertEqual(accounting["total_anchor_updates"], 49_152)
        self.assertEqual(
            accounting["anchor_updates_by_domain"],
            {"code": 16_384, "math": 16_384, "structured": 16_384},
        )
        self.assertEqual(
            accounting["continuation_updates_by_domain"],
            {"code": 4_454, "math": 4_453, "structured": 4_453},
        )
        totals = accounting["total_updates_by_domain"]
        self.assertLessEqual(max(totals.values()) - min(totals.values()), 1)

    def test_short_domain_continuation_pool_fails_closed(self):
        dataset, records = fixture(
            8_192,
            {"code": 5_000, "math": 4_000, "structured": 5_000},
        )
        with self.assertRaises(ValueError):
            build_balanced_unique_schedule(
                dataset,
                records,
                total_samples=62_512,
                seed=211011,
                examples_per_domain=8_192,
            )


if __name__ == "__main__":
    unittest.main()
