import json
import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.ingest import sha256_file
from genesis_ai.multidomain_curriculum_v2 import CURRICULUM_VERSION, EXAMPLES_PER_DOMAIN, TARGET_TRAINING_TOKENS
from genesis_ai.multidomain_gate_v2 import gate
from genesis_ai.multidomain_training_v2 import TRAINING_POLICY_VERSION, build_balanced_schedule


class FakeDataset:
    def __init__(self, anchors, continuation, record_ordinals):
        self.anchor_indices = tuple(anchors)
        self.continuation_indices = tuple(continuation)
        self.record_ordinals = tuple(record_ordinals)

    def __len__(self):
        return len(self.record_ordinals)


class MultidomainV2ScheduleTest(unittest.TestCase):
    def test_frozen_budget_matches_single_domain_capacity_per_skill(self):
        self.assertEqual(EXAMPLES_PER_DOMAIN, 4096)
        self.assertEqual(TARGET_TRAINING_TOKENS, 6_000_000)

        domains = ("code", "math", "structured")
        records = []
        for domain in domains:
            records.extend({"domain": domain} for _ in range(EXAMPLES_PER_DOMAIN))

        record_ordinals = []
        anchors = []
        continuation = []
        for ordinal in range(len(records)):
            for _ in range(2):
                anchors.append(len(record_ordinals))
                record_ordinals.append(ordinal)
        quotas = {"code": 4310, "math": 4309, "structured": 4309}
        offsets = {domain: 0 for domain in domains}
        for ordinal, record in enumerate(records):
            domain = record["domain"]
            if offsets[domain] < quotas[domain]:
                continuation.append(len(record_ordinals))
                record_ordinals.append(ordinal)
                offsets[domain] += 1
        dataset = FakeDataset(anchors, continuation, record_ordinals)
        selected, accounting = build_balanced_schedule(dataset, records, total_samples=37504, seed=123)
        self.assertEqual(len(selected), 37504)
        self.assertEqual(len(torch.unique(selected)), 37504)
        self.assertEqual(accounting["total_anchor_updates"], 24576)
        self.assertEqual(accounting["total_continuation_updates"], 12928)
        self.assertEqual(
            accounting["total_updates_by_domain"],
            {"code": 12502, "math": 12501, "structured": 12501},
        )


class MultidomainV2GateTest(unittest.TestCase):
    def _write(self, root: Path, name: str, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    @staticmethod
    def _domain_result(code: float, math: float, structured: float):
        return {
            "suite_version": "m6-domain-selection-v2",
            "suite_sha256": "suite",
            "domains": {
                "code": {"exact_accuracy": code},
                "math": {"exact_accuracy": math},
                "structured": {"exact_accuracy": structured},
            },
        }

    def test_exact_minimum_target_promotes_only_with_full_capacity_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.pt"
            candidate = root / "candidate.pt"
            baseline.write_bytes(b"baseline")
            candidate.write_bytes(b"candidate")
            training = self._write(root, "training.json", {
                "training_policy": TRAINING_POLICY_VERSION,
                "parent_checkpoint_sha256": sha256_file(baseline),
                "inference_checkpoint_sha256": sha256_file(candidate),
                "target_training_tokens": 6_000_000,
                "processed_tokens": 6_000_640,
                "procedural_updates": 37504,
                "schedule_unique_updates": 37504,
                "schedule_accounting": {"total_updates_by_domain": {"code": 12502, "math": 12501, "structured": 12501}},
                "cash_compute_cost_usd": 0.0,
            })
            reproduction = self._write(root, "repro.json", {"reproducible": True, "weights_equal": True})
            curriculum = self._write(root, "curriculum.json", {
                "curriculum_version": CURRICULUM_VERSION,
                "cash_compute_cost_usd": 0.0,
                "evaluation": {"exact_prompt_overlap_count": 0},
            })
            baseline_domain = self._write(root, "baseline-domain.json", self._domain_result(0.95, 0.0, 0.0))
            candidate_domain = self._write(root, "candidate-domain.json", self._domain_result(0.90, 0.50, 0.50))
            baseline_m3 = self._write(root, "baseline-m3.json", {"evaluation": {"loss": 3.35}, "contamination": {"blocking": False, "exact_overlap_count": 0}})
            candidate_m3 = self._write(root, "candidate-m3.json", {"evaluation": {"loss": 3.35}, "contamination": {"blocking": False, "exact_overlap_count": 0}})
            result = gate(
                baseline_checkpoint=baseline,
                candidate_checkpoint=candidate,
                training_path=training,
                reproduction_path=reproduction,
                curriculum_path=curriculum,
                baseline_domain_path=baseline_domain,
                candidate_domain_path=candidate_domain,
                baseline_m3_path=baseline_m3,
                candidate_m3_path=candidate_m3,
            )
            self.assertTrue(result["promoted"])
            self.assertAlmostEqual(result["gci_v1"]["relative_percent_change"], 100.0)


if __name__ == "__main__":
    unittest.main()
