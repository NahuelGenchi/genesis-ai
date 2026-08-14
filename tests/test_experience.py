import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.experience import collect_experience, load_tasks, write_experience_bundle


PRODUCER = {
    "kind": "genesis_checkpoint",
    "checkpoint_sha256": "a" * 64,
    "checkpoint_step": 160,
    "generation": {"top_k": 1, "base_seed": 7},
}


class ExperiencePipelineTest(unittest.TestCase):
    def test_only_verified_genesis_outputs_enter_training_data(self):
        tasks = [
            {
                "id": "task-good",
                "domain": "math",
                "difficulty": 1,
                "generator": "procedural-v1",
                "prompt": "Return 4.",
                "provenance": {"kind": "procedural"},
                "verifier": {"kind": "integer_exact", "version": "deterministic-v1", "expected": 4},
            },
            {
                "id": "task-bad",
                "domain": "math",
                "difficulty": 1,
                "generator": "procedural-v1",
                "prompt": "Return 9.",
                "provenance": {"kind": "procedural"},
                "verifier": {"kind": "integer_exact", "version": "deterministic-v1", "expected": 9},
            },
        ]
        answers = {"task-good": "4", "task-bad": "8"}
        accepted, audit = collect_experience(
            tasks,
            lambda task, ordinal: answers[task["id"]],
            producer_metadata=PRODUCER,
            min_score=1.0,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["response"], "4")
        self.assertNotIn("verifier", accepted[0])
        self.assertNotIn("expected", json.dumps(accepted[0], sort_keys=True))
        self.assertEqual([record["accepted"] for record in audit], [True, False])
        self.assertEqual(audit[1]["verification"]["reason"], "wrong_answer")

    def test_partial_score_threshold_is_configurable_and_reproducible(self):
        task = {
            "id": "task-code",
            "domain": "code",
            "difficulty": 2,
            "generator": "procedural-v1",
            "prompt": "expression",
            "provenance": {"kind": "procedural"},
            "verifier": {
                "kind": "restricted_expression",
                "version": "deterministic-v1",
                "tests": [
                    {"variables": {"x": 1, "y": 0}, "expected": 2},
                    {"variables": {"x": 2, "y": 0}, "expected": 4},
                ],
            },
        }
        producer = lambda task, ordinal: "2*x if x == 1 else 0"
        # Conditional syntax is intentionally unsafe, so this scores zero.
        accepted, audit = collect_experience([task], producer, producer_metadata=PRODUCER, min_score=0.0)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(audit[0]["verification"]["score"], 0.0)

        partial_task = {
            **task,
            "id": "task-partial",
            "verifier": {
                "kind": "restricted_expression",
                "version": "deterministic-v1",
                "tests": [
                    {"variables": {"x": 1, "y": 0}, "expected": 2},
                    {"variables": {"x": 2, "y": 0}, "expected": 5},
                ],
            },
        }
        half, audit_half = collect_experience(
            [partial_task], lambda task, ordinal: "2*x", producer_metadata=PRODUCER, min_score=0.5
        )
        strict, _ = collect_experience(
            [partial_task], lambda task, ordinal: "2*x", producer_metadata=PRODUCER, min_score=1.0
        )
        self.assertEqual(audit_half[0]["verification"]["score"], 0.5)
        self.assertEqual(len(half), 1)
        self.assertEqual(strict, [])

    def test_rejects_non_genesis_producer(self):
        task = {
            "id": "task-x",
            "prompt": "x",
            "verifier": {"kind": "integer_exact", "version": "deterministic-v1", "expected": 1},
        }
        with self.assertRaises(ValueError):
            collect_experience(
                [task],
                lambda task, ordinal: "1",
                producer_metadata={"kind": "external_api"},
            )

    def test_bundle_hashes_and_rejections_are_auditable(self):
        task = {
            "id": "task-x",
            "domain": "math",
            "difficulty": 1,
            "generator": "procedural-v1",
            "prompt": "Return 1.",
            "provenance": {"kind": "procedural"},
            "verifier": {"kind": "integer_exact", "version": "deterministic-v1", "expected": 1},
        }
        accepted, audit = collect_experience(
            [task], lambda task, ordinal: "0", producer_metadata=PRODUCER, min_score=1.0
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_experience_bundle(
                Path(directory), accepted, audit, producer_metadata=PRODUCER, min_score=1.0
            )
            self.assertEqual(manifest["attempted"], 1)
            self.assertEqual(manifest["accepted"], 0)
            self.assertEqual(manifest["rejected"], 1)
            self.assertEqual(len(manifest["files"]["accepted.jsonl"]), 64)
            audit_record = json.loads((Path(directory) / "audit.jsonl").read_text().strip())
            self.assertFalse(audit_record["accepted"])
            self.assertEqual(audit_record["task"]["id"], "task-x")
            self.assertEqual(audit_record["producer"]["checkpoint_step"], 160)

    def test_task_loader_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            record = {"id": "same", "prompt": "x"}
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                load_tasks(path)


if __name__ == "__main__":
    unittest.main()
