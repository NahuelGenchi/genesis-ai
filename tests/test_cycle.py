import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.cycle import build_candidate_record, build_no_candidate_record
from genesis_ai.ingest import sha256_file


class FirstCycleRecordTest(unittest.TestCase):
    def _bundle(self, root: Path, parent: Path, *, attempted: int, accepted: int) -> tuple[Path, Path]:
        tasks = root / "tasks.jsonl"
        tasks.write_text("".join(json.dumps({"id": f"task-{index}"}) + "\n" for index in range(attempted)), encoding="utf-8")
        bundle = root / "experience"
        bundle.mkdir()
        accepted_path = bundle / "accepted.jsonl"
        audit_path = bundle / "audit.jsonl"
        accepted_path.write_text(
            "".join(json.dumps({"id": f"exp-{index}"}) + "\n" for index in range(accepted)),
            encoding="utf-8",
        )
        audit_path.write_text(
            "".join(
                json.dumps({"id": f"exp-{index}", "accepted": index < accepted}) + "\n"
                for index in range(attempted)
            ),
            encoding="utf-8",
        )
        manifest = {
            "format_version": "1.0",
            "policy": {"version": "verified-experience-v1", "min_score": 1.0},
            "producer": {
                "kind": "genesis_checkpoint",
                "checkpoint_sha256": sha256_file(parent),
                "checkpoint_step": 160,
            },
            "attempted": attempted,
            "accepted": accepted,
            "rejected": attempted - accepted,
            "acceptance_rate": accepted / attempted,
            "files": {
                "accepted.jsonl": sha256_file(accepted_path),
                "audit.jsonl": sha256_file(audit_path),
            },
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return tasks, bundle

    def _common(self, parent: Path, tasks: Path, bundle: Path) -> dict:
        return {
            "parent_checkpoint": parent,
            "tasks_path": tasks,
            "experience_dir": bundle,
            "task_count": 6,
            "min_accepted": 2,
            "challenge_seed": 20260814,
            "generation_seed": 9107,
            "source_commit": "a" * 40,
            "workflow_run_id": "12345",
        }

    def test_no_candidate_is_fail_closed_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pt"
            parent.write_bytes(b"parent")
            tasks, bundle = self._bundle(root, parent, attempted=6, accepted=1)
            first = build_no_candidate_record(**self._common(parent, tasks, bundle))
            second = build_no_candidate_record(**self._common(parent, tasks, bundle))
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "no_candidate")
            self.assertFalse(first["candidate_trained"])
            self.assertFalse(first["promotion_attempted"])
            self.assertEqual(first["experience"]["accepted"], 1)
            self.assertEqual(first["reason"], "insufficient_verified_experience")

    def test_no_candidate_rejects_when_threshold_is_met(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pt"
            parent.write_bytes(b"parent")
            tasks, bundle = self._bundle(root, parent, attempted=6, accepted=2)
            with self.assertRaises(ValueError):
                build_no_candidate_record(**self._common(parent, tasks, bundle))

    def test_candidate_record_binds_measurements_and_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pt"
            candidate = root / "candidate.pt"
            parent.write_bytes(b"parent")
            candidate.write_bytes(b"candidate")
            tasks, bundle = self._bundle(root, parent, attempted=6, accepted=2)
            measurements = {}
            for name in ("parent_eval", "candidate_eval", "parent_bench", "candidate_bench"):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"name": name}), encoding="utf-8")
                measurements[name] = path
            promotion = root / "promotion.json"
            promotion.write_text(
                json.dumps(
                    {
                        "parent_checkpoint_sha256": sha256_file(parent),
                        "candidate_checkpoint_sha256": sha256_file(candidate),
                        "promoted": False,
                        "decision_sha256": "d" * 64,
                    }
                ),
                encoding="utf-8",
            )
            record = build_candidate_record(
                **self._common(parent, tasks, bundle),
                candidate_checkpoint=candidate,
                parent_evaluation=measurements["parent_eval"],
                candidate_evaluation=measurements["candidate_eval"],
                parent_benchmark=measurements["parent_bench"],
                candidate_benchmark=measurements["candidate_bench"],
                promotion=promotion,
            )
            self.assertEqual(record["status"], "candidate_rejected")
            self.assertTrue(record["candidate_trained"])
            self.assertTrue(record["promotion_attempted"])
            self.assertFalse(record["promoted"])
            self.assertEqual(record["promotion_decision_sha256"], "d" * 64)
            self.assertEqual(record["measurements"]["parent_evaluation_sha256"], sha256_file(measurements["parent_eval"]))

    def test_tampered_experience_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pt"
            parent.write_bytes(b"parent")
            tasks, bundle = self._bundle(root, parent, attempted=6, accepted=1)
            with (bundle / "audit.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                build_no_candidate_record(**self._common(parent, tasks, bundle))


if __name__ == "__main__":
    unittest.main()
