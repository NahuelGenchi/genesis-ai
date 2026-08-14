import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.eval_lab import check_exact_contamination, compare_results, document_split, load_suite


def _write_filtered(root: Path, documents: list[dict]) -> Path:
    data = root / "filtered"
    data.mkdir()
    shard = data / "shard-00000.jsonl"
    payload = b"".join(
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for document in documents
    )
    shard.write_bytes(payload)
    manifest = {
        "documents": len(documents),
        "shards": [{"file": shard.name, "documents": len(documents), "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}],
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data


class EvaluationLabTest(unittest.TestCase):
    def test_suite_definition_is_versioned(self):
        suite = load_suite(Path("evals/m3-v1.json"))
        self.assertEqual(suite["suite_version"], "m3-v1")

    def test_contamination_detects_cross_split_exact_text(self):
        train_id = next(f"train-{i}" for i in range(10000) if document_split(f"train-{i}", 0.1) == "train")
        validation_id = next(f"validation-{i}" for i in range(10000) if document_split(f"validation-{i}", 0.1) == "validation")
        duplicate = "The exact same normalized text appears in both evaluation partitions."
        with tempfile.TemporaryDirectory() as tmp:
            data = _write_filtered(Path(tmp), [{"id": train_id, "text": duplicate}, {"id": validation_id, "text": duplicate}])
            result = check_exact_contamination(data, 0.1)
            self.assertTrue(result["blocking"])
            self.assertEqual(result["exact_overlap_count"], 1)

    def test_comparison_rejects_mismatched_suite(self):
        incumbent = {"suite_version": "v1", "suite_sha256": "a", "data_manifest_sha256": "d", "primary_metric": {"name": "validation_loss", "value": 4.0, "lower_is_better": True}}
        candidate = {"suite_version": "v2", "suite_sha256": "b", "data_manifest_sha256": "d", "primary_metric": {"name": "validation_loss", "value": 3.0, "lower_is_better": True}}
        with self.assertRaisesRegex(ValueError, "suite_version"):
            compare_results(incumbent, candidate)

    def test_comparison_selects_lower_validation_loss(self):
        incumbent = {"suite_version": "v1", "suite_sha256": "a", "data_manifest_sha256": "d", "primary_metric": {"name": "validation_loss", "value": 4.0, "lower_is_better": True}}
        candidate = {"suite_version": "v1", "suite_sha256": "a", "data_manifest_sha256": "d", "primary_metric": {"name": "validation_loss", "value": 3.5, "lower_is_better": True}}
        result = compare_results(incumbent, candidate)
        self.assertEqual(result["winner"], "candidate")
        self.assertEqual(result["delta"], -0.5)


if __name__ == "__main__":
    unittest.main()
