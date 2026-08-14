import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.filtering import exact_fingerprint, filter_corpus, quality_reason


def _write_input(root: Path, documents: list[dict]) -> Path:
    input_dir = root / "input"
    input_dir.mkdir()
    shard = input_dir / "shard-00000.jsonl"
    payload = b"".join(
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for document in documents
    )
    shard.write_bytes(payload)
    manifest = {
        "format_version": "1.0",
        "documents": len(documents),
        "shards": [{
            "file": shard.name,
            "documents": len(documents),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }],
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return input_dir


class FilteringTest(unittest.TestCase):
    def test_exact_fingerprint_normalizes_whitespace(self):
        self.assertEqual(exact_fingerprint("alpha   beta"), exact_fingerprint("alpha\nbeta"))

    def test_quality_filters(self):
        self.assertEqual(quality_reason("tiny", min_chars=10), "too_short")
        self.assertEqual(quality_reason("x" * 128, min_chars=1), "repeated_character_run")
        self.assertIsNone(quality_reason("This is a sufficiently long, normal training document.", min_chars=20))

    def test_dedup_metrics_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = [
                {"id": "a", "text": "This is a sufficiently long document about efficient model training."},
                {"id": "b", "text": "This   is a sufficiently long document about efficient model training."},
                {"id": "c", "text": "short"},
                {"id": "d", "text": "Another useful document with enough unique content to remain in the corpus."},
            ]
            input_dir = _write_input(root, documents)
            first = root / "first"
            second = root / "second"
            metrics = filter_corpus(input_dir, first, min_chars=20, docs_per_shard=1)
            filter_corpus(input_dir, second, min_chars=20, docs_per_shard=1)

            self.assertEqual(metrics["input_documents"], 4)
            self.assertEqual(metrics["kept_documents"], 2)
            self.assertEqual(metrics["drop_reasons"], {"exact_duplicate": 1, "too_short": 1})
            for name in ("shard-00000.jsonl", "shard-00001.jsonl", "manifest.json", "metrics.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
