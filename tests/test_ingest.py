import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.ingest import IngestError, ingest_manifest


class IngestTest(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source.txt"
        source.write_text("alpha\n\nbeta\n\ngamma\n", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata = {
            "schema_version": "1.0",
            "id": "fixture-source",
            "name": "Fixture source",
            "origin": "local-test",
            "rights": {
                "basis": "test fixture",
                "training_allowed": True,
                "redistribution_allowed": False,
            },
            "retrieval": {
                "retrieved_at": "2026-08-14T00:00:00Z",
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            },
            "content": {
                "languages": ["en"],
                "domains": ["test"],
                "formats": ["text"],
            },
            "filtering_steps": [],
        }
        metadata_path = root / "source.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps([{"metadata": "source.json", "local_path": "source.txt"}]), encoding="utf-8")
        return manifest, metadata, metadata_path

    def test_deterministic_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _, _ = self._fixture(root)
            first = root / "first"
            second = root / "second"
            result = ingest_manifest(manifest, first, docs_per_shard=2)
            ingest_manifest(manifest, second, docs_per_shard=2)
            self.assertEqual(result["documents"], 3)
            self.assertEqual(len(result["shards"]), 2)
            for name in ("shard-00000.jsonl", "shard-00001.jsonl", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_rejects_unapproved_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, metadata, metadata_path = self._fixture(root)
            metadata["rights"]["training_allowed"] = False
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IngestError, "training_allowed"):
                ingest_manifest(manifest, root / "out")

    def test_rejects_bad_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, metadata, metadata_path = self._fixture(root)
            metadata["retrieval"]["sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IngestError, "checksum mismatch"):
                ingest_manifest(manifest, root / "out")

    def test_rejects_invalid_manifest_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"metadata": "source.json"}]), encoding="utf-8")
            with self.assertRaisesRegex(IngestError, "metadata and local_path"):
                ingest_manifest(manifest, root / "out")


if __name__ == "__main__":
    unittest.main()
