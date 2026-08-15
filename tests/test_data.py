import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.data import TokenDataset, sample_batch
from genesis_ai.tokenizer import ByteBPETokenizer


def _filtered_fixture(root: Path, count: int = 100) -> Path:
    input_dir = root / "filtered"
    input_dir.mkdir()
    documents = [
        {"id": f"doc-{index:04d}", "text": f"Document {index} contains enough deterministic text for token dataset testing."}
        for index in range(count)
    ]
    shard = input_dir / "shard-00000.jsonl"
    payload = b"".join(
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for document in documents
    )
    shard.write_bytes(payload)
    manifest = {
        "format_version": "1.0",
        "documents": count,
        "shards": [{
            "file": shard.name,
            "documents": count,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }],
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return input_dir


class TokenDatasetTest(unittest.TestCase):
    def test_train_validation_split_is_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _filtered_fixture(root)
            tokenizer = ByteBPETokenizer(())
            train = TokenDataset(data, tokenizer, 16, split="train")
            validation = TokenDataset(data, tokenizer, 16, split="validation")
            train_again = TokenDataset(data, tokenizer, 16, split="train")
            self.assertTrue(train.document_ids)
            self.assertTrue(validation.document_ids)
            self.assertFalse(set(train.document_ids) & set(validation.document_ids))
            self.assertEqual(train.document_ids, train_again.document_ids)
            self.assertTrue(torch.equal(train.tokens, train_again.tokens))

    def test_seeded_batch_sampling_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = TokenDataset(_filtered_fixture(root), ByteBPETokenizer(()), 8, split="train")
            first = torch.Generator().manual_seed(123)
            second = torch.Generator().manual_seed(123)
            x1, y1 = sample_batch(dataset, 4, first)
            x2, y2 = sample_batch(dataset, 4, second)
            self.assertTrue(torch.equal(x1, x2))
            self.assertTrue(torch.equal(y1, y2))

    def test_min_chars_uses_stripped_length_like_cpu_farm_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _filtered_fixture(root, count=2)
            shard = data / "shard-00000.jsonl"
            documents = [
                {"id": "short-padded", "text": "short" + " " * 200},
                {"id": "long", "text": "L" * 120},
            ]
            payload = b"".join(
                (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                for document in documents
            )
            shard.write_bytes(payload)
            manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            manifest["documents"] = 2
            manifest["shards"][0].update({"documents": 2, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)})
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            dataset = TokenDataset(data, ByteBPETokenizer(()), 8, split="all", min_chars=80)
            self.assertEqual(dataset.document_ids, ("long",))
            self.assertEqual(dataset.min_chars, 80)
            self.assertTrue((data / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
