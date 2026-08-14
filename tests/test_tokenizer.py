import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.tokenizer import ByteBPETokenizer, save_tokenizer, train_byte_bpe


def _filtered_fixture(root: Path) -> Path:
    input_dir = root / "filtered"
    input_dir.mkdir()
    documents = [
        {"id": "1", "text": "the quick brown fox jumps over the quick brown dog"},
        {"id": "2", "text": "the quick brown fox learns quickly and the dog learns too"},
        {"id": "3", "text": "mañana café — こんにちは — مرحبا — 😀"},
    ]
    shard = input_dir / "shard-00000.jsonl"
    payload = b"".join(
        (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
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
    (input_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    return input_dir


class TokenizerTest(unittest.TestCase):
    def test_multilingual_round_trip(self):
        tokenizer = ByteBPETokenizer(((116, 104, 256), (256, 101, 257)))
        text = "the café — 日本語 — مرحبا — 😀\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_training_is_deterministic_and_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _filtered_fixture(root)
            first, first_metrics = train_byte_bpe(input_dir, vocab_size=300, min_pair_count=2)
            second, second_metrics = train_byte_bpe(input_dir, vocab_size=300, min_pair_count=2)
            self.assertEqual(first.merges, second.merges)
            self.assertEqual(first_metrics, second_metrics)
            self.assertGreater(first_metrics["bytes_per_token"], 1.0)
            self.assertGreater(first_metrics["token_reduction_vs_bytes"], 0.0)
            self.assertEqual(first_metrics["round_trip_failures"], 0)

            first_path = root / "first.json"
            second_path = root / "second.json"
            save_tokenizer(first, first_path, first_metrics)
            save_tokenizer(second, second_path, second_metrics)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            loaded = ByteBPETokenizer.load(first_path)
            sample = "mañana café — こんにちは — 😀"
            self.assertEqual(loaded.decode(loaded.encode(sample)), sample)

    def test_trainer_rejects_tampered_filtered_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = _filtered_fixture(root)
            with (input_dir / "shard-00000.jsonl").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                train_byte_bpe(input_dir, vocab_size=280)


if __name__ == "__main__":
    unittest.main()
