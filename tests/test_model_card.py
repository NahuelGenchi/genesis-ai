import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.model_card import build_model_record


class ModelCardTest(unittest.TestCase):
    def test_builds_auditable_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = {
                "config": {"vocab_size": 512, "context_length": 128, "d_model": 96, "n_heads": 4, "n_layers": 3, "d_ff": 384},
                "parameter_count": 400000,
                "steps": 160,
                "resumed_from_step": 80,
                "batch_size": 8,
                "learning_rate": 0.001,
                "train_documents": 90,
                "train_tokens": 10000,
                "probe_loss_before": 6.2,
                "probe_loss_after": 5.1,
                "probe_loss_decreased": True,
                "last_training_loss": 5.0,
                "elapsed_seconds": 10.0,
                "data_manifest_sha256": "a" * 64,
            }
            evaluation = {"split": "validation", "documents": 10, "tokens": 1000, "loss": 5.2, "perplexity": 181.0}
            tokenizer = {"vocab_size": 512, "training": {"bytes_per_token": 1.62, "corpus_source_count": 3}}
            lock = {"sources": [{"id": "source-a", "ebook_id": 1, "language": "en", "sample_sha256": "b" * 64}]}
            paths = {}
            for name, value in (("run", run), ("eval", evaluation), ("tokenizer", tokenizer), ("lock", lock)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[name] = path
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            sample = root / "sample.txt"
            sample.write_text("The sample", encoding="utf-8")

            record, markdown = build_model_record(
                run_path=paths["run"],
                evaluation_path=paths["eval"],
                checkpoint_path=checkpoint,
                tokenizer_path=paths["tokenizer"],
                source_lock_path=paths["lock"],
                sample_path=sample,
            )
            self.assertEqual(record["model"], "genesis-tiny-v0")
            self.assertTrue(record["training"]["probe_loss_decreased"])
            self.assertEqual(record["sources"][0]["id"], "source-a")
            self.assertIn("Pipeline baseline only", markdown)
            self.assertIn("The sample", markdown)
            self.assertEqual(len(record["checkpoint"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
