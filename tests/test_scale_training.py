import json
import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.checkpoint import export_inference_checkpoint, save_checkpoint
from genesis_ai.config import ModelConfig
from genesis_ai.ingest import sha256_file
from genesis_ai.model import GenesisLM
from genesis_ai.scale_repro import compare_checkpoints
from genesis_ai.scale_training import BASE_LR, MIN_LR, _lr, validate_training_inputs
from genesis_ai.tokenizer import ByteBPETokenizer


class ScaleTrainingTest(unittest.TestCase):
    def _records_and_lock(self, root: Path) -> tuple[Path, Path, Path]:
        records = root / "records.jsonl"
        values = [
            {
                "id": "a",
                "curriculum": "m6-code-curriculum-v1",
                "prompt": "Write only an expression for 2*x + 3*y + 1.",
                "response": "2*x + 3*y + 1",
                "provenance": {"kind": "procedural_oracle"},
            },
            {
                "id": "b",
                "curriculum": "m6-code-curriculum-v1",
                "prompt": "Write only an expression for -4*x + 5*y - 2.",
                "response": "-4*x + 5*y - 2",
                "provenance": {"kind": "procedural_oracle"},
            },
        ]
        with records.open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(value, sort_keys=True) + "\n")

        public = root / "public"
        public.mkdir()
        manifest = public / "manifest.json"
        manifest.write_text(json.dumps({"format_version": "1.0", "documents": 1, "shards": []}), encoding="utf-8")

        lock = json.loads(Path("research/m6-code-curriculum-v1.json").read_text(encoding="utf-8"))
        lock["training"]["procedural"]["examples"] = len(values)
        lock["training"]["procedural"]["records_file_sha256"] = sha256_file(records)
        lock["public_text"]["manifest_sha256"] = sha256_file(manifest)
        lock_path = root / "lock.json"
        lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
        return records, lock_path, public

    def test_authorized_inputs_construct_exact_micro_2m(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, lock, public = self._records_and_lock(root)
            config, tokenizer, loaded_records, policy = validate_training_inputs(
                ladder_definition_path="experiments/m6-scaling-ladder-v1.json",
                ladder_result_path="research/m6-scaling-ladder-v1.json",
                curriculum_lock_path=lock,
                records_path=records,
                public_data=public,
                tokenizer_path="tokenizers/genesis-v0.json",
            )
            self.assertEqual(GenesisLM(config).parameter_count(), 1_895_808)
            self.assertEqual(tokenizer.vocab_size, 512)
            self.assertEqual(len(loaded_records), 2)
            self.assertEqual(policy["target_training_tokens"], 2_000_000)
            self.assertEqual(policy["target_tokens_per_step"], 1024)
            self.assertEqual(policy["procedural_fraction"], 0.8)
            self.assertEqual(policy["public_fraction"], 0.2)

            records.write_text(records.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_training_inputs(
                    ladder_definition_path="experiments/m6-scaling-ladder-v1.json",
                    ladder_result_path="research/m6-scaling-ladder-v1.json",
                    curriculum_lock_path=lock,
                    records_path=records,
                    public_data=public,
                    tokenizer_path="tokenizers/genesis-v0.json",
                )

    def test_learning_rate_schedule_is_bounded(self):
        self.assertGreater(_lr(1, 1955), 0.0)
        self.assertLess(_lr(1, 1955), BASE_LR)
        self.assertAlmostEqual(_lr(50, 1955), BASE_LR)
        self.assertAlmostEqual(_lr(1955, 1955), MIN_LR)

    def _inference_checkpoint(self, root: Path, name: str, seed: int) -> Path:
        torch.manual_seed(seed)
        tokenizer = ByteBPETokenizer(())
        model = GenesisLM(ModelConfig(vocab_size=tokenizer.vocab_size, context_length=8, d_model=8, n_heads=2, n_layers=1, d_ff=16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        training = root / f"{name}-train.pt"
        inference = root / f"{name}.pt"
        save_checkpoint(training, model=model, optimizer=optimizer, step=3, metadata={"stable": True}, tokenizer=tokenizer, batch_generator=generator)
        export_inference_checkpoint(training, inference)
        return inference

    def test_semantic_reproducibility_detects_equal_and_different_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._inference_checkpoint(root, "first", 7)
            same = self._inference_checkpoint(root, "same", 7)
            different = self._inference_checkpoint(root, "different", 8)
            self.assertTrue(compare_checkpoints(first, same)["reproducible"])
            self.assertFalse(compare_checkpoints(first, different)["reproducible"])


if __name__ == "__main__":
    unittest.main()
