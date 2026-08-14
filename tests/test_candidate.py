import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.candidate import ExperienceDataset, train_candidate, validate_experience_bundle
from genesis_ai.checkpoint import export_inference_checkpoint, load_model, save_checkpoint
from genesis_ai.config import ModelConfig
from genesis_ai.experience import collect_experience, write_experience_bundle
from genesis_ai.ingest import sha256_file
from genesis_ai.model import GenesisLM
from genesis_ai.tokenizer import ByteBPETokenizer


class CandidateTrainingTest(unittest.TestCase):
    def _make_parent(self, root: Path, *, seed: int = 1) -> Path:
        torch.manual_seed(seed)
        tokenizer = ByteBPETokenizer(())
        model = GenesisLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=16,
                d_model=16,
                n_heads=4,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        training = root / "parent-training.pt"
        parent = root / "parent.pt"
        save_checkpoint(
            training,
            model=model,
            optimizer=optimizer,
            step=3,
            metadata={"fixture": True},
            tokenizer=tokenizer,
            batch_generator=generator,
        )
        export_inference_checkpoint(training, parent)
        return parent

    def _make_bundle(self, root: Path, parent: Path, *, count: int = 2) -> Path:
        tasks = []
        answers = {}
        for index in range(count):
            expected = index + 1
            task_id = f"task-{expected}"
            tasks.append(
                {
                    "id": task_id,
                    "domain": "math",
                    "difficulty": 1,
                    "generator": "procedural-v1",
                    "prompt": f"Return only the integer {expected}.",
                    "provenance": {"kind": "procedural", "generator": "procedural-v1"},
                    "verifier": {
                        "kind": "integer_exact",
                        "version": "deterministic-v1",
                        "expected": expected,
                    },
                }
            )
            answers[task_id] = str(expected)
        producer = {
            "kind": "genesis_checkpoint",
            "checkpoint_sha256": sha256_file(parent),
            "checkpoint_step": 3,
            "generation": {"top_k": 1, "base_seed": 7},
        }
        accepted, audit = collect_experience(
            tasks,
            lambda task, ordinal: answers[task["id"]],
            producer_metadata=producer,
            min_score=1.0,
        )
        bundle = root / "experience"
        write_experience_bundle(bundle, accepted, audit, producer_metadata=producer, min_score=1.0)
        return bundle

    def test_dataset_masks_prompt_and_supervises_response_only(self):
        tokenizer = ByteBPETokenizer(())
        record = {"id": "exp-1", "prompt": "Return 7.", "response": "7"}
        dataset = ExperienceDataset([record], tokenizer, context_length=16)
        self.assertEqual(dataset.supervised_tokens, 1)
        x, y = dataset[0]
        self.assertEqual(tuple(x.shape), (16,))
        self.assertEqual(tuple(y.shape), (16,))
        self.assertEqual(int((y != -100).sum()), 1)
        self.assertEqual(int(y[y != -100][0]), ord("7"))

    def test_bundle_rejects_tamper_and_insufficient_experience(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self._make_parent(root)
            bundle = self._make_bundle(root, parent, count=1)
            with self.assertRaises(ValueError):
                validate_experience_bundle(bundle, parent, min_accepted=2)
            accepted_path = bundle / "accepted.jsonl"
            accepted_path.write_text(accepted_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_experience_bundle(bundle, parent)

    def test_bundle_must_match_exact_parent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self._make_parent(root, seed=1)
            other_parent = self._make_parent(root / "other", seed=2)
            bundle = self._make_bundle(root, parent)
            with self.assertRaises(ValueError):
                validate_experience_bundle(bundle, other_parent)

    def test_full_and_resumed_candidate_training_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self._make_parent(root)
            bundle = self._make_bundle(root, parent)
            parent_hash = sha256_file(parent)
            full = root / "full.pt"
            resumed = root / "resumed.pt"

            train_candidate(
                parent_checkpoint=parent,
                experience_dir=bundle,
                checkpoint=full,
                steps=4,
                batch_size=2,
                learning_rate=1e-3,
                seed=42,
                min_accepted=2,
                checkpoint_every=2,
            )
            train_candidate(
                parent_checkpoint=parent,
                experience_dir=bundle,
                checkpoint=resumed,
                steps=2,
                batch_size=2,
                learning_rate=1e-3,
                seed=42,
                min_accepted=2,
                checkpoint_every=2,
            )
            metadata = train_candidate(
                parent_checkpoint=parent,
                experience_dir=bundle,
                checkpoint=resumed,
                steps=4,
                batch_size=2,
                learning_rate=1e-3,
                seed=42,
                min_accepted=2,
                resume=resumed,
                checkpoint_every=2,
            )

            full_model, full_payload = load_model(full)
            resumed_model, resumed_payload = load_model(resumed)
            for name, tensor in full_model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed_model.state_dict()[name]), name)
            self.assertEqual(full_payload["step"], resumed_payload["step"])
            self.assertEqual(metadata["self_improvement"]["parent_checkpoint_sha256"], parent_hash)
            self.assertEqual(metadata["self_improvement"]["candidate_steps_completed"], 4)
            self.assertEqual(sha256_file(parent), parent_hash)


if __name__ == "__main__":
    unittest.main()
