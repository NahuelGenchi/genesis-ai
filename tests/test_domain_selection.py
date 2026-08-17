import json
import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.challenger import DOMAINS
from genesis_ai.checkpoint import export_inference_checkpoint, save_checkpoint
from genesis_ai.config import ModelConfig
from genesis_ai.domain_selection import generate_domain_tasks, oracle_response, run_selection
from genesis_ai.model import GenesisLM
from genesis_ai.tokenizer import ByteBPETokenizer
from genesis_ai.verifiers import verify_task


class DomainSelectionTest(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        torch.manual_seed(1)
        tokenizer = ByteBPETokenizer(())
        model = GenesisLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=128,
                d_model=16,
                n_heads=4,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(2)
        training = root / "training.pt"
        inference = root / "model.pt"
        save_checkpoint(
            training,
            model=model,
            optimizer=optimizer,
            step=3,
            metadata={"fixture": True},
            tokenizer=tokenizer,
            batch_generator=generator,
        )
        export_inference_checkpoint(training, inference)
        return inference

    def test_domain_tasks_are_deterministic_unique_and_oracle_valid(self):
        for ordinal, domain in enumerate(DOMAINS):
            first = generate_domain_tasks(domain=domain, seed=100 + ordinal, count=12, difficulty=1)
            second = generate_domain_tasks(domain=domain, seed=100 + ordinal, count=12, difficulty=1)
            self.assertEqual(first, second)
            self.assertEqual(len({task["id"] for task in first}), 12)
            for task in first:
                answer = oracle_response(task)
                self.assertTrue(verify_task(task, answer).passed, (domain, answer))

    def test_difficulty_five_code_oracles_are_valid_for_frozen_ladder_seed(self):
        tasks = generate_domain_tasks(domain="code", seed=40003, count=60, difficulty=5)
        self.assertEqual(len({task["id"] for task in tasks}), 60)
        for task in tasks:
            answer = oracle_response(task)
            self.assertTrue(verify_task(task, answer).passed, (task["id"], answer))

    def test_selection_is_reproducible_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self._checkpoint(root)
            suite = root / "suite.json"
            suite.write_text(
                json.dumps(
                    {
                        "format_version": "1.0",
                        "suite_version": "m6-domain-selection-v1",
                        "base_seed": 55,
                        "tasks_per_domain": 2,
                        "difficulty": 1,
                        "domains": ["math", "structured", "code"],
                        "generation": {
                            "max_new_tokens": 2,
                            "temperature": 1.0,
                            "top_k": 1,
                            "seed": 99,
                        },
                        "selection_rule": "highest_exact_accuracy_then_lowest_oracle_target_loss_then_domain_name",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            first = run_selection(checkpoint=checkpoint, suite_path=suite, device="cpu")
            second = run_selection(checkpoint=checkpoint, suite_path=suite, device="cpu")
            self.assertEqual(first, second)
            self.assertIn(first["selected_domain"], DOMAINS)
            for domain in DOMAINS:
                result = first["domains"][domain]
                self.assertEqual(result["task_count"], 2)
                self.assertGreater(result["oracle_target_tokens"], 0)
                self.assertGreater(result["oracle_target_loss"], 0)
                self.assertEqual(len(result["task_set_sha256"]), 64)
                self.assertEqual(len(result["response_set_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
