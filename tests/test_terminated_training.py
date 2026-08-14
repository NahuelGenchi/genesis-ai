import math
import unittest

import torch

from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM
from genesis_ai.position_alignment import rolling_target_loss
from genesis_ai.terminated_training import (
    TERMINATION_DELIMITER,
    TerminatedGenerationAlignedDataset,
    build_terminated_schedule,
)
from genesis_ai.tokenizer import ByteBPETokenizer


class TerminatedTrainingTest(unittest.TestCase):
    def _tokenizer(self):
        return ByteBPETokenizer(())

    def test_dataset_matches_rolling_loss_for_terminated_response(self):
        torch.manual_seed(19)
        tokenizer = self._tokenizer()
        context_length = 16
        model = GenesisLM(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                context_length=context_length,
                d_model=16,
                n_heads=4,
                n_layers=1,
                d_ff=32,
                dropout=0.0,
                position_encoding="learned",
            )
        )
        model.eval()
        prompt = "This prompt is deliberately longer than the tiny context window."
        response = "2*x+3"
        dataset = TerminatedGenerationAlignedDataset(
            [{"id": "a", "prompt": prompt, "response": response}],
            tokenizer,
            context_length,
        )
        rolling = rolling_target_loss(
            model=model,
            tokenizer=tokenizer,
            task={"prompt": prompt},
            response=response + TERMINATION_DELIMITER,
            device="cpu",
        )
        total = 0.0
        for index in range(len(dataset)):
            x, y = dataset[index]
            _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
            self.assertIsNotNone(loss)
            total += float(loss.detach())
        self.assertEqual(len(dataset), rolling["target_tokens"])
        self.assertTrue(math.isclose(total, rolling["loss_sum"], rel_tol=1e-6, abs_tol=1e-6))
        self.assertEqual(len(dataset.first_target_indices), 1)
        self.assertEqual(len(dataset.terminator_target_indices), 1)
        terminator_index = dataset.terminator_target_indices[0]
        self.assertNotEqual(dataset.first_target_indices[0], terminator_index)
        terminated_ids = [int(dataset.targets[index]) for index in range(len(dataset))]
        self.assertTrue(tokenizer.decode(terminated_ids, errors="replace").endswith("\n"))

    def test_schedule_covers_all_first_and_terminator_targets_without_duplicates(self):
        tokenizer = self._tokenizer()
        records = [
            {"id": "a", "prompt": "p-a", "response": "abcdef"},
            {"id": "b", "prompt": "p-b", "response": "ghijkl"},
            {"id": "c", "prompt": "p-c", "response": "mnopqr"},
        ]
        dataset = TerminatedGenerationAlignedDataset(records, tokenizer, 16)
        schedule_a = build_terminated_schedule(dataset, total_samples=15, seed=123)
        schedule_b = build_terminated_schedule(dataset, total_samples=15, seed=123)
        selected = set(int(value) for value in schedule_a.tolist())
        self.assertTrue(torch.equal(schedule_a, schedule_b))
        self.assertEqual(len(selected), len(schedule_a))
        self.assertTrue(set(dataset.first_target_indices).issubset(selected))
        self.assertTrue(set(dataset.terminator_target_indices).issubset(selected))

    def test_schedule_rejects_budget_below_anchor_count(self):
        tokenizer = self._tokenizer()
        dataset = TerminatedGenerationAlignedDataset(
            [
                {"id": "a", "prompt": "p-a", "response": "ab"},
                {"id": "b", "prompt": "p-b", "response": "cd"},
            ],
            tokenizer,
            16,
        )
        with self.assertRaises(ValueError):
            build_terminated_schedule(dataset, total_samples=3, seed=1)

    def test_existing_newline_in_oracle_is_rejected(self):
        tokenizer = self._tokenizer()
        with self.assertRaises(ValueError):
            TerminatedGenerationAlignedDataset(
                [{"id": "a", "prompt": "p", "response": "bad\nanswer"}],
                tokenizer,
                16,
            )


if __name__ == "__main__":
    unittest.main()
