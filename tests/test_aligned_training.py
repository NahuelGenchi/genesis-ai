import math
import unittest

import torch

from genesis_ai.aligned_training import (
    GenerationAlignedExperienceDataset,
    batch_from_indices,
    build_aligned_schedule,
)
from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM
from genesis_ai.position_alignment import rolling_target_loss
from genesis_ai.tokenizer import ByteBPETokenizer


class AlignedTrainingTest(unittest.TestCase):
    def _tokenizer(self):
        return ByteBPETokenizer(())

    def test_dataset_loss_matches_generation_aligned_rolling_loss(self):
        torch.manual_seed(17)
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
        prompt = "This prompt is deliberately much longer than sixteen byte tokens."
        response = "12+3"
        dataset = GenerationAlignedExperienceDataset(
            [{"id": "a", "prompt": prompt, "response": response}],
            tokenizer,
            context_length,
        )
        rolling = rolling_target_loss(
            model=model,
            tokenizer=tokenizer,
            task={"prompt": prompt},
            response=response,
            device="cpu",
        )
        total = 0.0
        for index in range(len(dataset)):
            x, y = dataset[index]
            _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
            self.assertIsNotNone(loss)
            total += float(loss.detach())
            self.assertEqual(int(dataset.predictor_positions[index]), context_length - 1)
        self.assertEqual(len(dataset), rolling["target_tokens"])
        self.assertTrue(math.isclose(total, rolling["loss_sum"], rel_tol=1e-6, abs_tol=1e-6))

    def test_right_padding_preserves_short_context_predictor_position(self):
        tokenizer = self._tokenizer()
        dataset = GenerationAlignedExperienceDataset(
            [{"id": "a", "prompt": "x", "response": "y"}],
            tokenizer,
            32,
        )
        prompt_tokens = len(tokenizer.encode("x\nAnswer:"))
        self.assertEqual(int(dataset.predictor_positions[0]), prompt_tokens - 1)
        x, y = dataset[0]
        self.assertEqual(int((y != -100).sum()), 1)
        self.assertEqual(int(y[dataset.predictor_positions[0]]), tokenizer.encode("y")[0])
        self.assertTrue(bool((x[prompt_tokens:] == 0).all()))

    def test_schedule_is_unique_deterministic_and_covers_every_first_target(self):
        tokenizer = self._tokenizer()
        records = [
            {"id": "a", "prompt": "p-a", "response": "abcdef"},
            {"id": "b", "prompt": "p-b", "response": "ghijkl"},
            {"id": "c", "prompt": "p-c", "response": "mnopqr"},
        ]
        dataset = GenerationAlignedExperienceDataset(records, tokenizer, 16)
        first = set(dataset.first_target_indices)
        schedule_a = build_aligned_schedule(dataset, total_samples=12, seed=123)
        schedule_b = build_aligned_schedule(dataset, total_samples=12, seed=123)
        self.assertTrue(torch.equal(schedule_a, schedule_b))
        self.assertEqual(len(torch.unique(schedule_a)), len(schedule_a))
        self.assertTrue(first.issubset(set(int(value) for value in schedule_a.tolist())))
        self.assertEqual(int((dataset.response_ordinals[schedule_a] == 0).sum()), len(records))
        x, y = batch_from_indices(dataset, schedule_a[:4])
        self.assertEqual(tuple(x.shape), (4, 16))
        self.assertEqual(tuple(y.shape), (4, 16))
        self.assertEqual(int((y != -100).sum()), 4)

    def test_schedule_rejects_budget_that_cannot_cover_first_tokens(self):
        tokenizer = self._tokenizer()
        dataset = GenerationAlignedExperienceDataset(
            [
                {"id": "a", "prompt": "p-a", "response": "ab"},
                {"id": "b", "prompt": "p-b", "response": "cd"},
            ],
            tokenizer,
            16,
        )
        with self.assertRaises(ValueError):
            build_aligned_schedule(dataset, total_samples=1, seed=1)


if __name__ == "__main__":
    unittest.main()
