import unittest

import torch

from genesis_ai.aligned_training import GenerationAlignedExperienceDataset
from genesis_ai.candidate import IGNORE_INDEX
from genesis_ai.tokenizer import ByteBPETokenizer


class GenerationAlignedExperienceDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteBPETokenizer(())
        self.context = 16
        self.prompt = "P" * 24
        self.response = "abc"
        self.records = [
            {
                "prompt": self.prompt,
                "response": self.response,
            }
        ]

    def test_each_response_token_uses_real_rolling_generation_context(self):
        dataset = GenerationAlignedExperienceDataset(self.records, self.tokenizer, self.context)
        prompt_ids = self.tokenizer.encode(self.prompt + "\nAnswer:")
        response_ids = self.tokenizer.encode(self.response)
        self.assertEqual(len(dataset), len(response_ids))
        self.assertEqual(dataset.supervised_response_tokens, len(response_ids))

        for index, target in enumerate(response_ids):
            x, y = dataset[index]
            expected_history = prompt_ids + response_ids[:index]
            expected_x = torch.tensor(expected_history[-self.context :], dtype=torch.long)
            self.assertTrue(torch.equal(x, expected_x))
            self.assertEqual(len(x), self.context)
            self.assertTrue(torch.all(y[:-1] == IGNORE_INDEX))
            self.assertEqual(int(y[-1]), target)

    def test_target_is_always_final_absolute_model_position(self):
        dataset = GenerationAlignedExperienceDataset(self.records, self.tokenizer, self.context)
        for index in range(len(dataset)):
            _, y = dataset[index]
            supervised = torch.nonzero(y != IGNORE_INDEX, as_tuple=False).flatten().tolist()
            self.assertEqual(supervised, [self.context - 1])

    def test_short_history_fails_closed_instead_of_inventing_padding(self):
        with self.assertRaises(ValueError):
            GenerationAlignedExperienceDataset(
                [{"prompt": "x", "response": "y"}],
                self.tokenizer,
                128,
            )


if __name__ == "__main__":
    unittest.main()
