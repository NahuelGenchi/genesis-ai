import math
import unittest

import torch

from genesis_ai.config import ModelConfig
from genesis_ai.domain_selection import _target_loss_sum
from genesis_ai.model import GenesisLM
from genesis_ai.position_alignment import predictor_positions, rolling_target_loss
from genesis_ai.tokenizer import ByteBPETokenizer


class PositionAlignmentTest(unittest.TestCase):
    def test_predictor_positions_match_without_tail_truncation(self):
        result = predictor_positions(prompt_tokens=20, response_tokens=5, context_length=128)
        self.assertEqual(result["static_first_predictor_position"], 19)
        self.assertEqual(result["generation_first_predictor_position"], 19)
        self.assertEqual(result["first_predictor_position_shift"], 0)

    def test_tail_anchored_static_window_shifts_first_response_prediction(self):
        result = predictor_positions(prompt_tokens=140, response_tokens=14, context_length=128)
        self.assertEqual(result["static_first_predictor_position"], 114)
        self.assertEqual(result["generation_first_predictor_position"], 127)
        self.assertEqual(result["first_predictor_position_shift"], 13)

    def test_static_and_rolling_loss_match_when_sequence_fits_context(self):
        torch.manual_seed(7)
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
                position_encoding="learned",
            )
        )
        model.eval()
        task = {"prompt": "Compute a short value."}
        response = "12"
        static_sum, static_tokens = _target_loss_sum(model, tokenizer, task, response, "cpu")
        rolling = rolling_target_loss(
            model=model,
            tokenizer=tokenizer,
            task=task,
            response=response,
            device="cpu",
        )
        self.assertEqual(static_tokens, rolling["target_tokens"])
        self.assertTrue(math.isclose(static_sum, rolling["loss_sum"], rel_tol=1e-6, abs_tol=1e-6))
        self.assertEqual(rolling["first_predictor_position_shift"], 0)


if __name__ == "__main__":
    unittest.main()
