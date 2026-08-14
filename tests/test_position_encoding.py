import unittest

import torch

from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM


class PositionEncodingTest(unittest.TestCase):
    def test_rotary_forward_shape_and_finite_loss(self):
        config = ModelConfig(vocab_size=300, context_length=32, d_model=32, n_heads=4, n_layers=2, d_ff=64, position_encoding="rotary")
        model = GenesisLM(config)
        x = torch.randint(0, 300, (2, 16))
        y = torch.randint(0, 300, (2, 16))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, 300))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(model.position_embedding)

    def test_rotary_removes_learned_position_parameters(self):
        learned = GenesisLM(ModelConfig(vocab_size=300, context_length=32, d_model=32, n_heads=4, n_layers=1, d_ff=64, position_encoding="learned"))
        rotary = GenesisLM(ModelConfig(vocab_size=300, context_length=32, d_model=32, n_heads=4, n_layers=1, d_ff=64, position_encoding="rotary"))
        self.assertEqual(learned.parameter_count() - rotary.parameter_count(), 32 * 32)

    def test_old_config_defaults_to_learned_positions(self):
        config = ModelConfig.from_dict({"vocab_size": 256, "context_length": 16, "d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "dropout": 0.0})
        self.assertEqual(config.position_encoding, "learned")

    def test_rotary_requires_even_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "even head dimension"):
            ModelConfig(vocab_size=256, context_length=16, d_model=36, n_heads=4, n_layers=1, d_ff=64, position_encoding="rotary").validate()


if __name__ == "__main__":
    unittest.main()
