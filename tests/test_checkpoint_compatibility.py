import unittest
from pathlib import Path

from genesis_ai.checkpoint import load_model, tokenizer_from_payload


class FrozenCheckpointCompatibilityTest(unittest.TestCase):
    def test_genesis_tiny_v0_still_loads(self):
        checkpoint = Path("checkpoints/genesis-tiny-v0.pt")
        self.assertTrue(checkpoint.is_file())
        model, payload = load_model(checkpoint)
        self.assertEqual(model.parameter_count(), 394560)
        self.assertEqual(model.config.ffn_type, "dense")
        self.assertEqual(model.config.position_encoding, "learned")
        self.assertEqual(tokenizer_from_payload(payload).vocab_size, 512)
        state_keys = model.state_dict().keys()
        self.assertIn("blocks.0.mlp.0.weight", state_keys)
        self.assertIn("blocks.0.mlp.2.weight", state_keys)


if __name__ == "__main__":
    unittest.main()
