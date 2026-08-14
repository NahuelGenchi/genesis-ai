import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.checkpoint import load_model, save_checkpoint
from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM


class GenesisLMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig(
            vocab_size=256,
            context_length=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
        )
        self.model = GenesisLM(self.config)

    def test_forward_shape_and_loss(self) -> None:
        x = torch.randint(0, 256, (2, 8))
        y = torch.randint(0, 256, (2, 8))
        logits, loss = self.model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 8, 256))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

    def test_generate_extends_sequence(self) -> None:
        x = torch.tensor([[65, 66, 67]])
        output = self.model.generate(x, 4, temperature=1.0, top_k=10)
        self.assertEqual(tuple(output.shape), (1, 7))

    def test_checkpoint_round_trip(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            save_checkpoint(path, model=self.model, optimizer=optimizer, step=3)
            restored, payload = load_model(path)
            self.assertEqual(payload["step"], 3)
            self.assertEqual(restored.config, self.config)


if __name__ == "__main__":
    unittest.main()
