import unittest

import torch

from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM
from genesis_ai.quantization import dynamic_int8, quantized_linear_count, serialized_state_dict_bytes


class QuantizationTest(unittest.TestCase):
    def test_dynamic_int8_quantizes_linears_and_runs(self):
        torch.manual_seed(4)
        model = GenesisLM(ModelConfig(vocab_size=256, context_length=16, d_model=32, n_heads=4, n_layers=1, d_ff=64))
        model.eval()
        quantized = dynamic_int8(model)
        self.assertGreater(quantized_linear_count(quantized), 0)
        x = torch.randint(0, 256, (2, 12))
        y = torch.randint(0, 256, (2, 12))
        with torch.no_grad():
            logits, loss = quantized(x, y)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

    def test_dynamic_int8_state_dict_is_smaller_on_representative_model(self):
        model = GenesisLM(ModelConfig(vocab_size=512, context_length=128, d_model=96, n_heads=4, n_layers=3, d_ff=384))
        quantized = dynamic_int8(model)
        self.assertLess(serialized_state_dict_bytes(quantized), serialized_state_dict_bytes(model))


if __name__ == "__main__":
    unittest.main()
