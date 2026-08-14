import unittest

import torch

from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM, RMSNorm, SwiGLUFFN


class ArchitectureVariantTest(unittest.TestCase):
    def test_legacy_config_defaults_are_unchanged(self):
        legacy = ModelConfig.from_dict({
            "vocab_size": 512,
            "context_length": 128,
            "d_model": 192,
            "n_heads": 6,
            "n_layers": 4,
            "d_ff": 768,
            "dropout": 0.0,
            "position_encoding": "learned",
            "ffn_type": "dense",
            "moe_experts": 4,
            "moe_top_k": 2,
            "moe_aux_loss_weight": 0.01,
        })
        self.assertEqual(legacy.norm_type, "layernorm")
        self.assertEqual(legacy.dense_activation, "gelu")
        model = GenesisLM(legacy)
        self.assertIsInstance(model.blocks[0].attn_norm, torch.nn.LayerNorm)
        self.assertIsInstance(model.blocks[0].mlp, torch.nn.Sequential)

    def test_rmsnorm_swiglu_forward_is_finite(self):
        config = ModelConfig(
            vocab_size=64,
            context_length=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=48,
            position_encoding="rotary",
            norm_type="rmsnorm",
            dense_activation="swiglu",
        )
        model = GenesisLM(config)
        self.assertIsInstance(model.blocks[0].attn_norm, RMSNorm)
        self.assertIsInstance(model.blocks[0].mlp, SwiGLUFFN)
        tokens = torch.randint(0, config.vocab_size, (2, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))

    def test_swiglu_512_matches_gelu_768_ffn_weight_budget(self):
        common = dict(
            vocab_size=512,
            context_length=128,
            d_model=192,
            n_heads=6,
            n_layers=1,
            dropout=0.0,
        )
        gelu = GenesisLM(ModelConfig(**common, d_ff=768, dense_activation="gelu"))
        swiglu = GenesisLM(ModelConfig(**common, d_ff=512, dense_activation="swiglu"))
        gelu_ffn = sum(parameter.numel() for parameter in gelu.blocks[0].mlp.parameters())
        swiglu_ffn = sum(parameter.numel() for parameter in swiglu.blocks[0].mlp.parameters())
        self.assertEqual(gelu_ffn, 294_912)
        self.assertEqual(swiglu_ffn, gelu_ffn)

    def test_invalid_variant_names_fail_closed(self):
        with self.assertRaises(ValueError):
            ModelConfig(norm_type="unknown").validate()
        with self.assertRaises(ValueError):
            ModelConfig(dense_activation="unknown").validate()


if __name__ == "__main__":
    unittest.main()
