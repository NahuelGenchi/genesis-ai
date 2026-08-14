import unittest

import torch

from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM, SparseMoE


class SparseMoETest(unittest.TestCase):
    def _config(self, experts: int = 4) -> ModelConfig:
        return ModelConfig(
            vocab_size=300,
            context_length=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            ffn_type="moe",
            moe_experts=experts,
            moe_top_k=2,
            moe_aux_loss_weight=0.01,
        )

    def test_sparse_forward_routes_top_k_and_router_gets_gradient(self):
        torch.manual_seed(5)
        model = GenesisLM(self._config())
        model.train()
        model.reset_routing_stats()
        x = torch.randint(0, 300, (3, 16))
        y = torch.randint(0, 300, (3, 16))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (3, 16, 300))
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        metrics = model.routing_metrics()
        self.assertEqual(len(metrics), 2)
        for layer in metrics:
            self.assertEqual(layer["assignments"], 3 * 16 * 2)
            self.assertGreater(layer["utilization"], 0.0)
            self.assertAlmostEqual(sum(layer["fractions"]), 1.0, places=6)
        first_moe = model.blocks[0].mlp
        self.assertIsInstance(first_moe, SparseMoE)
        assert isinstance(first_moe, SparseMoE)
        self.assertIsNotNone(first_moe.router.weight.grad)
        self.assertTrue(torch.isfinite(first_moe.router.weight.grad).all())

    def test_active_parameter_estimate_excludes_inactive_experts(self):
        moe = GenesisLM(ModelConfig(vocab_size=512, context_length=128, d_model=96, n_heads=4, n_layers=3, d_ff=192, ffn_type="moe", moe_experts=4, moe_top_k=2))
        dense = GenesisLM(ModelConfig(vocab_size=512, context_length=128, d_model=96, n_heads=4, n_layers=3, d_ff=384, ffn_type="dense"))
        self.assertGreater(moe.parameter_count(), moe.estimated_active_parameter_count())
        self.assertGreater(moe.parameter_count(), dense.parameter_count())
        self.assertLess(abs(moe.estimated_active_parameter_count() - dense.parameter_count()), 5000)

    def test_invalid_moe_top_k_fails(self):
        with self.assertRaisesRegex(ValueError, "moe_top_k"):
            ModelConfig(ffn_type="moe", moe_experts=2, moe_top_k=3).validate()


if __name__ == "__main__":
    unittest.main()
