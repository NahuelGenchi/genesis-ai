import unittest

from genesis_ai.architecture_tournament import _candidate_config, plan_candidate
from genesis_ai.config import ModelConfig


class ArchitectureTournamentTest(unittest.TestCase):
    def test_plan_never_exceeds_flop_budget(self):
        config = ModelConfig(vocab_size=512, context_length=128, d_model=192, n_heads=6, n_layers=4, d_ff=768)
        plan = plan_candidate(config, flop_budget=2_000_000_000_000, target_tokens_per_step=1024)
        self.assertGreater(plan["steps"], 0)
        self.assertLessEqual(plan["estimated_training_flops"], 2_000_000_000_000)
        self.assertEqual(plan["processed_tokens"], plan["steps"] * 1024)

    def test_parameter_matched_swiglu_has_same_ffn_weight_budget(self):
        gelu = _candidate_config({
            "context_length": 128,
            "d_model": 192,
            "n_heads": 6,
            "n_layers": 4,
            "d_ff": 768,
            "dense_activation": "gelu",
        })
        swiglu = _candidate_config({
            "context_length": 128,
            "d_model": 192,
            "n_heads": 6,
            "n_layers": 4,
            "d_ff": 512,
            "dense_activation": "swiglu",
        })
        gelu_ffn_weights_per_layer = 2 * gelu.d_model * gelu.d_ff
        swiglu_ffn_weights_per_layer = 3 * swiglu.d_model * swiglu.d_ff
        self.assertEqual(gelu_ffn_weights_per_layer, 294_912)
        self.assertEqual(swiglu_ffn_weights_per_layer, gelu_ffn_weights_per_layer)

    def test_candidate_config_is_frozen_to_project_vocab(self):
        self.assertEqual(_candidate_config({}).vocab_size, 512)
        with self.assertRaises(ValueError):
            _candidate_config({"vocab_size": 256})

    def test_tiny_flop_budget_fails_closed(self):
        config = ModelConfig(vocab_size=512, context_length=128, d_model=192, n_heads=6, n_layers=4, d_ff=768)
        with self.assertRaises(ValueError):
            plan_candidate(config, flop_budget=1, target_tokens_per_step=1024)


if __name__ == "__main__":
    unittest.main()
