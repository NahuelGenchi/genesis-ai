import unittest
from pathlib import Path

from genesis_ai.scaling import benchmark_stage, load_ladder
from genesis_ai.tokenizer import ByteBPETokenizer


class ScalingLadderTest(unittest.TestCase):
    def test_project_ladder_is_valid_and_sequential(self):
        tokenizer = ByteBPETokenizer.load(Path("tokenizers/genesis-v0.json"))
        ladder = load_ladder(Path("experiments/m6-scaling-ladder-v1.json"), vocab_size=tokenizer.vocab_size)
        self.assertEqual(ladder["stages"][0]["role"], "reference")
        self.assertEqual([stage["name"] for stage in ladder["stages"]], [
            "baseline-0.4m",
            "micro-2m",
            "small-5m",
            "medium-12m",
            "medium-25m",
        ])
        self.assertTrue(ladder["promotion"]["require_zero_cash_compute"])
        self.assertTrue(ladder["promotion"]["require_zero_exact_contamination"])

    def test_microbenchmark_reports_measured_feasibility_inputs(self):
        stage = {
            "name": "fixture",
            "role": "candidate",
            "training_tokens": 1024,
            "config": {
                "context_length": 8,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "dropout": 0.0,
                "position_encoding": "learned",
                "ffn_type": "dense",
            },
        }
        result = benchmark_stage(
            stage,
            vocab_size=256,
            seed=1,
            target_tokens_per_step=32,
            warmup_steps=1,
            measured_steps=1,
            feasibility={"local_max_training_hours": 6.0, "local_peak_rss_max_mb": 24576.0},
        )
        self.assertGreater(result["parameter_count"], 0)
        self.assertGreater(result["training_tokens_per_second"], 0)
        self.assertGreater(result["estimated_training_flops_per_token"], 0)
        self.assertGreater(result["model_parameter_bytes"], 0)
        self.assertGreater(result["optimizer_state_bytes"], 0)
        self.assertEqual(result["target_training_tokens"], 1024)
        self.assertIsInstance(result["local_cpu_feasible"], bool)


if __name__ == "__main__":
    unittest.main()
