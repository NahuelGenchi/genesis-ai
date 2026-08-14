import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.config import ModelConfig
from genesis_ai.model import GenesisLM
from genesis_ai.research import estimated_training_flops_per_token, load_experiment, plan_candidate, run_experiment
from genesis_ai.tokenizer import ByteBPETokenizer, save_tokenizer


def _filtered_fixture(root: Path, count: int = 40) -> Path:
    data = root / "filtered"
    data.mkdir()
    documents = [
        {"id": f"doc-{index:04d}", "text": (f"Document {index} contains deterministic research text. " * 4).strip()}
        for index in range(count)
    ]
    shard = data / "shard-00000.jsonl"
    payload = b"".join((json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode() for doc in documents)
    shard.write_bytes(payload)
    (data / "manifest.json").write_text(json.dumps({
        "documents": count,
        "shards": [{"file": shard.name, "documents": count, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}],
    }), encoding="utf-8")
    return data


class ResearchHarnessTest(unittest.TestCase):
    def test_planner_respects_compute_budget(self):
        config = ModelConfig(vocab_size=256, context_length=32, d_model=32, n_heads=4, n_layers=1, d_ff=64)
        budget = 500_000_000
        plan = plan_candidate("tiny", config, training_flop_budget=budget, target_tokens_per_step=128)
        self.assertLessEqual(plan.estimated_flops, budget)
        self.assertGreater(plan.steps, 0)
        self.assertEqual(plan.tokens_per_step, 128)
        self.assertEqual(plan.training_flops_per_token, estimated_training_flops_per_token(GenesisLM(config)))

    def test_smoke_experiment_is_reproducible_in_quality_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = _filtered_fixture(root)
            tokenizer_path = root / "tokenizer.json"
            save_tokenizer(ByteBPETokenizer(()), tokenizer_path, {})
            definition = root / "experiment.json"
            definition.write_text(Path("experiments/m4-harness-smoke.json").read_text(), encoding="utf-8")
            first = run_experiment(definition, data, tokenizer_path)
            second = run_experiment(definition, data, tokenizer_path)
            self.assertEqual(first["seed"], 4242)
            self.assertLess(first["budget_spread_fraction"], 0.05)
            self.assertEqual(
                [round(item["final_validation_loss"], 7) for item in first["results"]],
                [round(item["final_validation_loss"], 7) for item in second["results"]],
            )

    def test_definition_requires_unique_candidates(self):
        tokenizer = ByteBPETokenizer(())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            raw = json.loads(Path("experiments/m4-harness-smoke.json").read_text())
            raw["candidates"][1]["name"] = raw["candidates"][0]["name"]
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_experiment(path, tokenizer)


if __name__ == "__main__":
    unittest.main()
