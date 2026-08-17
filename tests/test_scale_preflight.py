import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.model import GenesisLM
from genesis_ai.scale_preflight import load_preflight_contract


DEFINITION = Path("experiments/m6-scale-5m-rope-preflight-v1.json")
FINALIST = Path("research/m6-architecture-finalist-v1.json")


class ScalePreflightTest(unittest.TestCase):
    def test_exact_scale_config_consumes_reproduced_rope_winner(self):
        definition, finalist, config = load_preflight_contract(DEFINITION, FINALIST)
        self.assertEqual(finalist["accepted_architecture"], "rope-only")
        self.assertTrue(finalist["decision"]["passed"])
        self.assertEqual(config.position_encoding, "rotary")
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_layers, 6)
        self.assertEqual(config.d_ff, 1056)
        self.assertEqual(GenesisLM(config).parameter_count(), 4_954_624)
        self.assertEqual(definition["target_training_tokens_per_replica"], 20_000_000)
        self.assertEqual(definition["maximum_projected_training_seconds_per_replica"], 14_400)

    def test_preflight_rejects_unreproduced_architecture(self):
        definition = json.loads(DEFINITION.read_text(encoding="utf-8"))
        finalist = json.loads(FINALIST.read_text(encoding="utf-8"))
        finalist["decision"]["passed"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition_path = root / "definition.json"
            finalist_path = root / "finalist.json"
            definition_path.write_text(json.dumps(definition), encoding="utf-8")
            finalist_path.write_text(json.dumps(finalist), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_preflight_contract(definition_path, finalist_path)

    def test_preflight_rejects_parameter_drift(self):
        definition = json.loads(DEFINITION.read_text(encoding="utf-8"))
        finalist = json.loads(FINALIST.read_text(encoding="utf-8"))
        definition["expected_parameter_count"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition_path = root / "definition.json"
            finalist_path = root / "finalist.json"
            definition_path.write_text(json.dumps(definition), encoding="utf-8")
            finalist_path.write_text(json.dumps(finalist), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_preflight_contract(definition_path, finalist_path)


if __name__ == "__main__":
    unittest.main()
