import unittest

from genesis_ai.config import ModelConfig


class ModelConfigTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        config = ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128)
        self.assertEqual(ModelConfig.from_dict(config.to_dict()), config)

    def test_invalid_head_division(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(d_model=63, n_heads=4).validate()


if __name__ == "__main__":
    unittest.main()
