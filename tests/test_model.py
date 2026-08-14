import random
import tempfile
import unittest
from pathlib import Path

import torch

from genesis_ai.checkpoint import load_model, restore_training_state, save_checkpoint, tokenizer_from_payload
from genesis_ai.config import ModelConfig
from genesis_ai.generate import generate_text
from genesis_ai.model import GenesisLM
from genesis_ai.tokenizer import ByteBPETokenizer


class GenesisLMTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig(
            vocab_size=300,
            context_length=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
        )
        self.model = GenesisLM(self.config)
        self.tokenizer = ByteBPETokenizer(tuple((index, index + 1, 256 + offset) for offset, index in enumerate(range(44))))

    def test_forward_shape_and_loss(self) -> None:
        x = torch.randint(0, 300, (2, 8))
        y = torch.randint(0, 300, (2, 8))
        logits, loss = self.model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 8, 300))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

    def test_generate_extends_sequence(self) -> None:
        x = torch.tensor([[65, 66, 67]])
        output = self.model.generate(x, 4, temperature=1.0, top_k=10)
        self.assertEqual(tuple(output.shape), (1, 7))

    def test_checkpoint_round_trip_carries_tokenizer(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        batch_generator = torch.Generator().manual_seed(9)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            save_checkpoint(
                path,
                model=self.model,
                optimizer=optimizer,
                step=3,
                tokenizer=self.tokenizer,
                batch_generator=batch_generator,
            )
            restored, payload = load_model(path)
            self.assertEqual(payload["step"], 3)
            self.assertEqual(restored.config, self.config)
            self.assertEqual(tokenizer_from_payload(payload).merges, self.tokenizer.merges)

    def test_resume_restores_optimizer_step_and_rng_exactly(self) -> None:
        torch.manual_seed(1234)
        random.seed(1234)
        config = ModelConfig(vocab_size=300, context_length=8, d_model=16, n_heads=4, n_layers=1, d_ff=32)
        model = GenesisLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_generator = torch.Generator().manual_seed(77)

        def update(target_model, target_optimizer, generator):
            x = torch.randint(0, 300, (2, 8), generator=generator)
            y = torch.randint(0, 300, (2, 8), generator=generator)
            target_optimizer.zero_grad(set_to_none=True)
            _, loss = target_model(x, y)
            assert loss is not None
            loss.backward()
            target_optimizer.step()
            return x.clone(), y.clone()

        update(model, optimizer, batch_generator)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "resume.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=1,
                metadata={"learning_rate": 1e-3},
                tokenizer=self.tokenizer,
                batch_generator=batch_generator,
            )
            expected_x, expected_y = update(model, optimizer, batch_generator)
            expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
            expected_random = random.random()

            restored, payload = load_model(path)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_generator = torch.Generator()
            restore_training_state(payload, restored_optimizer, restored_generator)
            actual_x, actual_y = update(restored, restored_optimizer, restored_generator)
            actual_random = random.random()

            self.assertEqual(payload["step"], 1)
            self.assertTrue(torch.equal(expected_x, actual_x))
            self.assertTrue(torch.equal(expected_y, actual_y))
            self.assertEqual(expected_random, actual_random)
            for name, expected in expected_weights.items():
                self.assertTrue(torch.equal(expected, restored.state_dict()[name]), name)

    def test_generation_is_seeded_and_uses_checkpoint_tokenizer(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        batch_generator = torch.Generator().manual_seed(9)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            save_checkpoint(
                path,
                model=self.model,
                optimizer=optimizer,
                step=0,
                tokenizer=self.tokenizer,
                batch_generator=batch_generator,
            )
            first = generate_text(str(path), "abc", max_new_tokens=8, temperature=0.9, top_k=20, seed=42)
            second = generate_text(str(path), "abc", max_new_tokens=8, temperature=0.9, top_k=20, seed=42)
            self.assertEqual(first, second)
            self.assertTrue(first.startswith("abc"))


if __name__ == "__main__":
    unittest.main()
