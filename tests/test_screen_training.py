import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from genesis_ai import autonomous_training
from genesis_ai.screen_training import train_screen


class ScreenTrainingTest(unittest.TestCase):
    def test_screen_budget_extension_is_temporary_and_non_promoting(self):
        original = dict(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET)

        def fake_train(**kwargs):
            self.assertEqual(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET[150_000], 64)
            return {"processed_tokens": 150_528, "cash_compute_cost_usd": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curriculum = root / "curriculum.json"
            curriculum.write_text(
                json.dumps(
                    {
                        "funnel_version": "weak-domain-successive-halving-v1",
                        "screening_only": True,
                        "promotion_eligible": False,
                        "promotion_authority": False,
                        "target_training_tokens": 150_000,
                        "replay_examples_per_domain": 64,
                        "variant_id": "structured-prefix-next-v1",
                        "stage": "tiny",
                    }
                ),
                encoding="utf-8",
            )
            run = root / "training.json"
            with patch(
                "genesis_ai.screen_training.autonomous_training.train_continuation",
                side_effect=fake_train,
            ) as trainer:
                result = train_screen(
                    parent_checkpoint=root / "parent.pt",
                    curriculum_lock=curriculum,
                    records_path=root / "records.jsonl",
                    public_data=root / "public",
                    tokenizer_path=root / "tokenizer.json",
                    checkpoint_path=root / "training.pt",
                    export_path=root / "candidate.pt",
                    run_path=run,
                )
                trainer.assert_called_once()
            self.assertEqual(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET, original)
            self.assertFalse(result["promotion_authority"])
            self.assertFalse(result["promotion_eligible"])
            self.assertTrue(result["screening_only"])
            persisted = json.loads(run.read_text(encoding="utf-8"))
            self.assertFalse(persisted["promotion_authority"])

    def test_rejects_any_screen_that_claims_promotion_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curriculum = root / "curriculum.json"
            curriculum.write_text(
                json.dumps(
                    {
                        "funnel_version": "weak-domain-successive-halving-v1",
                        "screening_only": True,
                        "promotion_eligible": True,
                        "promotion_authority": False,
                        "target_training_tokens": 150_000,
                        "replay_examples_per_domain": 64,
                        "variant_id": "structured-prefix-next-v1",
                        "stage": "tiny",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                train_screen(
                    parent_checkpoint=root / "parent.pt",
                    curriculum_lock=curriculum,
                    records_path=root / "records.jsonl",
                    public_data=root / "public",
                    tokenizer_path=root / "tokenizer.json",
                    checkpoint_path=root / "training.pt",
                    export_path=root / "candidate.pt",
                    run_path=root / "run.json",
                )


if __name__ == "__main__":
    unittest.main()
