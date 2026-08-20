from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genesis_ai import autonomous_training
from genesis_ai.weak_domain_funnel import FUNNEL_VERSION, TINY_TOKEN_BUDGET
from genesis_ai.weak_domain_training import (
    SCREEN_REPLAY_EXAMPLES_BY_BUDGET,
    train_screen,
    validate_screening_contract,
)


class WeakDomainTrainingTests(unittest.TestCase):
    def _write_lock(self, root: Path, **overrides: object) -> Path:
        payload: dict[str, object] = {
            "research_funnel_version": FUNNEL_VERSION,
            "funnel_stage": "tiny",
            "target_training_tokens": TINY_TOKEN_BUDGET,
            "replay_examples_per_domain": SCREEN_REPLAY_EXAMPLES_BY_BUDGET[TINY_TOKEN_BUDGET],
            "screening_only": True,
            "promotion_authority": False,
            "cash_compute_cost_usd": 0.0,
            "exact_holdout_prompt_overlap_count": 0,
        }
        payload.update(overrides)
        path = root / "curriculum.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_screening_contract_accepts_only_predeclared_zero_authority_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = self._write_lock(Path(tmp))
            result = validate_screening_contract(lock)
        self.assertEqual(result["target_training_tokens"], TINY_TOKEN_BUDGET)
        self.assertTrue(result["screening_only"])
        self.assertFalse(result["promotion_authority"])

    def test_screening_contract_rejects_authority_overlap_and_budget_drift(self) -> None:
        cases = (
            ({"promotion_authority": True}, "promotion authority"),
            ({"exact_holdout_prompt_overlap_count": 1}, "overlaps a frozen holdout"),
            ({"target_training_tokens": 224_999}, "predeclared tiny or medium budget"),
            ({"replay_examples_per_domain": 127}, "replay count"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as tmp:
                lock = self._write_lock(Path(tmp), **overrides)
                with self.assertRaisesRegex(ValueError, message):
                    validate_screening_contract(lock)

    def test_adapter_temporarily_opens_only_frozen_screen_budget_and_restores_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = self._write_lock(root)
            before = dict(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET)

            def fake_train_continuation(**_: object) -> dict[str, object]:
                self.assertEqual(
                    autonomous_training.REPLAY_EXAMPLES_BY_BUDGET[TINY_TOKEN_BUDGET],
                    SCREEN_REPLAY_EXAMPLES_BY_BUDGET[TINY_TOKEN_BUDGET],
                )
                return {"processed_tokens": TINY_TOKEN_BUDGET}

            with mock.patch.object(autonomous_training, "train_continuation", side_effect=fake_train_continuation):
                result = train_screen(
                    parent_checkpoint=root / "parent.pt",
                    curriculum_lock=lock,
                    records_path=root / "records.jsonl",
                    public_data=root / "public",
                    tokenizer_path=root / "tokenizer.json",
                    checkpoint_path=root / "candidate.pt",
                    export_path=root / "candidate-export.pt",
                    run_path=root / "run.json",
                )

            self.assertEqual(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET, before)
            self.assertEqual(result["processed_tokens"], TINY_TOKEN_BUDGET)
            self.assertTrue(result["screening_only"])
            self.assertFalse(result["promotion_authority"])
            self.assertEqual(result["cash_compute_cost_usd"], 0.0)

    def test_adapter_restores_production_budget_contract_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = self._write_lock(root)
            before = dict(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET)
            with mock.patch.object(autonomous_training, "train_continuation", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    train_screen(
                        parent_checkpoint=root / "parent.pt",
                        curriculum_lock=lock,
                        records_path=root / "records.jsonl",
                        public_data=root / "public",
                        tokenizer_path=root / "tokenizer.json",
                        checkpoint_path=root / "candidate.pt",
                        export_path=root / "candidate-export.pt",
                        run_path=root / "run.json",
                    )
            self.assertEqual(autonomous_training.REPLAY_EXAMPLES_BY_BUDGET, before)


if __name__ == "__main__":
    unittest.main()
