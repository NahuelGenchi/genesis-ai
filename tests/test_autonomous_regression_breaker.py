import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from genesis_ai.improvement_controller import (
    LEGACY_STRATEGY_ID,
    RESEARCH_STRATEGIES,
    _load_history,
    _require_autonomy_running,
    plan_next_cycle,
)


SHA = "a" * 64
SUITE_SHA = "b" * 64


def evaluation():
    return {
        "suite_version": "m6-domain-selection-v2",
        "suite_sha256": SUITE_SHA,
        "difficulty": 1,
        "domains": {
            "code": {"exact_accuracy": 0.95, "terminated_oracle_loss": 0.01, "termination_rate": 1.0},
            "math": {"exact_accuracy": 0.0, "terminated_oracle_loss": 5.0, "termination_rate": 1.0},
            "structured": {"exact_accuracy": 0.0, "terminated_oracle_loss": 7.5, "termination_rate": 1.0},
        },
    }


def write_cycle(root: Path, index: int, strategy_id: str, *, focus_gain: float, gci_change: float) -> None:
    cycle = root / f"{index:04d}"
    cycle.mkdir()
    plan = {}
    if strategy_id != LEGACY_STRATEGY_ID:
        plan["research_strategy"] = {"id": strategy_id}
    gate = {
        "baseline_checkpoint_sha256": SHA,
        "decision": "reject",
        "focus_absolute_gain": focus_gain,
        "gci_v1": {"absolute_point_change": gci_change},
    }
    (cycle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (cycle / "gate.json").write_text(json.dumps(gate), encoding="utf-8")


class AutonomousRegressionBreakerTest(unittest.TestCase):
    def test_negative_total_gci_retires_strategy_even_when_focus_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cycle(root, 1, LEGACY_STRATEGY_ID, focus_gain=0.05, gci_change=-1.0)
            write_cycle(root, 2, LEGACY_STRATEGY_ID, focus_gain=0.05, gci_change=-0.5)

            history = _load_history(root, SHA)
            self.assertEqual(history["strategy_zero_focus_gain"], {})
            self.assertEqual(history["strategy_nonpositive_gci"][LEGACY_STRATEGY_ID], 2)

            plan = plan_next_cycle(evaluation(), incumbent_checkpoint_sha256=SHA, history_root=root)
            self.assertEqual(plan["research_strategy"]["id"], "sequence-depth-v1")

    def test_all_exhausted_strategies_block_candidate_training_instead_of_repeating_last_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = 1
            strategy_ids = [LEGACY_STRATEGY_ID, *(item["id"] for item in RESEARCH_STRATEGIES)]
            for strategy_id in strategy_ids:
                for _ in range(2):
                    write_cycle(root, index, strategy_id, focus_gain=0.05, gci_change=-0.5)
                    index += 1

            with self.assertRaisesRegex(ValueError, "all predeclared repair strategies exhausted"):
                plan_next_cycle(evaluation(), incumbent_checkpoint_sha256=SHA, history_root=root)

    def test_persistent_research_hold_fails_closed_before_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "autonomy_status": "research_hold",
                        "circuit_breaker": {"active": True, "reason": "regression streak"},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"STATE": str(state)}, clear=False):
                with self.assertRaisesRegex(SystemExit, "research hold"):
                    _require_autonomy_running()


if __name__ == "__main__":
    unittest.main()
