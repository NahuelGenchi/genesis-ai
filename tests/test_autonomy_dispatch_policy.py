import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.autonomy_dispatch_policy import dispatch_decision
from genesis_ai.improvement_controller import (
    LEGACY_STRATEGY_ID,
    MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY,
    RESEARCH_STRATEGIES,
)


def write_state(root: Path, checkpoint: Path, *, cycle_index: int = 0):
    state = {
        "format_version": "1.0",
        "state_version": "autonomous-state-v1",
        "cycle_index": cycle_index,
        "difficulty": 1,
        "incumbent_checkpoint": checkpoint.as_posix(),
        "cash_compute_cost_usd": 0.0,
    }
    path = root / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def write_cycle(root: Path, index: int, *, checkpoint_sha: str, strategy_id=None, focus_gain=0.0):
    cycle = root / f"{index:04d}-fixture"
    cycle.mkdir()
    plan = {"decision": {"focus_domain": "structured"}}
    if strategy_id is not None:
        plan["research_strategy"] = {"id": strategy_id}
    gate = {
        "baseline_checkpoint_sha256": checkpoint_sha,
        "decision": "reject",
        "focus_absolute_gain": focus_gain,
    }
    (cycle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (cycle / "gate.json").write_text(json.dumps(gate), encoding="utf-8")


class AutonomyDispatchPolicyTest(unittest.TestCase):
    def test_all_predeclared_strategies_exhausted_blocks_new_canonical_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "incumbent.pt"
            checkpoint.write_bytes(b"incumbent")
            sha = hashlib.sha256(b"incumbent").hexdigest()
            history = root / "cycles"
            history.mkdir()
            index = 1
            for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                write_cycle(history, index, checkpoint_sha=sha)
                index += 1
            for strategy in RESEARCH_STRATEGIES:
                for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                    write_cycle(history, index, checkpoint_sha=sha, strategy_id=strategy["id"])
                    index += 1
            decision = dispatch_decision(
                state_path=write_state(root, checkpoint, cycle_index=index - 1),
                history_root=history,
            )
            self.assertFalse(decision["canonical_training_allowed"])
            self.assertTrue(decision["all_predeclared_strategies_exhausted"])
            self.assertEqual(decision["unexhausted_research_strategies"], [])
            self.assertEqual(decision["strategy_zero_focus_gain_rejections"][LEGACY_STRATEGY_ID], 3)

    def test_one_unexhausted_strategy_keeps_canonical_training_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "incumbent.pt"
            checkpoint.write_bytes(b"incumbent")
            sha = hashlib.sha256(b"incumbent").hexdigest()
            history = root / "cycles"
            history.mkdir()
            index = 1
            for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                write_cycle(history, index, checkpoint_sha=sha)
                index += 1
            for strategy in RESEARCH_STRATEGIES[:-1]:
                for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                    write_cycle(history, index, checkpoint_sha=sha, strategy_id=strategy["id"])
                    index += 1
            decision = dispatch_decision(
                state_path=write_state(root, checkpoint),
                history_root=history,
            )
            self.assertTrue(decision["canonical_training_allowed"])
            self.assertIn(RESEARCH_STRATEGIES[-1]["id"], decision["unexhausted_research_strategies"])

    def test_new_incumbent_resets_exhaustion_without_deleting_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.pt"
            old.write_bytes(b"old")
            old_sha = hashlib.sha256(b"old").hexdigest()
            history = root / "cycles"
            history.mkdir()
            index = 1
            for strategy_id in (LEGACY_STRATEGY_ID, *(strategy["id"] for strategy in RESEARCH_STRATEGIES)):
                for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                    write_cycle(
                        history,
                        index,
                        checkpoint_sha=old_sha,
                        strategy_id=None if strategy_id == LEGACY_STRATEGY_ID else strategy_id,
                    )
                    index += 1
            new = root / "new.pt"
            new.write_bytes(b"new")
            decision = dispatch_decision(
                state_path=write_state(root, new, cycle_index=index - 1),
                history_root=history,
            )
            self.assertTrue(decision["canonical_training_allowed"])
            self.assertEqual(decision["cycles_considered_for_incumbent"], 0)
            self.assertFalse(decision["all_predeclared_strategies_exhausted"])

    def test_positive_focus_gain_rejection_does_not_consume_zero_gain_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "incumbent.pt"
            checkpoint.write_bytes(b"incumbent")
            sha = hashlib.sha256(b"incumbent").hexdigest()
            history = root / "cycles"
            history.mkdir()
            write_cycle(history, 1, checkpoint_sha=sha, focus_gain=0.1)
            decision = dispatch_decision(
                state_path=write_state(root, checkpoint),
                history_root=history,
            )
            self.assertTrue(decision["canonical_training_allowed"])
            self.assertEqual(decision["strategy_zero_focus_gain_rejections"].get(LEGACY_STRATEGY_ID, 0), 0)


if __name__ == "__main__":
    unittest.main()
