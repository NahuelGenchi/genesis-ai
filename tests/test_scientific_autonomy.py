import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.improvement_controller import (
    HINT_END_TO_END_FAILURE_LIMIT,
    LEGACY_STRATEGY_ID,
    MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY,
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
            "math": {"exact_accuracy": 0.0, "terminated_oracle_loss": 5.0, "termination_rate": 0.93},
            "structured": {"exact_accuracy": 0.0, "terminated_oracle_loss": 7.5, "termination_rate": 1.0},
        },
    }


def evidence():
    return {
        "format_version": "1.0",
        "farm_version": "cpu-screen-v1",
        "cash_compute_cost_usd": 0.0,
        "guards_pass": True,
        "screening_only": True,
        "promotion_eligible": False,
        "source_commit": "c" * 40,
        "workflow_run_id": 12345,
        "gpu_policy": {
            "eligible_only_after_cpu_screen": True,
            "full_reproduction_and_frozen_evaluation_required_before_promotion": True,
            "screening_result_can_promote_checkpoint": False,
        },
        "expensive_stage_eligible": [
            {"lane": "data-filtering", "variant": "min-chars-80", "improvement_fraction": 0.10}
        ],
    }


def write_cycle(root: Path, index: int, *, strategy=None, hint=True, focus_gain=0.0, decision="reject"):
    cycle = root / f"{index:04d}-fixture"
    cycle.mkdir()
    plan = {"decision": {"focus_domain": "structured"}}
    if strategy is not None:
        plan["research_strategy"] = {"id": strategy}
    if hint:
        plan["research_evidence"] = {
            "applied_hint": {"lane": "data-filtering", "variant": "min-chars-80", "improvement_fraction": 0.10}
        }
    gate = {
        "baseline_checkpoint_sha256": SHA,
        "decision": decision,
        "focus_absolute_gain": focus_gain,
    }
    (cycle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (cycle / "gate.json").write_text(json.dumps(gate), encoding="utf-8")


class ScientificAutonomyTest(unittest.TestCase):
    def test_legacy_stagnation_selects_materially_different_sequence_depth_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(1, MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY + 1):
                write_cycle(root, index)
            plan = plan_next_cycle(
                evaluation(),
                incumbent_checkpoint_sha256=SHA,
                cycle_index=10,
                history_root=root,
            )
            self.assertEqual(plan["history_summary"]["strategy_zero_focus_gain"][LEGACY_STRATEGY_ID], 3)
            self.assertEqual(plan["research_strategy"]["id"], "sequence-depth-v1")
            self.assertEqual(plan["decision"]["mode"], "research-repair")
            self.assertEqual(plan["decision"]["focus_examples"], 1024)
            self.assertEqual(
                plan["decision"]["continuation_update_weights"],
                {"structured": 0.65, "code": 0.25, "math": 0.10},
            )
            self.assertTrue(plan["research_escalation"]["required"])
            self.assertIn("architecture-tournament", plan["research_escalation"]["actions"])

    def test_exhausted_strategy_rotates_to_next_high_level_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = 1
            for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                write_cycle(root, index)
                index += 1
            for _ in range(MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY):
                write_cycle(root, index, strategy="sequence-depth-v1", hint=False)
                index += 1
            plan = plan_next_cycle(
                evaluation(),
                incumbent_checkpoint_sha256=SHA,
                cycle_index=index,
                history_root=root,
            )
            self.assertEqual(plan["research_strategy"]["id"], "anti-forgetting-v1")
            self.assertEqual(plan["decision"]["focus_examples"], 1536)
            self.assertEqual(
                plan["decision"]["continuation_update_weights"],
                {"structured": 0.50, "code": 0.40, "math": 0.10},
            )

    def test_cpu_hint_is_retired_after_repeated_end_to_end_zero_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(1, HINT_END_TO_END_FAILURE_LIMIT + 1):
                write_cycle(root, index)
            plan = plan_next_cycle(
                evaluation(),
                incumbent_checkpoint_sha256=SHA,
                cycle_index=20,
                research_evidence=evidence(),
                history_root=root,
            )
            self.assertEqual(plan["decision"]["public_min_chars"], 0)
            research = plan["research_evidence"]
            self.assertIsNone(research["applied_hint"])
            self.assertEqual(len(research["retired_eligible_hints"]), 1)
            retired = research["retired_eligible_hints"][0]
            self.assertEqual(retired["variant"], "min-chars-80")
            self.assertEqual(retired["end_to_end_zero_gain_rejections"], HINT_END_TO_END_FAILURE_LIMIT)

    def test_history_and_strategy_are_hash_bound_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(1, 4):
                write_cycle(root, index)
            first = plan_next_cycle(evaluation(), incumbent_checkpoint_sha256=SHA, cycle_index=4, history_root=root)
            replay = plan_next_cycle(evaluation(), incumbent_checkpoint_sha256=SHA, cycle_index=4, history_root=root)
            self.assertEqual(first, replay)
            self.assertEqual(len(first["plan_sha256"]), 64)
            write_cycle(root, 4, strategy="sequence-depth-v1", hint=False)
            changed = plan_next_cycle(evaluation(), incumbent_checkpoint_sha256=SHA, cycle_index=5, history_root=root)
            self.assertNotEqual(first["plan_sha256"], changed["plan_sha256"])
            self.assertEqual(changed["history_summary"]["cycles_considered"], 4)


if __name__ == "__main__":
    unittest.main()
