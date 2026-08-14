import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.domain_selection import generate_domain_tasks
from genesis_ai.terminated_eval import (
    SELECTION_RULE,
    load_terminated_suite,
    split_generated_answer,
)


class TerminatedEvalTest(unittest.TestCase):
    def test_frozen_v2_suite_keeps_v1_task_identity_inputs(self):
        v1 = json.loads(Path("evals/m6-domain-selection-v1.json").read_text(encoding="utf-8"))
        v2 = load_terminated_suite("evals/m6-domain-selection-v2.json")
        for field in ("base_seed", "tasks_per_domain", "difficulty", "domains", "generation"):
            self.assertEqual(v2[field], v1[field])
        self.assertEqual(v2["termination"], {"delimiter": "\n", "required": True})
        self.assertEqual(v2["selection_rule"], SELECTION_RULE)
        for ordinal, domain in enumerate(v1["domains"]):
            tasks_v1 = generate_domain_tasks(
                domain=domain,
                seed=v1["base_seed"] + ordinal,
                count=v1["tasks_per_domain"],
                difficulty=v1["difficulty"],
            )
            tasks_v2 = generate_domain_tasks(
                domain=domain,
                seed=v2["base_seed"] + ordinal,
                count=v2["tasks_per_domain"],
                difficulty=v2["difficulty"],
            )
            self.assertEqual(tasks_v2, tasks_v1)

    def test_split_requires_real_generated_newline(self):
        answer, terminated = split_generated_answer("5*x + 1\ncontinued", "\n")
        self.assertTrue(terminated)
        self.assertEqual(answer, "5*x + 1")
        answer, terminated = split_generated_answer("5*x + 1", "\n")
        self.assertFalse(terminated)
        self.assertEqual(answer, "5*x + 1")

    def test_suite_rejects_optional_or_changed_terminator(self):
        raw = json.loads(Path("evals/m6-domain-selection-v2.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            raw["termination"] = {"delimiter": "\n", "required": False}
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_terminated_suite(path)


if __name__ == "__main__":
    unittest.main()
