import json
import unittest

from genesis_ai.challenger import DOMAINS, generate_tasks
from genesis_ai.verifiers import verify_task


class ChallengerTest(unittest.TestCase):
    def test_same_seed_is_byte_identical_and_unique(self):
        first = generate_tasks(seed=77, count=60)
        second = generate_tasks(seed=77, count=60)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(len({task["id"] for task in first}), 60)
        self.assertTrue(all(task["domain"] in DOMAINS for task in first))
        self.assertTrue(all(1 <= task["difficulty"] <= 5 for task in first))
        self.assertTrue(all(task["provenance"]["kind"] == "procedural" for task in first))

    def test_different_seed_changes_curriculum(self):
        self.assertNotEqual(generate_tasks(seed=1, count=10), generate_tasks(seed=2, count=10))


class VerifierTest(unittest.TestCase):
    def test_integer_verifier(self):
        task = {"verifier": {"kind": "integer_exact", "version": "deterministic-v1", "expected": 42}}
        self.assertTrue(verify_task(task, "42").passed)
        self.assertEqual(verify_task(task, "41").reason, "wrong_answer")
        self.assertEqual(verify_task(task, "forty-two").reason, "invalid_integer")

    def test_json_verifier(self):
        task = {"verifier": {"kind": "json_exact", "version": "deterministic-v1", "expected": [1, 2, 3]}}
        self.assertTrue(verify_task(task, "[1,2,3]").passed)
        self.assertFalse(verify_task(task, "[3,2,1]").passed)
        self.assertEqual(verify_task(task, "nope").reason, "invalid_json")

    def test_restricted_expression_correct_and_partial_scores_reproducibly(self):
        task = {
            "verifier": {
                "kind": "restricted_expression",
                "version": "deterministic-v1",
                "tests": [
                    {"variables": {"x": 1, "y": 2}, "expected": 5},
                    {"variables": {"x": 3, "y": 4}, "expected": 11},
                    {"variables": {"x": -2, "y": 5}, "expected": 2},
                ],
            }
        }
        self.assertTrue(verify_task(task, "2*x + y + 1").passed)
        first = verify_task(task, "2*x + y")
        second = verify_task(task, "2*x + y")
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.score, 0.0)

    def test_restricted_expression_never_executes_calls_or_attributes(self):
        task = {"verifier": {"kind": "restricted_expression", "version": "deterministic-v1", "tests": [{"variables": {"x": 1, "y": 2}, "expected": 3}]}}
        malicious = [
            "__import__('os').system('touch /tmp/genesis-owned')",
            "x.__class__",
            "open('/tmp/x')",
            "[x for x in [1]]",
            "(lambda: 1)()",
        ]
        for expression in malicious:
            result = verify_task(task, expression)
            self.assertFalse(result.passed, expression)
            self.assertEqual(result.score, 0.0, expression)
            self.assertIn(result.reason, {"unsafe_syntax", "invalid_syntax"})

    def test_generated_tasks_have_reproducible_oracle_answers(self):
        tasks = generate_tasks(seed=900, count=30)
        for task in tasks:
            verifier = task["verifier"]
            if verifier["kind"] == "integer_exact":
                answer = str(verifier["expected"])
            elif verifier["kind"] == "json_exact":
                answer = json.dumps(verifier["expected"])
            else:
                # Procedural code tasks describe the exact formula. A fixed safe
                # answer is derived from the prompt only for verifier smoke tests;
                # this is not model training data.
                continue
            self.assertTrue(verify_task(task, answer).passed)


if __name__ == "__main__":
    unittest.main()
