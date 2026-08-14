import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.gci_ladder import build_ladder_manifest, compare_ladders, score_ladder
from genesis_ai.terminated_eval import load_terminated_suite


SUITES = [
    "evals/m6-domain-selection-v2.json",
    "evals/m6-domain-ladder-d2-v1.json",
    "evals/m6-domain-ladder-d3-v1.json",
    "evals/m6-domain-ladder-d4-v1.json",
    "evals/m6-domain-ladder-d5-v1.json",
]


def result(difficulty, score, *, suite_sha=None, checkpoint="a" * 64):
    accuracy = score / 100.0
    return {
        "suite_version": "m6-domain-selection-v2" if difficulty == 1 else f"m6-domain-ladder-d{difficulty}-v1",
        "suite_sha256": suite_sha or (str(difficulty) * 64),
        "checkpoint_sha256": checkpoint,
        "difficulty": difficulty,
        "domains": {
            "code": {"exact_accuracy": accuracy},
            "math": {"exact_accuracy": accuracy},
            "structured": {"exact_accuracy": accuracy},
        },
    }


class GCILadderTest(unittest.TestCase):
    def test_project_ladder_has_no_exact_cross_suite_prompt_or_task_overlap(self):
        manifest = build_ladder_manifest(SUITES)
        self.assertEqual(manifest["difficulties"], [1, 2, 3, 4, 5])
        self.assertEqual(manifest["suite_count"], 5)
        self.assertEqual(manifest["total_tasks"], 900)
        self.assertEqual(manifest["exact_cross_suite_prompt_overlap_count"], 0)
        self.assertEqual(manifest["exact_cross_suite_task_overlap_count"], 0)

    def test_ladder_loader_accepts_only_version_difficulty_match(self):
        suite = load_terminated_suite("evals/m6-domain-ladder-d2-v1.json")
        self.assertEqual(suite["difficulty"], 2)
        raw = json.loads(Path("evals/m6-domain-ladder-d2-v1.json").read_text(encoding="utf-8"))
        raw["difficulty"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_terminated_suite(path)

    def test_strict_harmonic_ladder_penalizes_hard_difficulty_collapse(self):
        scores = [90, 80, 70, 60, 50]
        ladder = score_ladder([result(index + 1, value) for index, value in enumerate(scores)])
        expected = 5 / sum(1 / value for value in scores)
        self.assertAlmostEqual(ladder["ladder_score"], expected)
        self.assertEqual(ladder["worst_domain_exact_percent"], 50.0)

        collapsed = score_ladder([result(1, 90), result(2, 80), result(3, 70), result(4, 60), result(5, 0)])
        self.assertEqual(collapsed["ladder_score"], 0.0)

    def test_comparison_requires_identical_suite_hashes(self):
        baseline = score_ladder([result(d, 20) for d in range(1, 6)])
        candidate = score_ladder([result(d, 40) for d in range(1, 6)])
        comparison = compare_ladders(baseline, candidate)
        self.assertAlmostEqual(comparison["relative_percent_change"], 100.0)

        changed = dict(candidate)
        changed["difficulty_suite_sha256"] = dict(candidate["difficulty_suite_sha256"])
        changed["difficulty_suite_sha256"]["5"] = "x" * 64
        with self.assertRaises(ValueError):
            compare_ladders(baseline, changed)


if __name__ == "__main__":
    unittest.main()
