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
        "suite_version": "m6-domain-selection-v2",
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
        self.assertEqual(manifest["total_generated_tasks"], 900)
        self.assertEqual(manifest["exact_cross_suite_prompt_overlap_count"], 0)
        self.assertEqual(manifest["exact_cross_suite_task_overlap_count"], 0)
        self.assertLessEqual(manifest["total_unique_prompts_across_difficulties"], 900)
        self.assertLessEqual(manifest["total_unique_tasks_across_difficulties"], 900)
        for difficulty, suite in enumerate(manifest["suites"], 1):
            self.assertEqual(suite["difficulty"], difficulty)
            self.assertEqual(suite["generated_task_count"], 180)
            self.assertEqual(sum(suite["domain_generated_counts"].values()), 180)
            self.assertEqual(suite["duplicate_prompt_count"], 180 - suite["unique_prompt_count"])
            self.assertEqual(suite["duplicate_task_count"], 180 - suite["unique_task_count"])

    def test_frozen_difficulty_one_intra_suite_repeats_do_not_break_cross_suite_audit(self):
        manifest = build_ladder_manifest(SUITES)
        d1 = manifest["suites"][0]
        self.assertEqual(d1["path"], "evals/m6-domain-selection-v2.json")
        self.assertGreaterEqual(d1["duplicate_prompt_count"], 0)
        self.assertGreaterEqual(d1["duplicate_task_count"], 0)
        self.assertEqual(manifest["exact_cross_suite_prompt_overlap_count"], 0)
        self.assertEqual(manifest["exact_cross_suite_task_overlap_count"], 0)

    def test_existing_loader_accepts_harder_files_under_same_frozen_protocol(self):
        suite = load_terminated_suite("evals/m6-domain-ladder-d2-v1.json")
        self.assertEqual(suite["suite_version"], "m6-domain-selection-v2")
        self.assertEqual(suite["difficulty"], 2)
        raw = json.loads(Path("evals/m6-domain-ladder-d2-v1.json").read_text(encoding="utf-8"))
        raw["suite_version"] = "invented-ladder-protocol"
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
