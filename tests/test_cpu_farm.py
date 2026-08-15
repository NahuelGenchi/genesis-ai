import json
import unittest
from pathlib import Path

from genesis_ai.cpu_farm import FARM_VERSION, aggregate, load_definition, matrix, run_screen


ROOT = Path(__file__).resolve().parents[1]
DEFINITION = ROOT / "experiments" / "cpu-farm-v1.json"


class CPUFarmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = load_definition(DEFINITION)

    def test_definition_is_bounded_and_covers_every_lane(self) -> None:
        self.assertEqual(self.definition["runner"], "ubuntu-latest")
        self.assertFalse(self.definition["paid_runners_allowed"])
        self.assertTrue(self.definition["public_only"])
        self.assertTrue(self.definition["screening_only"])
        self.assertLessEqual(len(self.definition["candidates"]), 20)
        self.assertLessEqual(self.definition["max_parallel"], 12)
        self.assertLessEqual(self.definition["timeout_minutes"], 12)
        self.assertEqual(
            set(self.definition["lanes"]),
            {"architecture", "optimizer", "tiny-model", "tokenizer", "data-filtering", "evaluation", "verifier"},
        )
        self.assertEqual(len(self.definition["candidates"]), 19)

    def test_matrix_marks_only_model_lanes_for_torch(self) -> None:
        values = matrix(self.definition)["include"]
        self.assertEqual(len(values), 19)
        for item in values:
            self.assertEqual(item["needs_torch"], item["lane"] in {"architecture", "optimizer", "tiny-model"})
            self.assertEqual(item["seed"], self.definition["seed"])

    def test_non_model_screens_are_deterministic_guards(self) -> None:
        seed = self.definition["seed"]
        first = run_screen(self.definition, lane="data-filtering", variant="min-chars-40", seed=seed)
        second = run_screen(self.definition, lane="data-filtering", variant="min-chars-40", seed=seed)
        self.assertEqual(first, second)
        self.assertFalse(first["promotion_eligible"])
        self.assertTrue(first["screening_only"])
        for variant in ("integer-exact", "json-exact", "restricted-expression"):
            result = run_screen(self.definition, lane="verifier", variant=variant, seed=seed)
            self.assertEqual(result["metrics"]["metric_value"], 1.0)
        for variant in ("m3-contract", "domain-v2-contract"):
            result = run_screen(self.definition, lane="evaluation", variant=variant, seed=seed)
            self.assertEqual(result["metrics"]["metric_value"], 1.0)

    def test_tokenizer_screen_round_trips(self) -> None:
        result = run_screen(
            self.definition,
            lane="tokenizer",
            variant="genesis-v0-trim384",
            seed=self.definition["seed"],
        )
        self.assertEqual(result["metrics"]["round_trip_failures"], 0)
        self.assertEqual(result["metrics"]["vocab_size"], 384)
        self.assertGreater(result["metrics"]["metric_value"], 1.0)

    def test_aggregate_can_shortlist_but_never_promote(self) -> None:
        values = {
            ("architecture", "layernorm-gelu"): 2.0,
            ("architecture", "rmsnorm-gelu"): 1.8,
            ("architecture", "layernorm-swiglu"): 1.9,
            ("optimizer", "adamw"): 2.0,
            ("optimizer", "sgd"): 2.1,
            ("tiny-model", "tiny-96x2"): 2.0,
            ("tiny-model", "tiny-128x2"): 1.7,
            ("tiny-model", "tiny-128x3"): 1.8,
            ("tokenizer", "genesis-v0"): 2.0,
            ("tokenizer", "genesis-v0-trim384"): 2.1,
            ("tokenizer", "byte-256"): 1.0,
            ("data-filtering", "min-chars-40"): 1.0,
            ("data-filtering", "min-chars-20"): 0.8,
            ("data-filtering", "min-chars-80"): 0.7,
            ("evaluation", "m3-contract"): 1.0,
            ("evaluation", "domain-v2-contract"): 1.0,
            ("verifier", "integer-exact"): 1.0,
            ("verifier", "json-exact"): 1.0,
            ("verifier", "restricted-expression"): 1.0,
        }
        results = []
        for candidate in self.definition["candidates"]:
            lane = candidate["lane"]
            variant = candidate["variant"]
            objective = self.definition["lanes"][lane]["objective"]
            results.append(
                {
                    "format_version": "1.0",
                    "farm_version": FARM_VERSION,
                    "lane": lane,
                    "variant": variant,
                    "seed": self.definition["seed"],
                    "screening_only": True,
                    "promotion_eligible": False,
                    "cash_compute_cost_usd": 0.0,
                    "runner_contract": "ubuntu-latest",
                    "metrics": {"metric_name": "fixture", "metric_value": values[(lane, variant)], "objective": objective},
                }
            )
        summary = aggregate(self.definition, results)
        self.assertFalse(summary["promotion_eligible"])
        self.assertTrue(summary["screening_only"])
        self.assertTrue(summary["guards_pass"])
        self.assertTrue(summary["gpu_policy"]["eligible_only_after_cpu_screen"])
        self.assertFalse(summary["gpu_policy"]["screening_result_can_promote_checkpoint"])
        shortlisted = {(item["lane"], item["variant"]) for item in summary["expensive_stage_eligible"]}
        self.assertIn(("architecture", "rmsnorm-gelu"), shortlisted)
        self.assertIn(("tiny-model", "tiny-128x2"), shortlisted)
        self.assertIn(("tokenizer", "genesis-v0-trim384"), shortlisted)
        self.assertNotIn(("optimizer", "sgd"), shortlisted)

    def test_workflow_forbids_self_hosted_and_paid_runner_shapes(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "cpu-research-farm.yml").read_text(encoding="utf-8")
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("larger", workflow.lower())
        self.assertGreaterEqual(workflow.count("runs-on: ubuntu-latest"), 3)
        self.assertIn("max-parallel: 12", workflow)
        self.assertIn("timeout-minutes: 12", workflow)
        self.assertIn("github.event.repository.private == false", workflow)
        self.assertIn("retention-days: 1", workflow)


if __name__ == "__main__":
    unittest.main()
