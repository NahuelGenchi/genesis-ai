import unittest

from genesis_ai.multidomain_curriculum import CURRICULUM_VERSION, generate_records
from genesis_ai.tokenizer import ByteBPETokenizer


class MultidomainCurriculumTest(unittest.TestCase):
    def test_generation_is_balanced_deterministic_and_oracle_clean(self):
        tokenizer = ByteBPETokenizer(())
        first, first_summary = generate_records(
            tokenizer=tokenizer,
            holdout_prompt_hashes=set(),
            examples_per_domain=4,
            seed=1234,
            difficulty=1,
            context_length=128,
        )
        second, second_summary = generate_records(
            tokenizer=tokenizer,
            holdout_prompt_hashes=set(),
            examples_per_domain=4,
            seed=1234,
            difficulty=1,
            context_length=128,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(len(first), 12)
        counts = {"code": 0, "math": 0, "structured": 0}
        for record in first:
            self.assertEqual(record["curriculum"], CURRICULUM_VERSION)
            self.assertNotIn("\n", record["response"])
            counts[record["domain"]] += 1
        self.assertEqual(counts, {"code": 4, "math": 4, "structured": 4})
        self.assertEqual(first_summary["exact_holdout_prompt_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
