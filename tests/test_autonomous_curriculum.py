import hashlib
import random
import unittest

from genesis_ai.autonomous_curriculum import generate_domain_records
from genesis_ai.challenger import build_task
from genesis_ai.tokenizer import ByteBPETokenizer


class AutonomousCurriculumTest(unittest.TestCase):
    def test_generates_unique_plan_bound_oracle_records(self):
        tokenizer = ByteBPETokenizer(())
        seen = set()
        records, metrics = generate_domain_records(
            tokenizer=tokenizer,
            domain="code",
            role="focus",
            count=8,
            difficulty=1,
            seed=12345,
            holdout_prompt_hashes=set(),
            global_seen_prompt_hashes=seen,
            context_length=128,
            plan_sha256="a" * 64,
        )
        self.assertEqual(len(records), 8)
        self.assertEqual(metrics["examples"], 8)
        self.assertEqual(len({record["prompt"] for record in records}), 8)
        for record in records:
            self.assertEqual(record["plan_sha256"], "a" * 64)
            self.assertEqual(record["role"], "focus")
            self.assertEqual(record["domain"], "code")
            self.assertNotIn("\n", record["response"])
            self.assertEqual(record["provenance"]["kind"], "procedural_oracle")

    def test_skips_frozen_holdout_prompt_hash(self):
        tokenizer = ByteBPETokenizer(())
        seed = 6789
        first = build_task(random.Random(seed), "math", 1)
        blocked = hashlib.sha256(str(first["prompt"]).encode("utf-8")).hexdigest()
        seen = set()
        records, metrics = generate_domain_records(
            tokenizer=tokenizer,
            domain="math",
            role="replay",
            count=4,
            difficulty=1,
            seed=seed,
            holdout_prompt_hashes={blocked},
            global_seen_prompt_hashes=seen,
            context_length=128,
            plan_sha256="b" * 64,
        )
        prompts = {record["prompt"] for record in records}
        self.assertNotIn(str(first["prompt"]), prompts)
        self.assertGreater(metrics["attempts"], 4)

    def test_rejects_invalid_role(self):
        with self.assertRaises(ValueError):
            generate_domain_records(
                tokenizer=ByteBPETokenizer(()),
                domain="code",
                role="invalid",
                count=1,
                difficulty=1,
                seed=1,
                holdout_prompt_hashes=set(),
                global_seen_prompt_hashes=set(),
                context_length=128,
                plan_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
