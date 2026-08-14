import random
import unittest

from genesis_ai.challenger import build_task
from genesis_ai.domain_selection import oracle_response
from genesis_ai.termination_diagnostic import analyze_response
from genesis_ai.tokenizer import ByteBPETokenizer


class TerminationDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.task = build_task(random.Random(111), "code", 1)
        self.oracle = oracle_response(self.task)
        self.tokenizer = ByteBPETokenizer(())
        self.oracle_ids = self.tokenizer.encode(self.oracle)

    def test_exact_oracle_is_strict_pass(self):
        result = analyze_response(
            task=self.task,
            oracle=self.oracle,
            oracle_ids=self.oracle_ids,
            generated_ids=self.oracle_ids,
            decoded_response=self.tokenizer.decode(self.oracle_ids),
            tokenizer=self.tokenizer,
        )
        self.assertTrue(result["strict_pass"])
        self.assertTrue(result["oracle_token_prefix"])
        self.assertTrue(result["oracle_text_prefix"])
        self.assertFalse(result["prefix_then_extra"])

    def test_correct_oracle_plus_continuation_is_detected(self):
        extra_ids = self.tokenizer.encode(" + 999")
        generated_ids = self.oracle_ids + extra_ids
        result = analyze_response(
            task=self.task,
            oracle=self.oracle,
            oracle_ids=self.oracle_ids,
            generated_ids=generated_ids,
            decoded_response=self.tokenizer.decode(generated_ids),
            tokenizer=self.tokenizer,
        )
        self.assertFalse(result["strict_pass"])
        self.assertTrue(result["oracle_token_prefix"])
        self.assertTrue(result["oracle_text_prefix"])
        self.assertTrue(result["prefix_then_extra"])
        self.assertEqual(result["extra_text"], " + 999")

    def test_wrong_prefix_is_not_misclassified_as_termination(self):
        generated_ids = self.tokenizer.encode("0") + self.oracle_ids
        result = analyze_response(
            task=self.task,
            oracle=self.oracle,
            oracle_ids=self.oracle_ids,
            generated_ids=generated_ids,
            decoded_response=self.tokenizer.decode(generated_ids),
            tokenizer=self.tokenizer,
        )
        self.assertFalse(result["oracle_token_prefix"])
        self.assertFalse(result["oracle_text_prefix"])
        self.assertFalse(result["prefix_then_extra"])


if __name__ == "__main__":
    unittest.main()
