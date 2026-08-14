import unittest

from genesis_ai.alignment_diagnostic import end_anchored_logit_positions, longest_common_prefix


class AlignmentDiagnosticTest(unittest.TestCase):
    def test_end_anchored_response_occupies_tail_positions(self):
        positions = end_anchored_logit_positions(prompt_length=140, response_length=14, context_length=128)
        self.assertEqual(positions[0], 114)
        self.assertEqual(positions[-1], 127)
        self.assertEqual(positions, list(range(114, 128)))
        self.assertEqual(127 - positions[0], 13)

    def test_first_target_position_moves_with_response_length(self):
        short = end_anchored_logit_positions(prompt_length=140, response_length=13, context_length=128)
        long = end_anchored_logit_positions(prompt_length=140, response_length=16, context_length=128)
        self.assertEqual(short[0], 115)
        self.assertEqual(long[0], 112)
        self.assertEqual(short[-1], 127)
        self.assertEqual(long[-1], 127)

    def test_longest_common_prefix(self):
        self.assertEqual(longest_common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(longest_common_prefix([9], [1, 2]), 0)
        self.assertEqual(longest_common_prefix([1, 2, 3, 4], [1, 2]), 2)
        self.assertEqual(longest_common_prefix([], [1, 2]), 0)

    def test_invalid_response_length_fails_closed(self):
        with self.assertRaises(ValueError):
            end_anchored_logit_positions(prompt_length=140, response_length=128, context_length=128)


if __name__ == "__main__":
    unittest.main()
