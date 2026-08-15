import unittest

from genesis_ai.structured_diagnostic import _common_prefix_length, _multiset_overlap, _quartile_index


class StructuredDiagnosticTest(unittest.TestCase):
    def test_common_prefix_length_stops_at_first_error(self):
        self.assertEqual(_common_prefix_length([1, 2, 3, 4], [1, 2, 9, 4]), 2)
        self.assertEqual(_common_prefix_length([1, 2], [1, 2, 3]), 2)
        self.assertEqual(_common_prefix_length([1], [9]), 0)

    def test_multiset_overlap_handles_duplicates(self):
        self.assertEqual(_multiset_overlap([1, 1, 2, 4], [1, 2, 2, 3]), 2)
        self.assertEqual(_multiset_overlap([], [1, 2]), 0)
        self.assertEqual(_multiset_overlap(["1"], [1]), 0)

    def test_quartile_index_assigns_each_position_exactly_once(self):
        for length in range(1, 33):
            buckets = [0, 0, 0, 0]
            for position in range(length):
                bucket = _quartile_index(position, length)
                self.assertIn(bucket, range(4))
                buckets[bucket] += 1
            self.assertEqual(sum(buckets), length)
        self.assertEqual([_quartile_index(i, 8) for i in range(8)], [0, 0, 1, 1, 2, 2, 3, 3])

    def test_quartile_index_rejects_out_of_range_positions(self):
        with self.assertRaises(ValueError):
            _quartile_index(0, 0)
        with self.assertRaises(ValueError):
            _quartile_index(4, 4)
        with self.assertRaises(ValueError):
            _quartile_index(-1, 4)


if __name__ == "__main__":
    unittest.main()
