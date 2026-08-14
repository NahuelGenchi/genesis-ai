import unittest

from genesis_ai.bootstrap_corpus import strip_gutenberg_wrapper, systematic_paragraph_sample
from genesis_ai.ingest import IngestError


class BootstrapCorpusTest(unittest.TestCase):
    def test_strips_gutenberg_wrapper(self):
        text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n\nalpha\n\nbeta\n\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nfooter"
        self.assertEqual(strip_gutenberg_wrapper(text), "alpha\n\nbeta")

    def test_missing_marker_fails(self):
        with self.assertRaisesRegex(IngestError, "START marker"):
            strip_gutenberg_wrapper("no wrapper")

    def test_systematic_sample_is_deterministic_and_bounded(self):
        text = "\n\n".join(f"paragraph {index} " + "x" * 80 for index in range(100))
        first, first_stride = systematic_paragraph_sample(text, 1000)
        second, second_stride = systematic_paragraph_sample(text, 1000)
        self.assertEqual(first, second)
        self.assertEqual(first_stride, second_stride)
        self.assertLessEqual(len(first.encode("utf-8")), 1000)
        self.assertGreater(first_stride, 1)


if __name__ == "__main__":
    unittest.main()
