import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.result_provenance import bind_result


class ResultProvenanceTest(unittest.TestCase):
    def test_binds_source_commit_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text('{"format_version":"1.0"}\n', encoding="utf-8")
            bind_result(path, "abc123", "42")
            value = json.loads(path.read_text())
            self.assertEqual(value["result_provenance"], {"source_commit": "abc123", "workflow_run_id": "42"})

    def test_requires_provenance_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                bind_result(path, "", "42")


if __name__ == "__main__":
    unittest.main()
