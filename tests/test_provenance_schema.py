import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProvenanceSchemaTest(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "schemas/dataset-source.schema.json").read_text())
        self.example = json.loads((ROOT / "data/source.example.json").read_text())

    def test_contract_requires_core_provenance(self):
        required = set(self.schema["required"])
        self.assertTrue({"origin", "rights", "retrieval", "content"} <= required)
        self.assertEqual(self.schema["properties"]["rights"]["properties"]["training_allowed"]["const"], True)
        self.assertIn("sha256", self.schema["properties"]["retrieval"]["required"])

    def test_example_satisfies_required_contract(self):
        for key in self.schema["required"]:
            self.assertIn(key, self.example)
        self.assertTrue(self.example["rights"]["basis"].strip())
        self.assertIs(self.example["rights"]["training_allowed"], True)
        self.assertRegex(self.example["retrieval"]["sha256"], re.compile(r"^[0-9a-f]{64}$"))


if __name__ == "__main__":
    unittest.main()
