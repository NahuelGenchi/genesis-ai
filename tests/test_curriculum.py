import json
import tempfile
import unittest
from pathlib import Path

from genesis_ai.curriculum import build_curriculum, load_spec
from genesis_ai.ingest import sha256_file
from genesis_ai.tokenizer import ByteBPETokenizer


class CurriculumTest(unittest.TestCase):
    def _filtered_public_fixture(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        documents = [
            {
                "id": "public:00000000",
                "text": "A deterministic public-domain paragraph used only as a small unit-test fixture for language mixing.",
            },
            {
                "id": "public:00000001",
                "text": "A second deterministic paragraph ensures the public-text summary covers more than one document.",
            },
        ]
        shard = root / "shard-00000.jsonl"
        with shard.open("w", encoding="utf-8", newline="\n") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        manifest = {
            "format_version": "1.0",
            "documents": len(documents),
            "shards": [
                {
                    "file": shard.name,
                    "documents": len(documents),
                    "sha256": sha256_file(shard),
                    "size_bytes": shard.stat().st_size,
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return root

    def _small_spec(self, root: Path) -> Path:
        spec = json.loads(Path("experiments/m6-code-curriculum-v1.json").read_text(encoding="utf-8"))
        spec["procedural_examples"] = 32
        path = root / "spec.json"
        path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_full_build_is_deterministic_and_excludes_holdout_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_data = self._filtered_public_fixture(root / "public")
            spec = self._small_spec(root)
            first_records = root / "records-a.jsonl"
            second_records = root / "records-b.jsonl"
            first = build_curriculum(
                spec_path=spec,
                selection_result_path="research/m6-domain-selection-v1.json",
                evaluation_suite_path="evals/m6-domain-selection-v1.json",
                tokenizer_path="tokenizers/genesis-v0.json",
                public_data=public_data,
                public_source_lock="data/bootstrap-tokenizer-lock.json",
                public_source_catalog="data/bootstrap-tokenizer-sources.json",
                records_path=first_records,
            )
            second = build_curriculum(
                spec_path=spec,
                selection_result_path="research/m6-domain-selection-v1.json",
                evaluation_suite_path="evals/m6-domain-selection-v1.json",
                tokenizer_path="tokenizers/genesis-v0.json",
                public_data=public_data,
                public_source_lock="data/bootstrap-tokenizer-lock.json",
                public_source_catalog="data/bootstrap-tokenizer-sources.json",
                records_path=second_records,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["selected_domain"], "code")
            procedural = first["training"]["procedural"]
            self.assertEqual(procedural["examples"], 32)
            self.assertGreater(procedural["prefix_truncated_examples"], 0)
            self.assertEqual(procedural["truncation_policy"], "left_prefix_only_preserve_all_response_tokens")
            self.assertLessEqual(procedural["max_training_window_tokens"], 129)
            self.assertEqual(first["evaluation_separation"]["exact_prompt_overlap_count"], 0)
            self.assertEqual(first["training"]["procedural_batch_fraction"], 0.8)
            self.assertEqual(first["training"]["public_text_batch_fraction"], 0.2)
            self.assertEqual(first["cash_compute_cost_usd"], 0.0)
            self.assertEqual(first_records.read_bytes(), second_records.read_bytes())

            records = [json.loads(line) for line in first_records.read_text(encoding="utf-8").splitlines()]
            tokenizer = ByteBPETokenizer.load(Path("tokenizers/genesis-v0.json"))
            self.assertEqual(len(records), 32)
            self.assertTrue(all(record["provenance"]["kind"] == "procedural_oracle" for record in records))
            self.assertTrue(all("verifier" not in record for record in records))
            self.assertTrue(all(len(tokenizer.encode(record["response"])) <= 128 for record in records))

    def test_seed_collision_and_bad_mix_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = json.loads(Path("experiments/m6-code-curriculum-v1.json").read_text(encoding="utf-8"))
            spec["training_seed"] = spec["evaluation_base_seed"]
            path = root / "collision.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_spec(path)

            spec["training_seed"] = 46001
            spec["procedural_batch_fraction"] = 0.9
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_spec(path)


if __name__ == "__main__":
    unittest.main()
