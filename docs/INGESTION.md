# Corpus ingestion

## Rule
Only local sources with approved provenance metadata enter training shards.

Metadata follows `schemas/dataset-source.schema.json`.

## Manifest
A JSON array. Each item has exactly:

```json
{"metadata":"source.json","local_path":"source.txt"}
```

Paths are relative to the manifest.

## Run

```bash
genesis-ingest data/manifest.json data/shards
```

Optional shard size:

```bash
genesis-ingest data/manifest.json data/shards --docs-per-shard 5000
```

## Output
- deterministic `shard-00000.jsonl` files;
- deterministic `manifest.json` with shard hashes and counts.

## Hard failures
- malformed manifest/metadata;
- training permission not explicitly `true`;
- missing source file;
- size or SHA-256 mismatch;
- invalid UTF-8;
- zero output documents.

Raw corpora stay outside Git unless redistribution is explicitly approved.
