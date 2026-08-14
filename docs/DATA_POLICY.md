# Data Policy

## Rule

Every training source must have known provenance and a documented right-to-use basis **before ingestion**.

Approved source metadata follows [`schemas/dataset-source.schema.json`](../schemas/dataset-source.schema.json).

A source is eligible only when:

- its rights basis is explicit;
- `training_allowed` is `true`;
- the retrieved artifact has a SHA-256 checksum;
- retrieval time, size, languages, domains, formats, and filtering steps are recorded.

## Never commit

- raw copyrighted corpora without redistribution rights;
- secrets or personal data;
- scraped private data;
- proprietary model outputs used as training data.

Raw training data stays outside Git unless it is redistributable and intentionally approved by an Issue.

`data/source.example.json` is metadata-only and is **not** an approved training corpus.
