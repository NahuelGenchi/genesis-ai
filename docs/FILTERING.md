# Corpus filtering

## Current pipeline
1. Verify input shard size + SHA-256.
2. Reject malformed documents.
3. Apply conservative quality checks.
4. Remove canonical exact duplicates.
5. Write deterministic filtered shards.
6. Write deterministic `metrics.json`.

## Quality checks
- minimum character count;
- NUL bytes;
- excessive control characters;
- very long repeated-character runs;
- heavily repeated lines.

These are intentionally conservative to avoid deleting valid code, math, or multilingual text.

## Exact dedup
Fingerprint = SHA-256 of NFKC-normalized text with whitespace collapsed.

## Near-duplicate plan
Not enabled yet.

Next experiment:
1. benchmark SimHash/MinHash candidates on labeled document pairs;
2. measure false-positive rate and memory cost;
3. choose a threshold only if quality-per-compute improves;
4. add it behind a documented configuration gate.

## Run
```bash
genesis-filter data/shards data/filtered
```

Every run reports kept/dropped counts and reasons.
