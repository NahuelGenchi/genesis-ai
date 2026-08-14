# Bootstrap tokenizer corpus

## Purpose
Provide a small, human-authored, multilingual corpus for the first tokenizer only.

## Sources
- Project Gutenberg #1342 — Jane Austen — English.
- Project Gutenberg #2000 — Miguel de Cervantes — Spanish.
- Project Gutenberg #17489 — Victor Hugo — French.

The catalog records canonical download and rights pages. Project Gutenberg currently marks these eBooks public domain in the USA and tells users outside the USA to check local law.

## Safety / provenance
- No AI-generated text.
- No raw books committed.
- Raw redistribution disabled.
- Full upstream SHA-256 and sampled SHA-256 locked after first run.
- Header/license wrapper removed before training.
- Deterministic systematic paragraph sample: max 180 KB/source.
- Corpus goes through `genesis-ingest` and `genesis-filter` before tokenizer training.

## Output
Only these generated records may be committed:
- `data/bootstrap-tokenizer-lock.json`
- `tokenizers/genesis-v0.json`

The tokenizer artifact records compression, round-trip, corpus, catalog, filter, and source-lock hashes.
