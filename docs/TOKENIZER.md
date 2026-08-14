# Tokenizer

## Design
Deterministic byte-level BPE trained from scratch.

- Base vocabulary: all 256 byte values.
- Pretokenization: contiguous whitespace / non-whitespace spans.
- Merges: highest weighted adjacent-pair frequency.
- Ties: smallest numeric pair wins.
- Default target: 1,024 tokens.
- No external tokenizer library.

The byte base guarantees arbitrary UTF-8 text can be represented. Every training run verifies encode → decode on the complete filtered input.

## Input gate
Training reads only integrity-checked filtered shards produced by `genesis-filter`.

The artifact records:
- input manifest SHA-256;
- filter-metrics SHA-256 when available;
- documents / UTF-8 bytes / tokens;
- bytes per token;
- reduction versus byte baseline;
- requested and actual vocabulary size;
- merge count;
- round-trip failures.

## Run
```bash
genesis-tokenizer-train data/filtered tokenizers/genesis-v0.json --vocab-size 1024
```

A tokenizer is not considered project-ready until its training corpus is independently approved and tracked.
