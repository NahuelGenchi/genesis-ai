# M2 model pipeline

## Input
Only filtered corpus shards plus a committed Genesis tokenizer artifact.

`ByteDataset` is retired from M2. Documents are deterministically split by SHA-256 of document ID, so train/validation documents do not overlap.

## Training
- Random weight initialization for new runs.
- Tokenizer vocabulary determines model vocabulary.
- Default tiny baseline: context 128, width 96, 4 heads, 3 layers, FFN 384.
- Seeded batch-position generator.
- Fixed probe batch measures learning before/after.
- Training checkpoint keeps model, optimizer, step, tokenizer, and RNG states.
- `--resume` continues from the stored step and RNG state.
- Optional inference export removes optimizer/RNG state.

## Generation
Checkpoint-embedded tokenizer is mandatory. Seed, temperature, and top-k are explicit.

## Evaluation
Uses the checkpoint tokenizer and the deterministic validation document split.

## Rule
M2 is an end-to-end pipeline baseline, not a claim of useful language-model quality.
