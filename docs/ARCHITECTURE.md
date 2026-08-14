# Architecture

## Current path

`data -> bytes/tokenizer -> causal LM -> checkpoint -> eval -> candidate promotion`

## Planned path

`data + self-play -> filtering/verifiers -> training -> candidate -> eval gates -> promotion`

## Design priorities

1. Quality per FLOP.
2. Data quality.
3. Reproducibility.
4. Small experiments before scaling.
5. Efficient inference.

## Current baseline

- Framework: PyTorch.
- Tokenization: byte-level bootstrap baseline.
- Model: decoder-only causal Transformer.
- Initialization: random.
- Training: next-token prediction.
- Checkpoints: local only; ignored by Git.

This baseline exists to validate the pipeline, not to define the final architecture.
