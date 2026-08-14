# genesis-micro-2m-v1

## Status
**Promoted M6 useful-domain checkpoint.** Research model only; not a general assistant.

## Architecture
- Parameters: 1,895,808
- Context: 128
- Width: 192
- Heads: 6
- Layers: 4
- FFN: 768
- Position encoding: learned

## Training
- Policy: `m6-micro-2m-terminated-training-v1`
- Response dataset: `generation-aligned-terminated-response-v1`
- Answer terminator: generated newline
- Processed tokens: 2,001,920
- Steps: 1955
- Mix: 80% procedural / 20% public-domain
- First-response-target coverage: 100%
- Terminator-target coverage: 100%
- Generation-aligned target updates: 12,512
- Seed: 97001
- Required cash compute: $0.00

## Demonstrated capability
Frozen stop-aware domain: **restricted integer-expression synthesis**.

- Strict stopped exact accuracy: 0.00% → **95.00%**
- Absolute gain: **95.00%**
- Candidate termination rate: 100.00%
- Candidate terminated-oracle loss: 0.011801
- Holdout tasks: 60

## General-language regression gate
- M3 validation loss: 3.686833 → 3.355112
- Regression fraction: -9.00%
- Exact contamination overlap: 0

## Curriculum
- Procedural examples: 4,096
- Public-domain documents: 1,082
- Public-domain tokens: 274,653
- Frozen holdout prompt overlap: 0

## Checkpoint
- SHA-256: `0ba16a931451ffbdc369aa685845b0ddb9b6ed03910ac773b50e837fd9886d7e`
- Bytes: 7,604,191

## Limitations
- Useful-domain research checkpoint, not a general-purpose assistant.
- Demonstrated capability is restricted integer-expression synthesis at difficulty 1.
- The answer-stop protocol is a generated newline, not a dedicated learned EOS token.
- General-language training data remains extremely small.
- No preference tuning, tool-use training, safety tuning, or broad factual training.
- Success on the frozen 60-task holdout does not imply broad coding ability.
