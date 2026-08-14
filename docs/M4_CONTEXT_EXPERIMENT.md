# M4 context experiment

Issue: #28.

## Question
Can rotary position encoding improve validation loss per estimated training FLOP, and does doubling context to 256 help at the same budget?

## Candidates
- `learned-128` — current learned absolute positions.
- `rotary-128` — same width/depth/context; RoPE on Q/K; no learned position table.
- `rotary-256` — same rotary architecture with 256-token context.

## Controls
- Same approved corpus and `genesis-v0` tokenizer.
- Same seed: 7331.
- Same optimizer/LR.
- Same estimated training-FLOP budget: 500B.
- Target 1,024 training tokens/step.
- 20 deterministic validation batches.

## Decision rule
Lowest final validation loss wins this tiny-scale experiment. Wall time and exact estimated FLOPs are recorded but do not override quality under the fixed budget.

This result is a tiny-scale architecture signal, not evidence that the same ranking holds at frontier scale.
