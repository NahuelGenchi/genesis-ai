# M6 generation-aligned training

Issue: #115.

## Evidence

#116 proved the rejected #97 model was trained/scored under a response window that did not match free generation:

- legacy static loss: `0.0006654457`;
- generation-aligned rolling loss: `3.9263686956`;
- rolling token accuracy: `45.90%`;
- first-token accuracy: `55%`;
- full oracle sequences: `0/60`;
- learned-position/context shift: `60/60` tasks, mean `13.85` positions.

## Training policy

`m6-micro-2m-aligned-training-v1` keeps the accepted #97 experiment fixed except for procedural response-window construction/sampling.

Frozen:
- random initialization seed `97001`;
- 1,895,808 parameters;
- context 128;
- learned absolute positions;
- AdamW/LR/warmup/gradient clipping;
- deterministic one-thread CPU execution;
- 1,955 steps / 2,001,920 processed tokens;
- exact 80% procedural / 20% public schedule;
- #96 tokenizer, corpus, curriculum, provenance and holdout separation;
- $0 required cash compute.

## Generation-aligned procedural dataset

Each oracle response token becomes one training target.

For response token `t`:
1. history is `prompt + previously supplied oracle response tokens`;
2. keep exactly the latest 128 history tokens, matching `GenesisLM.generate`;
3. place that context at the same learned position IDs generation uses;
4. supervise only `t` at the predictor position;
5. append `t` to teacher-forced history for the next target.

If a context is shorter than 128, padding occurs only **after** the predictor. Causal attention therefore cannot let padding affect the supervised prediction.

The frozen curriculum contains 61,559 response targets. The 2M compute budget permits 12,512 procedural target updates. The deterministic schedule:
- covers all 4,096 first-response targets exactly once;
- selects 8,416 unique continuation targets;
- contains no duplicate target contexts;
- is seed/hash bound.

## Promotion

The aligned candidate must still pass every blocking gate:
- strict frozen code exact accuracy gain >= 5 percentage points;
- M3 validation-loss regression <= 2%;
- zero exact contamination;
- zero code-holdout prompt overlap;
- exact parameter count;
- frozen compute/mix contract;
- generation-alignment metadata contract;
- independent semantic reproducibility;
- $0 required cash compute.

Rolling teacher-forced metrics are diagnostic only. They cannot substitute for strict autoregressive exact accuracy.

Only a fully promoted candidate may publish `checkpoints/genesis-micro-2m-v1.pt` and its model card.
