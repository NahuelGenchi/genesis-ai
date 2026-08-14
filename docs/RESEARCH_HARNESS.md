# Architecture experiment harness

## Rule
Architecture changes compete under the same estimated training-FLOP budget, corpus, tokenizer, seed, optimizer, and validation procedure.

## Budget
Estimator `dense-training-v1`:

`6 × parameters + 12 × layers × context × width` FLOPs/token.

This is a coarse **relative experiment budget**, not a hardware FLOP measurement.

For each candidate the harness:
1. instantiates the requested architecture;
2. chooses batch size near the target tokens/step;
3. chooses the largest integer step count within the FLOP budget;
4. trains from random weights using the same seed;
5. records initial/final validation loss, tokens, steps, estimated FLOPs, wall time, and throughput.

## Comparison
Lowest final validation loss wins only within the same experiment definition/data/tokenizer.

Equal estimated FLOPs are preferred over equal parameter count or equal wall time because architecture changes alter both active compute and CPU implementation speed.

## Limitation
The estimator is intentionally simple. M4 decisions are hypotheses for later validation, not claims about frontier-scale efficiency.
