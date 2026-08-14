# Architecture experiment harness

## Rule
Architecture changes compete under the same estimated training-compute budget, corpus, tokenizer, seed, optimizer, and validation procedure.

## Estimators
### `dense-training-v1`
Used by the frozen #28 context experiment:

`6 × total parameters + 12 × layers × context × width` FLOPs/token.

### `active-training-v2`
Used from #29 onward:

`6 × estimated active parameters + 12 × layers × context × width` FLOPs/token.

For sparse MoE, inactive expert weights are excluded while router parameters remain active.

These are coarse **relative experiment budgets**, not hardware FLOP measurements.

## Harness
For each candidate:
1. instantiate the architecture;
2. choose batch size near target tokens/step;
3. choose the largest integer step count within budget;
4. train from random weights with the same seed;
5. record initial/final validation loss, tokens, steps, estimated compute, wall time, throughput, and active/total parameters;
6. record routing metrics when the model is sparse.

## Comparison
Lowest final validation loss wins only within the same experiment definition/data/tokenizer. Routing collapse can invalidate an MoE result even if loss improves.

Equal estimated active compute is preferred over equal total parameter count or wall time because architecture changes alter active work and implementation speed.

## Limitation
The estimator is intentionally simple. M4 decisions are hypotheses for later validation, not claims about frontier-scale efficiency.
