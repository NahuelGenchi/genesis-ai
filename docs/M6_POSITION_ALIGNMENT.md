# M6 generation-position alignment

Issue: #116.

## Evidence entering this investigation

The rejected `micro-2m` candidate has two apparently contradictory measurements on the same frozen code holdout:

- legacy teacher-forced oracle-target loss: `0.0006654457`;
- strict autoregressive exact accuracy: `0/60`;
- oracle token-prefix matches in #111 diagnostic: `0/60`.

Therefore missing answer termination is not the cause.

## Hypothesis

`micro-2m` uses learned absolute position embeddings with context length 128.

Legacy response-only training and `_target_loss_sum` tail-anchor the whole prompt+response sequence into one 129-token window. For long prompts, the first response token is therefore trained/scored at an earlier learned position because space is reserved for later oracle tokens.

Free generation behaves differently: before every new token, `GenesisLM.generate` keeps the latest 128 tokens and predicts from the final position in that window.

If this positional mismatch is material, a model can achieve very low legacy teacher-forced loss without producing the same tokens autoregressively.

## Diagnostic contract

The frozen #97 candidate is reconstructed with exactly the accepted `m6-micro-2m-training-v2` trajectory. No checkpoint is published.

On the unchanged 60-task code holdout, record:

- legacy static oracle-target loss;
- generation-aligned rolling teacher-forced loss;
- greedy token accuracy under rolling contexts;
- first-token correctness;
- number of tasks whose first-response predictor position differs between static scoring and generation;
- representative position shifts.

The rolling score builds the exact context used by generation for each next oracle token: prompt + previously supplied oracle tokens, truncated to the latest 128 tokens, with the prediction taken from the last model position.

## Boundaries

- `m6-domain-selection-v1` remains immutable.
- `m6-micro-2m-v1` remains a frozen rejected experiment.
- No task, oracle answer, training target, architecture, seed, or promotion threshold changes during diagnosis.
- No proprietary/model-generated targets are introduced.
- Required cash compute remains $0.

A training fix is allowed only after measurements prove the mismatch materially explains the static-loss/free-generation contradiction.
