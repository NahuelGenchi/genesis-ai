# M6 generation-position alignment

Issue: #116.

## Evidence entering this investigation

The rejected `micro-2m` candidate had two apparently contradictory measurements on the same frozen code holdout:

- legacy teacher-forced oracle-target loss: `0.0006654457`;
- strict autoregressive exact accuracy: `0/60`;
- oracle token-prefix matches in #111 diagnostic: `0/60`.

Missing answer termination was therefore already ruled out.

## Structural cause

`micro-2m` uses learned absolute position embeddings with context length 128.

Legacy response-only training and `_target_loss_sum` tail-anchor the whole prompt+response sequence into one 129-token window. For long prompts, the first response token is trained/scored at an earlier learned position because the fixed window reserves space for later oracle tokens.

Free generation behaves differently: before every new token, `GenesisLM.generate` keeps the latest 128 tokens and predicts from the final position in that window.

The same answer token is therefore learned/scored at a different absolute position than the one used during generation.

## Frozen diagnostic

Workflow run `31838537349` reconstructed the rejected #97 model under the exact deterministic v2 training trajectory and measured the unchanged 60-task code holdout.

Measured result:

- legacy static oracle-target loss: **0.0006654457**;
- generation-aligned rolling oracle-target loss: **3.9263686956**;
- generation-aligned greedy token accuracy: **409 / 891 = 45.90%**;
- first answer token top-1 correct: **33 / 60 = 55.00%**;
- all oracle tokens greedy-correct under rolling teacher forcing: **0 / 60**;
- tasks with a first-response predictor-position shift: **60 / 60 = 100%**;
- mean first-response predictor shift: **+13.85 learned positions**.

Representative task:

- prompt tokens: 133;
- response tokens: 14;
- legacy first-response predictor position: **114**;
- generation first-response predictor position: **127**;
- shift: **+13** positions;
- legacy static task loss: `0.0005930`;
- generation-aligned rolling task loss: `3.9184855`.

## Conclusion

The position-alignment hypothesis is **confirmed**. The near-zero legacy loss is not representative of the conditioning layout used during free generation.

The frozen v1 evaluator itself remains valid for the primary capability metric because strict free-running exact accuracy already uses the real autoregressive path. What must change is the procedural training layout, not the holdout or promotion threshold.

The next training policy must predict procedural answer tokens from the same rolling last-128-token context used by generation, with the target taken from the final model position. Any replacement candidate must still pass the unchanged frozen exact-accuracy, M3, contamination, reproducibility, and $0 gates.

## Boundaries

- `m6-domain-selection-v1` remains immutable.
- `m6-micro-2m-v1` remains a frozen rejected experiment.
- No task, oracle answer, architecture, seed, or promotion threshold was changed by diagnosis.
- No proprietary/model-generated targets were introduced.
- Required cash compute remained $0.

`research/m6-position-alignment-v1.json` is the accepted diagnostic record. The automatic diagnostic workflow is frozen to a manual-only verifier so this evidence cannot silently rerun or change.
