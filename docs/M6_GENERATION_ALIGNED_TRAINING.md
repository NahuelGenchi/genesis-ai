# M6 generation-aligned micro-2m training

Issue: #121.

## Why this experiment exists

#116 proved that the rejected micro-2m model was trained/scored under a different learned-position layout than real autoregressive generation:

- legacy static oracle loss: `0.0006654457`;
- generation-aligned rolling loss: `3.9263686956`;
- rolling greedy token accuracy: `45.90%`;
- first-answer-token accuracy: `55.00%`;
- mean first-predictor shift: `+13.85` positions.

The frozen free-running exact benchmark is unchanged. This experiment changes training alignment only.

## Procedural dataset v1

For every frozen #96 prompt/response pair and every oracle response token `j`:

1. build `history = prompt + "\nAnswer:" + oracle_response[:j]`;
2. keep exactly the last 128 history tokens;
3. feed those 128 tokens to the unchanged learned-position model;
4. supervise **only** the next oracle token;
5. place that target at index 127, so it is predicted from the final model position exactly like `GenesisLM.generate`.

No padding is invented. The dataset fails closed if any procedural history is shorter than 128 tokens.

The frozen prompts, deterministic oracle responses, tokenizer, holdout, and public-domain corpus are unchanged.

## Controlled variables

Held constant from the rejected #97 v2 experiment:

- architecture: exactly 1,895,808 parameters;
- context: 128;
- learned absolute positions;
- tokenizer;
- random seed: `97001`;
- AdamW / LR / warmup / cosine decay / grad clip;
- single-thread deterministic CPU execution;
- 1,955 steps;
- 2,001,920 processed tokens;
- exact 80% procedural / 20% public-domain step mix;
- frozen 4,096 procedural records;
- frozen public-domain source lock;
- required cash compute: $0.

Changed variable: **procedural response-target positioning only**.

Because each aligned procedural item supervises one answer token, the run records how many of the frozen oracle-token contexts are sampled within the unchanged compute budget.

## Reproducibility

The candidate is trained twice independently. Promotion requires equal:

- model tensors;
- architecture config;
- tokenizer;
- checkpoint step;
- stable metadata.

Serialized checkpoint bytes may differ; model semantics may not.

## Evaluation and promotion

The candidate is evaluated against the unchanged:

- `m6-domain-selection-v1` free-running exact code holdout;
- M3 validation-loss/contamination suite;
- #35 scale-promotion thresholds.

Additional non-promotional diagnostics record generation-aligned rolling loss/token accuracy using the frozen #116 measurement code.

All original blocking gates remain:

- code exact-accuracy gain >= **+5 percentage points**;
- M3 validation-loss regression <= **2%**;
- exact contamination = 0;
- frozen code-holdout prompt overlap = 0;
- exact parameter count = 1,895,808;
- >=2M processed tokens, exact 80/20 mix;
- semantic reproducibility;
- required cash compute = $0.

An additional gate requires training metadata to prove:

- policy `rolling_last_context_predict_final_position`;
- procedural target model position = `127`.

## Publication boundary

Whether accepted or rejected, compact experiment evidence is committed under `research/m6-aligned-micro-2m-v1/`.

Only a full pass may publish:

- `checkpoints/genesis-micro-2m-aligned-v1.pt`;
- `models/genesis-micro-2m-aligned-v1/MODEL_CARD.md`;
- `models/genesis-micro-2m-aligned-v1/metrics.json`.

A rejected candidate remains ephemeral and does not replace the promoted baseline.
