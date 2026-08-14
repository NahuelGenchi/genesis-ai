# M6 micro-2m training and gate

Issue: #97.

## Authorized stage
Only `micro-2m` is authorized by the frozen #35 ladder.

Architecture:
- vocabulary: 512;
- context: 128;
- width: 192;
- heads: 6;
- layers: 4;
- FFN: 768;
- dense FFN;
- learned positions;
- exact parameters: **1,895,808**.

Larger M6 stages remain locked regardless of local feasibility.

## Training v1
- Random initialization; no inherited model weights.
- Seed: `97001`.
- AdamW.
- Base LR: `1e-3`.
- 50-step linear warmup.
- Cosine decay to `1e-4`.
- Gradient clipping: 1.0.
- Batch: 8 × 128 = 1,024 processed tokens/step.
- Exact schedule: 4 procedural steps, then 1 public-text step.
- 391 complete cycles = **1,955 steps**.
- Processed tokens: **2,001,920** (1,920 above the 2M target; +0.096%).
- Mix: exactly **80% procedural / 20% public-domain** by training steps and processed tokens.
- Required cash compute: $0.

Procedural batches use response-only loss and the frozen #96 left-prefix truncation policy. Public-domain batches use standard next-token loss.

## Reproducibility
The one-shot run trains the model **twice independently** with the same frozen inputs and seed.

Promotion requires semantic equality of:
- all model tensors;
- architecture config;
- tokenizer;
- checkpoint step;
- stable checkpoint metadata.

Byte-identical checkpoint files are recorded but are not required because serialization details are not model semantics.

## Scale-promotion gate
The candidate is promoted only if every gate passes:
1. frozen code exact accuracy gains at least **+5 percentage points** over #95;
2. M3 validation-loss regression is at most **2%**;
3. M3 exact contamination overlap is zero;
4. frozen #96 code-holdout prompt overlap remains zero;
5. parameter count is exactly 1,895,808;
6. at least 2M tokens are processed under the exact 80/20 schedule;
7. independent training is reproducible;
8. required cash compute remains $0.

Candidate creation is not promotion.

## Publication
The compact gate result is committed whether the candidate passes or fails.

Only if all gates pass may the workflow commit:
- `checkpoints/genesis-micro-2m-v1.pt`;
- `models/genesis-micro-2m-v1/MODEL_CARD.md`;
- `models/genesis-micro-2m-v1/metrics.json`.

A rejected candidate stays ephemeral and does not replace the current promoted model.
