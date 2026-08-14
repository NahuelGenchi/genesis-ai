# M6 Multi-Domain Continuation v1

## Objective

Continue the first promoted `micro-2m` Genesis checkpoint on a deterministic, verifier-backed mix of `code`, `math`, and `structured` tasks without using any external model outputs or paid compute.

## Parent

The experiment starts from `checkpoints/genesis-micro-2m-v1.pt`, the promotion-gated Genesis checkpoint. It does not restart from random weights and does not import third-party model weights.

## Curriculum

- 1,365 unique examples per domain; 4,095 total.
- Difficulty 1 for the first breadth transfer experiment.
- Oracle answers are derived only from deterministic local verifiers.
- Every frozen `m6-domain-selection-v2` holdout prompt is reconstructed before training and excluded by hash.
- Every target response receives a generated newline termination target.
- 80% procedural / 20% public-text continuation mix.
- 2,000,000 target continuation tokens, $0 cash compute.

## Training

`m6-multidomain-continuation-v1` loads the promoted parent weights, creates a fresh deterministic AdamW optimizer, and trains one target token per exact autoregressive generation context. Every record's first answer token and terminator are mandatory schedule anchors. Remaining unique continuation contexts are biased toward the missing `structured` and `math` skills while retaining code practice.

Two independent CPU continuations must produce semantically identical weights before evaluation can count.

## Promotion

The candidate is compared with its parent on the exact same frozen stop-aware suite.

Blocking gates:

- GCI-v1 relative improvement >= 100%.
- code exact >= 90%.
- math exact >= 50%.
- structured exact >= 50%.
- M3 validation-loss regression <= 2%.
- zero frozen-holdout prompt overlap.
- semantic weight reproduction.
- cash compute = $0.

Rejected candidates publish no checkpoint. Passing candidates may publish `genesis-micro-2m-v2.pt` plus an auditable model card.

## Escalation

If the 2M continuation fails, inspect per-domain exact accuracy and continuation allocation before spending more compute. Increase training budget or alter curriculum balance only through a new versioned Issue/experiment. Do not silently tune against the frozen holdout.
