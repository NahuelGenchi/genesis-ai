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

## Training v2
- Random initialization; no inherited model weights.
- Seed: `97001`.
- CPU-only reproducibility contract.
- Torch intra-op threads: **1**.
- OpenMP/MKL/OpenBLAS/NumExpr/vecLib worker limits: **1** in the runner environment.
- PyTorch deterministic algorithms required.
- AdamW with `foreach=False`, `fused=False`.
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

### Why v2 exists
The first full v1 attempt (`31835552262`) trained both models successfully but failed independent reproduction:
- config/tokenizer/step matched;
- **weights differed**;
- stable metadata therefore differed too.

Both runs began with the same step-1 loss (`6.245032`) and then numerically diverged under the runner's multi-threaded CPU execution. v2 removes that source of reduction/update-order variation rather than weakening the equality requirement.

## Reproducibility
The accepted v2 experiment (`31836252107`) trained the model twice independently under the frozen one-thread contract.

Semantic reproduction passed:
- weights equal;
- config equal;
- tokenizer equal;
- checkpoint step equal;
- stable metadata equal.

Serialized checkpoint bytes differed, which is explicitly non-semantic and does not weaken the tensor-equality requirement.

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

## Measured v1 experiment result
The v2-trained candidate **was rejected** by the frozen scale-promotion gate.

Capability:
- frozen code exact accuracy: **0/60 → 0/60**;
- exact-accuracy gain: **+0.00 percentage points**;
- code oracle-target loss: `6.6200023013 → 0.0006654457`;
- procedural probe loss: `6.2664729804 → 0.0006633719`.

General-language evaluation:
- M3 validation loss: `3.6868329406 → 3.2608576655`;
- relative change: **-11.55%** (improvement);
- exact contamination overlap: **0**.

All gates passed except `code_exact_accuracy`. The candidate therefore did **not** replace the promoted baseline.

The near-zero code oracle-target loss paired with 0/60 strict generation is diagnostic evidence, not a promotion metric. Issue #111 investigates the answer-termination/evaluation interface without altering this historical result.

## Publication
Because the gate rejected the candidate:
- `checkpoints/genesis-micro-2m-v1.pt` was **not** published;
- no micro-2m model card or metrics package was published;
- the candidate weights remained ephemeral;
- the complete compact experiment measurements were committed under `research/m6-micro-2m-v1/`.

The automatic training workflow is now frozen to a manual-only verifier of this historical rejected experiment so later evaluator/trainer changes cannot silently overwrite it.
