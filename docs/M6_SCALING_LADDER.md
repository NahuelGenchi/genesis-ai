# M6 scaling ladder

Issue: #35.

## Rule
Scale **sequentially** and only after evidence. Fitting in RAM is not permission to skip a stage.

## Measured ladder
Workflow run `31810045132` at source commit `57bde9a4c28baa254d74b3a4cf683bffbffde8a4` measured every v1 stage on `rayzza-linux` with 1,024 tokens/step.

| Stage | Exact params | Train tok/s | Target tokens | Est. target time | Peak RSS |
|---|---:|---:|---:|---:|---:|
| baseline-0.4m | 394,560 | 108,116 | 177,152 | ~1.6 s | 324.9 MB |
| **micro-2m** | **1,895,808** | **36,021** | **2,000,000** | **~55.5 s** | **385.0 MB** |
| small-5m | 4,889,088 | 16,491 | 5,000,000 | ~5.1 min | 496.2 MB |
| medium-12m | 10,872,576 | 8,993 | 10,000,000 | ~18.5 min | 637.1 MB |
| medium-25m | 25,510,912 | 3,900 | 15,000,000 | ~64.1 min | 960.6 MB |

All five stages satisfy the v1 local feasibility limits and require **$0** cash compute.

The timing values are microbenchmark extrapolations from fixed random batches. Real corpus loading, evaluation, checkpointing, and other pipeline work can increase wall-clock time.

## Decision
**Authorized next stage: `micro-2m`.**

`small-5m`, `medium-12m`, and `medium-25m` remain locked even though they fit the machine. `micro-2m` must first pass the scale-promotion gate.

All v1 stages keep the M4 operational defaults: dense FFN, learned positions, context 128. RoPE/MoE are not promoted because their current CPU implementations are slower.

## Local feasibility gate
A planned run is locally feasible only when:
- estimated target training time ≤ **6 hours**;
- measured process peak RSS ≤ **24,576 MB**;
- required cash compute cost = **$0**.

Free external compute may later accelerate an already-authorized stage, but must never create required spend.

## Scale-promotion gate
Stage N+1 does **not** authorize N+2 until N+1:
- improves the selected useful-domain exact accuracy by at least **5 percentage points**;
- keeps M3 validation-loss regression ≤ **2%**;
- has zero blocking exact contamination;
- produces a reproducible checkpoint;
- uses $0 required compute.

#36 will define/freeze the useful-domain evaluation before a scaled checkpoint can pass this gate.

## Measurement record
`research/m6-scaling-ladder-v1.json` records:
- exact parameter/active-parameter counts;
- estimated training FLOPs/token;
- measured training tokens/s;
- target-run time estimate;
- model/gradient/optimizer tensor bytes;
- process peak RSS;
- local feasibility;
- source commit/workflow provenance.

The v1 benchmark workflow is now **manual-only** and verifies the committed record instead of recomputing it. Historical measurements cannot silently change.
