# M6 scaling ladder

Issue: #35.

## Rule
Scale **sequentially** and only after evidence. Fitting in RAM is not permission to skip a stage.

## Ladder
| Stage | Shape | Target training tokens |
|---|---|---:|
| baseline-0.4m | 96d × 3L × 384ff | 177,152 reference |
| micro-2m | 192d × 4L × 768ff | 2,000,000 |
| small-5m | 256d × 6L × 1024ff | 5,000,000 |
| medium-12m | 384d × 6L × 1536ff | 10,000,000 |
| medium-25m | 512d × 8L × 2048ff | 15,000,000 |

All v1 stages keep the M4 operational defaults: dense FFN, learned positions, context 128. RoPE/MoE are not promoted because their current CPU implementations are slower.

Exact parameter counts come from the runner microbenchmark, not stage names.

## Local feasibility gate
Each stage is measured on the self-hosted CPU with the same 1,024 tokens/step target.

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

## Measurement
`experiments/m6-scaling-ladder-v1.json` defines the ladder. The runner records:
- exact parameter/active-parameter counts;
- estimated training FLOPs/token;
- measured training tokens/s;
- target-run time estimate;
- model/gradient/optimizer tensor bytes;
- process peak RSS;
- local feasibility.

The first feasible candidate after the reference becomes `next_stage`; larger feasible stages remain locked.
