# M4 sparse MoE experiment

Issue: #29.

## Question
Can sparse experts add total model capacity without increasing estimated active training compute per token?

## Candidates
- `dense-384` — dense FFN width 384.
- `moe4-top2-192` — 4 experts, 2 active, width 192/expert.
- `moe8-top2-192` — 8 experts, 2 active, width 192/expert.

Top-2 × 192 keeps active expert FFN width comparable to dense 384. Router parameters remain active and are counted.

## Routing
Each token selects top-k experts from a learned softmax router. Selected weights are renormalized. A small load-balancing auxiliary loss (`0.01`) discourages collapse.

Every run records per-layer expert assignment counts, fractions, utilization, max load, and min load.

## Compute accounting
From #29 onward, research uses `active-training-v2`:

`6 × estimated active parameters + 12 × layers × context × width` FLOPs/token.

For MoE, inactive expert weights are excluded from the active-parameter estimate. This is a relative research estimator; actual CPU throughput remains separately measured.

## Controls
Same approved corpus/tokenizer, seed 8112, optimizer/LR, ~500B estimated active training FLOPs, ~1,024 tokens/step, and 20 validation batches.

## Decision rule
Quality improvement is only credible if routing is non-collapsed and the active-compute budget is matched. Real CPU throughput is reported separately because sparse dispatch overhead can dominate at this scale.
