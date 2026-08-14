# M4 sparse MoE experiment

Issue: #29.

## Question
Can sparse experts add total model capacity without increasing estimated active training compute per token?

## Controls
Same approved corpus/tokenizer, seed 8112, optimizer/LR, ~500B estimated active training FLOPs, ~1,024 tokens/step, and 20 validation batches.

## Results
| Candidate | Val loss | Total params | Active params | Est. FLOPs | CPU tok/s |
|---|---:|---:|---:|---:|---:|
| dense-384 | 3.65190 | 394,560 | 394,560 | 497.75B | 110,220 |
| moe4-top2-192 | 3.60818 | 616,896 | 395,712 | 498.97B | 68,431 |
| moe8-top2-192 | **3.56403** | 1,060,416 | 396,864 | 497.31B | 33,400 |

## Routing
All MoE layers used **100% of experts**.

- 4-expert model: worst layer max share 31.9%; minimum share 18.5%.
- 8-expert model: worst layer max share 21.0%; minimum share 6.6%.

No routing-collapse gate fired.

## Decision
- **Quality winner:** `moe8-top2-192`; ~2.4% lower validation loss than dense at essentially equal estimated active compute.
- `moe4-top2-192` also improves loss by ~1.2%.
- Total capacity rises substantially while active parameters stay within ~0.6% of dense.
- **Operational default remains dense**: the current Python sparse dispatch is ~38% slower for 4 experts and ~70% slower for 8 experts on this CPU runner.
- Keep sparse MoE as a strong architecture hypothesis; optimize dispatch before promotion.

The result shows useful extra capacity-per-estimated-active-FLOP at tiny scale. It does not prove the same gain at frontier scale, and the active-FLOP estimator does not capture routing/dispatch overhead.

## Provenance
Accepted result: `research/m4-moe-v1.json` from workflow `31773547158`, source commit `544c3c477828b3d72dd288b535446cc7b9613fb4`.
