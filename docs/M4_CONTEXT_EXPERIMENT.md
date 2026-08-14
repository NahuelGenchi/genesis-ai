# M4 context experiment

Issue: #28.

## Question
Can rotary position encoding improve validation loss per estimated training FLOP, and does doubling context to 256 help at the same budget?

## Controls
Same approved corpus/tokenizer, seed 7331, optimizer/LR, ~500B estimated training FLOPs, ~1,024 tokens/step, and 20 validation batches.

## Results
| Candidate | Val loss | Params | Est. FLOPs | CPU tok/s |
|---|---:|---:|---:|---:|
| learned-128 | 3.59777 | 394,560 | 497.75B | 83,168 |
| rotary-128 | **3.51775** | 382,272 | 498.70B | 64,046 |
| rotary-256 | 3.67201 | 382,272 | 497.96B | 61,457 |

## Decision
- **Quality winner:** rotary-128; ~2.2% lower validation loss than learned-128 at essentially equal estimated FLOPs and ~3.1% fewer parameters.
- **Operational default:** keep learned-128 for now. The current rotary implementation is ~23.0% lower CPU throughput than learned-128.
- Reject rotary-256 at this scale/budget: worse validation loss and lower CPU throughput.

RoPE remains a promising quality hypothesis, but it must be optimized before promotion. This tiny-scale ranking is not assumed to hold at frontier scale.

## Provenance
Final accepted run: workflow `31773193621`, source commit `679c53097e23b5538fb244e6d3ffe9b0a873fd70`.

The validation losses matched the earlier run exactly. Only wall-time/throughput changed. #65 froze this experiment after the final accepted run.
