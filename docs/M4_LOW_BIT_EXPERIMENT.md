# M4 low-bit inference experiment

Issue: #30.

## Question
Does CPU dynamic INT8 reduce inference storage and/or decode cost without materially damaging validation quality?

## Baseline
Frozen `checkpoints/genesis-tiny-v0.pt`.

## Variant
PyTorch dynamic INT8 quantization of `nn.Linear` layers. Embeddings and unsupported operations stay floating point.

## Controls
Same frozen checkpoint, approved document-disjoint validation corpus, 20 validation batches, identical sampling parameters, and 3 × 256-token decode timing runs.

## Results
| Metric | FP32 | INT8 |
|---|---:|---:|
| Validation loss | 3.686833 | 3.686446 |
| Perplexity | 39.9182 | 39.9028 |
| Serialized state dict | 1,586,699 B | 650,116 B |
| Median decode | 1,108.71 tok/s | 806.61 tok/s |

INT8 quantized 13 Linear modules.

## Tradeoff
- Validation loss change: **-0.0105%** — effectively unchanged at this scale.
- Serialized state size: **59.03% smaller**.
- Decode throughput: **27.25% slower** on this CPU.

## Decision
- **Do not promote dynamic INT8 as the default runtime**: it increases decode time/cash-equivalent compute use on this machine.
- Keep it as a storage-constrained option/hypothesis: quality is preserved and serialized weights are much smaller.
- Future low-bit work should target a backend/format that preserves the size benefit without the measured CPU slowdown.

This is dynamic INT8 CPU inference, not FP8/INT4/FP4. The result is specific to this tiny model, PyTorch `2.13.0+cpu`, and this runner.

## Provenance
Accepted result: `research/m4-low-bit-v1.json`, workflow `31773897730`, source commit `85857e31dcaf25d72ba514302173d9f514dec740`.
