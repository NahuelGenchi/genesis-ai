# M4 low-bit inference experiment

Issue: #30.

## Question
Does CPU dynamic INT8 reduce inference storage and/or decode cost without materially damaging validation quality?

## Baseline
Frozen `checkpoints/genesis-tiny-v0.pt`.

## Variant
PyTorch dynamic INT8 quantization of `nn.Linear` layers. Embeddings and unsupported operations stay floating point.

## Controls
- Same frozen checkpoint weights.
- Same approved document-disjoint validation corpus.
- 20 validation batches.
- Same prompt/sampling parameters.
- 256 generated tokens, 3 timing repetitions; median throughput reported.
- FP32 and INT8 serialized state-dict sizes measured separately.

## Metrics
- validation loss / perplexity;
- loss regression percent;
- serialized state-dict size reduction;
- median decode tokens/s and speedup;
- number of quantized Linear modules;
- output sample hash.

## Interpretation
This is **dynamic INT8 CPU inference**, not FP8/INT4/FP4. It is a zero-dependency first low-bit measurement. Promotion requires measured benefit; smaller files alone do not justify a slower or materially worse runtime.
