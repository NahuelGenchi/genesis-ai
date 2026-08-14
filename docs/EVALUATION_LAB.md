# Evaluation lab

## Rule
A model is compared only with the same versioned suite and the same data manifest.

Current suite: `evals/m3-v1.json`.

## M3-v1
1. Exact train/validation contamination check.
2. Validation token loss + perplexity.
3. Three fixed seeded generations.
4. Machine-readable checkpoint/data/suite hashes.
5. Primary comparison metric: validation loss (lower is better).

Exact train/eval text overlap is **blocking**. A failed run writes contamination metadata before exiting.

## Performance benchmark
The Linux self-hosted runner measures:
- training tokens/second;
- decode tokens/second;
- process peak RSS;
- parameter count;
- coarse active FLOPs/token;
- estimated cash compute cost per million tokens.

Cost formula:

`hourly_cost × 1,000,000 / (tokens_per_second × 3,600)`

For the current owned self-hosted runner, hourly cash-compute cost is set to `$0`. This **does not** mean total economic cost is zero: electricity, hardware ownership, networking, storage, and labor are excluded.

## Interpretation
The evaluation lab measures reproducible change. It does not turn a tiny literature model into a capable assistant, and validation loss alone is not a sufficient frontier-quality metric.
