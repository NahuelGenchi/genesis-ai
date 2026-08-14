# M5 checkpoint promotion gate

Issue: #34.

## Rule
N+1 is promoted only when **every blocking gate passes** under `evals/promotion-v1.json`.

Candidate creation is not promotion.

## Required gates
1. **Validation quality** — at least 0.5% lower M3-v1 validation loss.
2. **Contamination** — exact train/validation overlap remains zero.
3. **Decode throughput** — no more than 20% regression.
4. **Training throughput** — no more than 20% regression.
5. **Peak process RSS** — no more than 15% regression.
6. **Parameter count** — unchanged for self-improvement fine-tuning.
7. **Experience learning** — #82 candidate metadata must report lower verified-experience loss.

## Comparability requirements
Promotion fails closed before scoring when:
- parent/candidate evaluation suite version/hash differ;
- data-manifest hashes differ;
- benchmark device/hardware metadata differ;
- evaluation/benchmark checkpoint hashes do not match the supplied files;
- candidate lineage does not name the exact parent checkpoint;
- candidate metadata is not `candidate-training-v1`.

## Decision record
`genesis-promote` emits deterministic JSON containing:
- policy SHA-256;
- parent/candidate checkpoint SHA-256;
- evaluation suite/data identities;
- every gate, observed value, requirement, and pass/fail;
- final `promote|reject` decision;
- deterministic decision SHA-256.

A rejection exits with status `2`; malformed/incomparable inputs raise an error.

## CLI
```bash
genesis-promote \
  --parent checkpoints/genesis-tiny-v0.pt \
  --candidate runs/m5/candidate.pt \
  --parent-eval runs/m5/parent-eval.json \
  --candidate-eval runs/m5/candidate-eval.json \
  --parent-benchmark runs/m5/parent-benchmark.json \
  --candidate-benchmark runs/m5/candidate-benchmark.json \
  --policy evals/promotion-v1.json \
  --output runs/m5/promotion.json
```

Promotion policy changes require a new tracked policy version; historical decisions must not silently inherit new thresholds.
