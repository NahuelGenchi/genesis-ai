# M5 verified experience pipeline

Issue: #33.

## Rule
Only responses produced by a Genesis checkpoint may enter candidate experience.

No external AI API, oracle answer, verifier expected value, or corrected answer may be substituted for the model response.

## Outputs
- `accepted.jsonl`: prompt + exact Genesis response + score + provenance.
- `audit.jsonl`: every attempt, including rejected answers and verifier reason.
- `manifest.json`: policy, checkpoint provenance, counts, acceptance rate, file hashes.

Hidden verifier payloads are excluded from `accepted.jsonl`.

## Policy
Default `min_score = 1.0`.

The threshold is configurable in `[0,1]`; every bundle records the exact threshold and policy version.

## Producer
`CheckpointProducer` loads one local Genesis checkpoint, records its SHA-256/step, and generates deterministic responses when `top_k=1` using per-task seeds.

## Safety
- Genesis checkpoints only.
- Deterministic verifiers only.
- No arbitrary code execution.
- Duplicate task IDs rejected.
- Accepted and audit files are SHA-256 locked by the manifest.

## CLI
```bash
genesis-experience \
  --tasks runs/m5/tasks.jsonl \
  --checkpoint checkpoints/genesis-tiny-v0.pt \
  --output-dir runs/m5/experience \
  --min-score 1.0 \
  --top-k 1
```

Zero accepted examples is a valid result. The pipeline must never manufacture a passing answer to force candidate training data.
