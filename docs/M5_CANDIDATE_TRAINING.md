# M5 candidate training

Issue: #82.

## Boundary
Candidate N+1 may train only from an integrity-checked #33 experience bundle produced by the exact parent Genesis checkpoint.

## Fail closed
Training stops when:
- `accepted.jsonl` or `audit.jsonl` hash differs from the manifest;
- manifest/audit counts disagree;
- accepted/audit IDs disagree;
- experience score policy is below the required threshold;
- producer checkpoint SHA-256 differs from the parent checkpoint;
- accepted experience is below `--min-accepted`;
- resume provenance differs from the parent or experience bundle.

## Objective
Only verified **response tokens** receive loss. Prompt tokens are context and use target `-100` (`cross_entropy` ignore index).

No verifier oracle/expected values are training targets.

## Lineage
Every candidate checkpoint records:
- candidate-training policy version;
- parent checkpoint SHA-256 + step;
- experience manifest SHA-256 + file hashes;
- experience acceptance policy/count;
- seed, batch size, learning rate and steps;
- supervised response-token count;
- experience loss before/after.

The parent checkpoint is SHA-256 checked again after training and is never overwritten.

## Resume
Candidate checkpoints store optimizer + RNG + batch-generator state. With identical seed/budget/config, interrupted/resumed training must match uninterrupted training exactly.

## CLI
```bash
genesis-candidate \
  --parent checkpoints/genesis-tiny-v0.pt \
  --experience runs/m5/experience \
  --checkpoint runs/m5/candidate.pt \
  --steps 50 \
  --min-accepted 16 \
  --required-min-score 1.0
```

Candidate creation does **not** imply promotion. #34 decides whether N+1 may replace N.
