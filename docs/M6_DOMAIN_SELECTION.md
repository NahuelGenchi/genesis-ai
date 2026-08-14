# M6 useful-domain selection

Issue: #95.

## Purpose
Select #36's first target domain from a frozen baseline measurement rather than preference.

## Suite v1
`evals/m6-domain-selection-v1.json` evaluates the frozen `genesis-tiny-v0` checkpoint on:
- `math` — exact integer addition;
- `structured` — exact JSON ascending sort;
- `code` — restricted integer expressions over `x,y`.

Controls:
- 60 unique tasks per domain;
- difficulty 1 only;
- deterministic disjoint per-domain seeds derived from base seed `36001`;
- greedy generation (`top_k=1`), 32 new tokens;
- deterministic exact verifiers;
- no generated response text committed.

## Metrics
Each domain records:
- exact correct count / accuracy;
- response-set SHA-256;
- task-set SHA-256;
- oracle-set SHA-256;
- token-weighted **oracle-target loss** on the ground-truth answer tokens only.

The model never receives the oracle answer in its prompt. Oracle values exist only inside deterministic evaluation logic.

## Selection rule
1. highest exact accuracy;
2. if tied, lowest oracle-target loss;
3. if still tied, lexicographically smallest domain name.

This rule is fixed before seeing the baseline result.

## Interpretation
Exact accuracy is the capability metric. Oracle-target loss is only the deterministic tie-breaker for choosing the earliest trainable domain when exact accuracy is equally weak.

The committed result will be frozen before #96 generates any training examples, keeping evaluation seeds/data separate from the curriculum.
