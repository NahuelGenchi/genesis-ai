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

## Measured baseline
Workflow run `31810825418` at source commit `067c0fdb627f315cf881624e1c79474e342108b3` measured the frozen checkpoint SHA-256 `4db01f8239cfc28933bc3152b7b698ffe089491ce1acf5b66de8adc53b9f8ed9`.

| Domain | Exact | Accuracy | Oracle-target loss |
|---|---:|---:|---:|
| **code** | **0 / 60** | **0%** | **6.620002** |
| structured | 0 / 60 | 0% | 7.916341 |
| math | 0 / 60 | 0% | 9.135551 |

All three domains tie at zero exact capability. The predeclared tie-breaker therefore selects the lowest oracle-target loss.

## Decision
**Selected M6 target domain: `code` — restricted integer-expression synthesis.**

#96 may now create deterministic procedural training examples for this domain using seeds that are disjoint from the frozen selection suite.

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

This rule was fixed before seeing the baseline result.

## Frozen
The v1 selection workflow is now **manual-only** and verifies `research/m6-domain-selection-v1.json` instead of recomputing it. Evaluation tasks/seeds are frozen before any #96 curriculum is generated.
