# GCI-Ladder v1

## Purpose

Prevent saturation of an easy verifier suite from being mistaken for broad reasoning capability.

## Suites

The ladder contains five stop-aware suites with 60 tasks per domain at each difficulty (`900` total tasks):

- difficulty 1: existing `m6-domain-selection-v2`;
- difficulty 2: `m6-domain-ladder-d2-v1`;
- difficulty 3: `m6-domain-ladder-d3-v1`;
- difficulty 4: `m6-domain-ladder-d4-v1`;
- difficulty 5: `m6-domain-ladder-d5-v1`.

Every new suite uses a distinct base/generation seed, deterministic greedy generation, and a required generated newline terminator. The ladder manifest reconstructs all generated tasks and fails closed if **any exact prompt or task overlaps another ladder level**. It stores only hashes/counts, not task content.

## Metric

For each difficulty:

`GCI(d) = mean(code exact, math exact, structured exact) * 100`

`GCI-Ladder` is the harmonic mean of the five difficulty GCIs. If any difficulty GCI is zero, the strict ladder score is zero. This makes hard-task collapse impossible to hide behind a high easy-suite average.

Also report the worst single domain/difficulty exact accuracy.

## Comparison

Before/after ladder comparisons require the exact same suite hashes at every difficulty. If the baseline ladder score is zero, relative percentage improvement is reported as `N/A (zero baseline)` and the absolute point change remains authoritative.

## Training separation

Ladder suite task records are evaluation-only. Curriculum generators may reconstruct prompt hashes solely to exclude overlaps. The autonomous controller consumes aggregate ladder metrics only and cannot read private task/prompt/answer content.

## Frontier claims

Difficulty-1 exact accuracy alone must not be used as evidence of frontier-level intelligence. Model cards should report GCI-Ladder, worst-domain accuracy, M3 general-language loss, reproducibility, compute, and contamination boundaries before making broad capability claims.
