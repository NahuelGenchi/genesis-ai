# GCI-Ladder v1

## Purpose

Prevent saturation of one easy verifier suite from being mistaken for broad reasoning capability, and give the remote autonomous controller frozen higher-difficulty targets without requiring human intervention.

## Suites

The ladder contains five stop-aware suite files with 60 generated tasks per domain at each difficulty (`900` generated evaluations total):

- difficulty 1: existing frozen `evals/m6-domain-selection-v2.json`;
- difficulty 2: `evals/m6-domain-ladder-d2-v1.json`;
- difficulty 3: `evals/m6-domain-ladder-d3-v1.json`;
- difficulty 4: `evals/m6-domain-ladder-d4-v1.json`;
- difficulty 5: `evals/m6-domain-ladder-d5-v1.json`.

All five files use the already-frozen stop-aware evaluation protocol `suite_version: m6-domain-selection-v2`. Difficulty, suite-file SHA-256, seeds, and generation budget identify each ladder level. This deliberately avoids changing evaluator semantics merely to add harder frozen inputs.

Difficulties 2–5 use distinct base/generation seeds, deterministic greedy generation, and a required generated newline terminator.

## Contamination boundary

The manifest reconstructs prompt/task hashes for each level and fails closed if a unique prompt or task from one difficulty appears at another difficulty.

The existing difficulty-1 suite is immutable and may contain repeated generated identities inside its own level. Those repeats are reported explicitly as generated count, unique count, and duplicate count; they are **not** treated as cross-difficulty contamination. Changing difficulty-1 tasks to force intra-suite uniqueness would break the frozen benchmark contract.

The manifest stores hashes and aggregate counts only, not holdout task content.

## Metric

For each difficulty:

`GCI(d) = mean(code exact, math exact, structured exact) * 100`

`GCI-Ladder` is the harmonic mean of the five difficulty GCIs. If any difficulty GCI is zero, the strict ladder score is zero. Also report the worst single domain/difficulty exact accuracy.

Before/after ladder comparisons require identical suite hashes at every difficulty. If the baseline ladder score is zero, relative percentage improvement is `N/A (zero baseline)` and absolute point change remains authoritative.

## Autonomous integration

`autonomous-improvement.yml` already maps controller difficulties 1–5 to these exact files. When all domains at the current level reach the controller's mastery threshold, the next scheduled cycle can resolve the next frozen suite, establish an incumbent baseline on it, and continue without editing workflow code or requiring a personal computer.

Curriculum generation may reconstruct target-suite prompt hashes only for exclusion. The controller itself consumes aggregate measurements, not private task/prompt/answer content.

## Frontier claims

Difficulty-1 exact accuracy alone is not evidence of frontier-level capability. Broad model claims should report the ladder, worst-domain accuracy, M3 general-language loss, reproducibility, compute, and contamination boundaries.
