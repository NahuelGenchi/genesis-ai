# Autonomous Improvement Controller v1

## Purpose

Choose and execute the next Genesis training target automatically from aggregate capability measurements while keeping private holdout content, incumbent weights, and promotion authority outside the controller.

## Input boundary

The controller may consume only:

- frozen suite identity/version;
- current difficulty;
- aggregate per-domain exact accuracy;
- aggregate per-domain terminated oracle loss;
- aggregate termination rate;
- incumbent checkpoint SHA-256.

Prompt text, task bodies, oracle answers, generated responses, and other holdout content are rejected by the controller API.

## Decision

The controller deterministically chooses the weakest domain by:

1. lowest exact accuracy;
2. highest aggregate oracle loss when accuracy ties;
3. domain name as the final deterministic tie-breaker.

When all domains reach at least 80% strict exact accuracy, the controller raises difficulty instead of repeatedly polishing the same easy suite. Difficulty escalation requires a new frozen target suite and an incumbent baseline on that suite **before** training; cross-difficulty before/after comparisons are forbidden.

## Replay-safe adaptive compute

Each cycle uses:

- 4,096 fresh focus-domain verifier-oracle examples;
- 512 replay examples from each non-focus domain;
- mandatory first-answer and terminator coverage;
- unique target contexts only;
- continuation capacity weighted 70% to the focus skill and 15% to each replay skill;
- an 80/20 procedural/public-data step mix.

The total training budget is bounded by observed focus capability:

- <50% exact: 3M tokens;
- 50–80% exact: 2.5M;
- >=80% exact: 2M.

The 2M floor is intentional. With 4,096 focus records plus two 512-record replay sets, first-answer and terminator anchors already require 10,240 procedural target updates. Smaller budgets would recreate the anchor-starvation failure measured in #136.

## Curriculum execution boundary

`autonomous_curriculum.py` consumes the hash-bound plan in a separate layer from the controller. It may reconstruct target-suite prompts only to create exclusion hashes; it never sends task bodies or answers back into the planner.

It generates fresh deterministic verifier-oracle records for the focus and replay domains, verifies zero target-holdout prompt overlap, and binds the corpus to:

- plan SHA-256;
- incumbent checkpoint SHA-256;
- target suite hash/difficulty;
- tokenizer hash;
- public-corpus manifest;
- $0 cash compute.

## Candidate training boundary

`autonomous_training.py` continues from the exact incumbent checkpoint. It builds a unique generation-aligned target schedule, covers every first-answer and terminator anchor, then allocates remaining continuation targets 70/15/15 across focus/replay domains. If any domain lacks enough unique continuation contexts, the cycle fails **before** training rather than silently duplicating targets.

The incumbent is never edited in place. Primary and independent replica candidates are separate checkpoints.

## Promotion boundary

`autonomous_gate.py` is external to the planner/trainer. A candidate is promoted only when every hash/lineage check and every plan-declared gate passes:

- same-suite comparison;
- minimum focus-domain gain;
- minimum GCI gain;
- bounded non-focus regressions;
- <=2% M3 validation-loss regression;
- zero M3 exact contamination;
- zero target-holdout overlap;
- independent semantic reproduction;
- exact controller training budget;
- $0 cash compute.

Only this immutable promotion decision may replace the incumbent.

## Remote autonomous execution

`.github/workflows/autonomous-improvement.yml` turns the controller into a persistent remote loop.

- Runs only on GitHub-hosted `ubuntu-latest`; `self-hosted` is forbidden.
- Runs automatically every Tuesday and Friday at 05:17 UTC.
- Also runs once when autonomy/controller/evaluator wiring changes on `main`.
- Uses serialized concurrency and a five-hour hard job timeout.
- Rebuilds the provenance-locked public corpus from source each cycle.
- Evaluates the current incumbent before planning.
- Creates a plan-bound curriculum and trains a primary candidate plus independent deterministic replica.
- Evaluates incumbent and candidate on the same frozen target suite and on the M3 regression boundary.
- Commits immutable cycle measurements on both rejection and promotion.
- Commits checkpoint weights and updates the incumbent pointer only after every external promotion gate passes.
- Uses `research/autonomous/state.json` as the minimal persistent state needed for the next scheduled run.

The routine loop therefore has no dependency on a personal computer or a manual workflow dispatch. A machine owned by the user may remain offline indefinitely without stopping scheduled attempts.

## Difficulty escalation

Difficulty 1 uses `evals/m6-domain-selection-v2.json`. Difficulties 2–5 are resolved to the frozen GCI-Ladder suite files. If the controller requests a difficulty whose frozen suite is not present on `main`, the workflow fails closed before curriculum generation or training. Adding the frozen ladder suite later automatically unlocks the next level without changing controller semantics.

## Persistence rules

Rejected candidate weights remain ephemeral and are never committed. Each completed cycle records plan, curriculum lock, training accounting, reproduction evidence, identical-suite evaluations, M3 evaluations, gate output, and a concise summary under `research/autonomous/cycles/`.

A passing candidate is copied to `checkpoints/genesis-autonomous-incumbent.pt`, documented under `models/genesis-autonomous-incumbent/`, and becomes the parent of the next scheduled cycle. Git history remains the immutable audit trail for promoted incumbent changes.

## Anti-reward-hacking boundary

The planner cannot inspect or modify private holdout answers or graders. Curriculum generation, model training, and promotion evaluation remain separate code/data surfaces. This prevents the self-improvement controller from directly optimizing against private evaluation content or rewriting its own success criterion.
