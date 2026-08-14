# Autonomous Improvement Controller v1

## Purpose

Choose the next Genesis training target automatically from aggregate capability measurements while keeping private holdout content, incumbent weights, and promotion authority outside the controller.

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

When all domains reach at least 80% strict exact accuracy, the controller raises difficulty instead of repeatedly polishing the same easy suite.

## Adaptive compute

Focus budget is bounded by observed aggregate capability:

- <50% exact: 2M focus-training tokens;
- 50–80% exact: 1M;
- >=80% exact: 0.5M.

Every cycle also specifies incumbent-domain replay and an 80/20 procedural/public-data mix. This is a planning budget, not permission to bypass the separate training and promotion gates.

## Promotion boundary

The controller never edits the incumbent checkpoint. It emits a hash-bound candidate plan only.

A downstream candidate still requires:

- same-suite capability improvement;
- positive focus-domain gain;
- bounded non-focus regressions;
- <=2% M3 validation-loss regression;
- zero holdout overlap;
- independent semantic reproduction;
- $0 cash compute.

Only an external immutable promotion gate may replace the incumbent.

## Anti-reward-hacking boundary

The planner cannot inspect or modify private holdout answers or graders. Curriculum generators and promotion evaluators remain separate code/data surfaces. This prevents the self-improvement controller from directly optimizing against private evaluation content.
