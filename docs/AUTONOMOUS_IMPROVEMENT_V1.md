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

This is a planning budget, not permission to bypass separate training and promotion gates.

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
