# Genesis AI

Genesis AI is a **zero-budget, from-scratch AI research project**.

The long-term mission is to build an independent model family that improves capability per unit of compute over time without relying on proprietary model weights or inference APIs.

## Hard constraints

- **$0 required spend.** Paid compute, paid APIs, and paid services are not project requirements.
- **Independent weights.** Models start from random initialization.
- **No proprietary-model dependency.** Production, training, and evaluation must not require ChatGPT, Claude, or another proprietary model.
- **Evidence before scale.** Larger experiments happen only after smaller experiments justify them.
- **Everything is tracked.** Every task, bug, decision, experiment, documentation change, and enhancement belongs to a GitHub Issue and Milestone.
- **Human-readable tracking.** Issues stay short, explicit, and easy to scan.

## Current stage

**M0 — Bootstrap & Rules**

The initial repository contains:

- a tiny causal language-model baseline;
- local training, generation, evaluation, and checkpoint code;
- lightweight tests and CI;
- provenance and evaluation policies;
- a controlled self-improvement architecture;
- versioned manifests for all project Milestones, Issues, and labels.

## Tracking rule

GitHub **Milestones + Issues are the source of truth**.

No meaningful project work should exist only in chat, commits, pull requests, or documentation. If work matters, it gets an Issue.

See [`PROJECT_RULES.md`](PROJECT_RULES.md).

## Quick local check

```bash
./scripts/test_local.sh
```

## Model principle

Start tiny. Measure. Improve. Scale only what wins.
