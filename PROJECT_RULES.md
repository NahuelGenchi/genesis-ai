# Project Rules

## Work tracking

1. **Issue first.** No work starts without an Issue.
2. **Milestone required.** Every Issue belongs to exactly one Milestone.
3. **One clear outcome per Issue.** Split large work.
4. **Everything counts.** Code, docs, research, experiments, bugs, refactors, tooling, CI, and process changes all need Issues.
5. **Errors are tracked.** Any discovered defect gets a Bug Issue before or alongside the fix.
6. **Experiments are tracked.** Hypothesis, setup, result, decision.
7. **PRs close Issues.** Prefer `Closes #N`.
8. **No hidden roadmap.** Status belongs in GitHub Milestones/Issues.

## Issue writing style

Keep every Issue readable in under one minute.

Required sections:

- **Goal** — one sentence.
- **Why** — one sentence.
- **Done when** — short checklist.
- **Notes** — only if needed.

## Cost

- Required project spend: **$0**.
- Paid services must never become required dependencies.
- Free quotas may be used only when they cannot create automatic charges.
- Prefer local CPU/GPU, free research compute, free CI quotas, and reproducible offline workflows.

## Model independence

- No proprietary model weights.
- No proprietary inference API in the model runtime.
- No distillation from proprietary models.
- No proprietary-model-generated training corpus as a project dependency.
- Public research ideas may be implemented independently.

## Model promotion

A candidate checkpoint replaces the current checkpoint only when it:

- improves required evaluation metrics;
- has no blocking regression;
- remains within compute/cost constraints;
- passes reproducibility checks.
