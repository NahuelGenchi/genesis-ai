# Project Rules

## Work tracking

1. **Issue first.** No work starts without an Issue.
2. **Milestone required.** Every Issue belongs to exactly one Milestone.
3. **One clear outcome per Issue.** Split large work.
4. **Everything counts.** Code, docs, research, experiments, bugs, refactors, tooling, CI, and process changes all need Issues.
5. **Errors are tracked.** Any discovered defect gets a Bug Issue before or alongside the fix.
6. **Experiments are tracked.** Hypothesis, setup, result, decision.
7. **PRs reference Issues.** Do not auto-close an Issue before its required model-improvement report is shown.
8. **No hidden roadmap.** Status belongs in GitHub Milestones/Issues.

## Issue writing style

Keep every Issue readable in under one minute.

Required sections:

- **Goal** — one sentence.
- **Why** — one sentence.
- **Done when** — short checklist.
- **Notes** — only if needed.

## Issue closure reporting

Immediately before every Issue is closed, show exactly these two model-progress percentages:

- **Model improvement since previous closed Issue:** `±X.XX%`
- **Model improvement since Issue #1:** `±Y.YY%`

Rules:

- Use the active milestone's frozen primary capability metric; do not change the metric after seeing candidate results.
- For M6, the primary capability metric is the frozen selected-domain **exact accuracy**, and reported improvement is its absolute percentage-point change.
- Issue #1 is treated as **0% model capability** because no model existed at project start.
- If an Issue does not change model weights or the frozen capability result, report `+0.00%`.
- A capability gain never overrides regression gates; M3 general-quality, contamination, reproducibility, compute, and cost requirements remain blocking.

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
