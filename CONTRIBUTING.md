# Contributing

Genesis AI treats reproducibility and concise tracking as part of the research result.

## Before changing anything

1. Open a GitHub Issue.
2. Assign exactly one Milestone.
3. Keep the Issue readable in under one minute: **Goal / Why / Done when**.
4. Make the smallest coherent change.
5. Add or update tests when behavior changes.
6. Run `./scripts/test_local.sh`.
7. Open a PR with `Closes #N`.

Research experiments follow the same rule. Record the hypothesis, fixed setup, result, and decision in the Issue.

## Reproducible experiment flow

```text
Issue
  ↓
local/unit validation
  ↓
GitHub CPU screen when applicable
  ↓ only measured model winners
Kaggle/Colab accelerator job when justified
  ↓
frozen evaluation + regression/contamination gates
  ↓
promotion decision
```

Do not jump directly to GPU training because an idea sounds promising. Cheap evidence comes first.

## Good first contributions

You do not need a GPU. Good starter changes include:

- deterministic benchmark cases;
- verifier edge cases;
- data-quality/filtering tests using redistributable fixtures;
- CPU-farm reporting and visualization;
- reproducibility/documentation fixes;
- small efficiency experiments with an explicit baseline.

Browse [good first issues](https://github.com/NahuelGenchi/genesis-ai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) or [help wanted](https://github.com/NahuelGenchi/genesis-ai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22).

## Data and model lineage

- Use only data with documented provenance and reviewed training rights.
- Never commit secrets, credentials, private data, or proprietary-model outputs as training data.
- Never use proprietary model weights, distillation, or runtime inference dependencies.
- Raw corpora stay outside Git unless redistribution is explicitly allowed and tracked.

## Accelerator experiments

Accelerator jobs must follow [`docs/FREE_ACCELERATORS.md`](docs/FREE_ACCELERATORS.md). A GPU job must be backed by a CPU-farm summary that lists the exact model-side variant in `expensive_stage_eligible`.

Kaggle/Colab output is still a candidate. It cannot bypass frozen evaluation or promote a checkpoint directly.

## Capability claims

Use committed benchmark evidence. State narrow results narrowly. A synthetic/domain benchmark win is not evidence of broad assistant or frontier capability.
