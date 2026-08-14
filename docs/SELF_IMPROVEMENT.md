# Self-Improvement

The deployed model never edits its own live weights.

## Loop

1. Current model solves or proposes tasks.
2. Deterministic or independent verifiers score results.
3. High-value experience enters a candidate dataset.
4. A new checkpoint is trained offline.
5. Candidate and incumbent run the same eval suite.
6. Candidate is promoted only if gates pass.

## Guardrail

No automatic deployment based only on the model's own judgment.
