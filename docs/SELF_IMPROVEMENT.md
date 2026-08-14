# Self-improvement

## Safety rule
A deployed checkpoint never rewrites itself.

The loop is offline:

`challenge → solve → verify → experience → train candidate → fixed evals → promote/reject`

Only a new checkpoint that passes explicit gates may replace an incumbent.

## Challenger v1
`procedural-v1` generates deterministic tasks without proprietary APIs.

Domains:
- `math` — exact integer arithmetic;
- `structured` — exact JSON transforms;
- `code` — restricted integer expressions over `x`/`y`.

Every task records ID, generator version, seed/ordinal, domain, difficulty, prompt, verifier spec, and procedural provenance. Task IDs are SHA-256-derived; duplicates are rejected.

## Deterministic verifier v1
The verifier does **not** ask the model whether it was correct.

- integers: strict parse + exact comparison;
- JSON: parse + exact structural comparison;
- code: custom interpreter over a restricted Python-expression AST.

### Code safety
Model-written code is never passed to `eval`, `exec`, a shell, subprocess, import system, or Python runtime execution API.

Allowed expression pieces:
- integer constants;
- approved variable names from the test case;
- `+ - * // % **`;
- unary `+ -`;
- parentheses.

Calls, attributes, indexing, comprehensions, lambdas, containers, imports, comparisons, booleans, and other syntax are rejected. Expression length, AST size, exponent, and result magnitude are bounded.

## Current limit
This is curriculum/verifier infrastructure, not evidence that `genesis-tiny-v0` can solve these tasks. Verified model experience is not created until the model actually produces passing answers.
