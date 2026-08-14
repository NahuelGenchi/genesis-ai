# Evaluation Policy

Every promoted checkpoint must be compared against the incumbent on frozen, versioned suites.

Track at minimum:

- validation loss/perplexity;
- reasoning accuracy on verifiable tasks;
- code-test pass rate;
- regression count;
- training compute/time;
- inference tokens/second;
- peak memory;
- estimated cost per million tokens when applicable.

Private holdouts must not enter training data.

## Genesis Capability Index (GCI-v1)

`GCI-v1` is the project-wide breadth metric for the frozen stop-aware domain suite. It is the arithmetic mean of strict exact accuracy in `code`, `math`, and `structured`, expressed on a 0-100 scale.

- Current validated code-specialist evidence (`95%`, `0%`, `0%`) = `31.6667` GCI.
- A candidate at (`90%`, `50%`, `50%`) = `63.3333` GCI, exactly a `+100%` relative GCI improvement while preserving at least 90% code exact accuracy.
- Relative GCI improvement is undefined at a zero baseline; report the absolute point gain plus `N/A (zero baseline)`.
- GCI comparisons are valid only when suite version and suite hash are identical.

GCI is a breadth summary, not a replacement for blocking per-domain, regression, contamination, reproducibility, or cost gates.

## Issue-close capability reporting

Every closed Issue must end with an auditable capability footer. Publication state and model quality are separate facts.

Required fields:

- `Model capability change in this Issue:` primary benchmark before -> after, absolute percentage-point delta, and relative percent only when the baseline is greater than zero.
- `Model capability change since Issue #1:` same benchmark anchor. If Issue #1 had no model or the baseline score is zero, report the absolute gain and `relative %: N/A (zero baseline)`; never report `+0.00%` for a real zero-to-positive capability gain.
- `GCI-v1:` before -> after, absolute point change, and relative percent when defined.
- `Model weights changed:` yes/no.
- `Checkpoint promoted:` yes/no.

Diagnostics, documentation, and infrastructure Issues may have no weight change. They must preserve the latest validated capability evidence rather than resetting project progress to zero.

When evaluator semantics change, both incumbent and candidate must be evaluated on the same new suite before any improvement claim is made. Cross-suite percentages are forbidden.
