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

## Issue-close capability reporting

Every closed Issue must end with an auditable capability footer. Publication state and model quality are separate facts.

Required fields:

- `Model capability change in this Issue:` primary benchmark before -> after, absolute percentage-point delta, and relative percent only when the baseline is greater than zero.
- `Model capability change since Issue #1:` same benchmark anchor. If Issue #1 had no model or the baseline score is zero, report the absolute gain and `relative %: N/A (zero baseline)`; never report `+0.00%` for a real zero-to-positive capability gain.
- `Model weights changed:` yes/no.
- `Checkpoint promoted:` yes/no.

Diagnostics, documentation, and infrastructure Issues may have no weight change. They must preserve the latest validated capability evidence rather than resetting project progress to zero.

When evaluator semantics change, both incumbent and candidate must be evaluated on the same new suite before any improvement claim is made. Cross-suite percentages are forbidden.
