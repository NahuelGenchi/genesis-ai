# Evaluation Policy

Every promoted checkpoint must be compared against the incumbent.

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
