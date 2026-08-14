# M6 answer alignment diagnostic

Issue: #118.

## Purpose
Measure why #97's micro-2m model has near-zero teacher-forced code target loss while free-running code generation remains 0/60 exact.

## Structural hypothesis
The current response-only dataset and oracle-target-loss scorer anchor a fixed 128-token window to the **end of the full oracle response**.

For an oracle response of 14 tokens, this places the first response target at absolute model position 114 and the final response target at position 127.

Free generation is different: `model.generate` takes the last 128 tokens of the current history, so every newly generated next token is predicted at absolute model position 127.

Because micro-2m uses learned absolute position embeddings, those are materially different conditioning layouts.

## Frozen measurement
The one-shot diagnostic reproduces the exact rejected #97 model, then scores the unchanged 60-task code holdout in three ways:

1. **End-anchored teacher forcing** — current training/oracle-loss layout.
2. **Generation-aligned teacher forcing** — for each oracle token, score the next token from the exact last-128-token history layout used by free generation.
3. **Free running** — unchanged greedy v1 generation; record longest exact oracle-token prefix.

## Metrics
The committed result records:
- first-response-token top-1 accuracy, mean probability, and mean NLL under both teacher-forcing layouts;
- all-response-token top-1 accuracy/NLL under both layouts;
- per-response-position metrics;
- end-anchored absolute model-position ranges;
- free-running first-token accuracy;
- longest-correct-prefix mean/max/histogram;
- task/suite/checkpoint hashes.

No training data, model weights, architecture, or benchmark semantics are changed by this Issue.
