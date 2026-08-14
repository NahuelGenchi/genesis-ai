# M6 Multi-Domain Full-Capacity v2

## Purpose

Correct the under-supervision failure measured in #136 without changing the incumbent, architecture, tokenizer, evaluator, or promotion threshold.

## Frozen inputs

- Parent: promoted `genesis-micro-2m-v1`.
- Evaluator: unchanged `m6-domain-selection-v2`.
- Difficulty: 1.
- Domains: code, math, structured.
- Examples: 4,096 unique verifier-oracle records per domain; 12,288 total.
- Holdout prompts are reconstructed before curriculum generation and excluded by SHA-256.
- Responses use explicit generated newline termination.
- Public text remains 20% of steps.
- Cash compute cost: $0.

## Compute

Target continuation budget: 6,000,000 tokens.

At 1,024 processed tokens/step:

- 5,860 total steps;
- 4,688 procedural steps;
- 1,172 public-text steps;
- 37,504 unique procedural target updates.

Mandatory first-answer + terminator anchors consume exactly 24,576 target updates. The remaining 12,928 continuation targets are frozen as:

- code: 4,310;
- math: 4,309;
- structured: 4,309.

Total supervised target contexts therefore become:

- code: 12,502;
- math: 12,501;
- structured: 12,501.

This intentionally restores approximately the 12,512 target-context capacity that produced the first 95%-exact code checkpoint, but now for each skill independently inside one continuation run.

## Scientific isolation

v1 remains immutable. v2 changes only curriculum capacity and continuation replay balance. The optimizer, architecture, parent, evaluator, termination protocol, public-data fraction, and primary promotion target remain unchanged so the effect is attributable.

## Promotion

Every gate is blocking:

- GCI-v1 relative improvement >= 100%;
- code exact >= 90%;
- math exact >= 50%;
- structured exact >= 50%;
- M3 validation-loss regression <= 2%;
- zero M3 exact contamination;
- zero frozen-holdout prompt overlap;
- exact semantic reproduction from an independent run;
- exact 37,504 unique procedural updates and per-domain allocation;
- target training budget >= 6M tokens;
- cash compute = $0.

A rejected candidate publishes no checkpoint. A passing candidate publishes `genesis-micro-2m-v2.pt` with a model card and immutable result bundle.
