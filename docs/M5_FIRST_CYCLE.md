# M5 first self-improvement cycle

Issue: #87.

## Fixed cycle v1
- Parent: `checkpoints/genesis-tiny-v0.pt`.
- 60 deterministic procedural tasks.
- Difficulty: 1 only.
- Domains: math, structured JSON, restricted expression.
- Challenge seed: `20260814`.
- Genesis generation: greedy (`top_k=1`), 16 new tokens, seed `9107`.
- Acceptance: deterministic verifier score exactly `1.0`.
- Candidate threshold: **8 accepted answers**.

## Measured result
Workflow run `31808968420` at source commit `413792faf6c82fe75b3dd0faa8c5a216d9e510fa` completed successfully.

- Attempted: **60**.
- Accepted: **0**.
- Rejected: **60**.
- Acceptance rate: **0%**.
- Status: **`no_candidate`**.
- Reason: `insufficient_verified_experience`.
- Candidate training: **skipped**.
- Evaluation/benchmark/promotion: **skipped**.
- Parent checkpoint remained unchanged: SHA-256 `4db01f8239cfc28933bc3152b7b698ffe089491ce1acf5b66de8adc53b9f8ed9`.

This is a valid safety result: the baseline is not yet capable enough to provide verified self-training signal under v1, so the system refused to manufacture one.

## Fail closed
If fewer than 8 answers pass, the cycle records:
- `status: no_candidate`;
- accepted/rejected counts;
- parent/task/experience hashes;
- `reason: insufficient_verified_experience`.

No answer is corrected or replaced. No candidate is trained.

## Candidate path
Only when the threshold is met:
1. train N+1 for 50 deterministic candidate steps;
2. rebuild the approved corpus and verify the tokenizer;
3. run M3-v1 on parent and candidate;
4. benchmark both on the same runner;
5. apply `promotion-v1`;
6. record `candidate_promoted` or `candidate_rejected`.

Candidate files and raw experience remain ephemeral. Only the compact cycle JSON is committed.

## Immutability
The executed workflow hashed the parent before and after the cycle and failed if it changed.

## Record
`research/m5-cycle-v1.json` binds:
- workflow source commit/run;
- parent checkpoint;
- task file;
- experience manifest/accepted/audit files;
- candidate/evaluation/benchmark/promotion files when present;
- final cycle status;
- deterministic record SHA-256.

## Frozen
The v1 workflow is now **manual-only** and verifies the committed record rather than recomputing the cycle. The historical result cannot silently change.
