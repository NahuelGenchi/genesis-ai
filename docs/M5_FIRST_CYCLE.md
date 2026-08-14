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
The workflow hashes the parent before and after the cycle and fails if it changes.

## Record
`research/m5-cycle-v1.json` binds:
- workflow source commit/run;
- parent checkpoint;
- task file;
- experience manifest/accepted/audit files;
- candidate/evaluation/benchmark/promotion files when present;
- final cycle status;
- deterministic record SHA-256.

After the first accepted run, the automatic workflow is frozen to manual-only so the historical v1 result cannot silently rerun.
