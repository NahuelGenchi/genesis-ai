# M6 code answer termination investigation

Issue: #111.

## Problem
The rejected #97 micro-2m candidate produced:
- frozen code exact accuracy: `0/60`;
- code oracle-target loss: `0.0006654457` from baseline `6.6200023013`;
- reproducible model tensors;
- improved M3 validation loss.

Because the v1 evaluator always generated 32 new tokens and the curriculum taught no EOS/answer-stop marker, answer termination was the leading hypothesis.

The benchmark was deliberately left unchanged until generated-response evidence could test that hypothesis.

## Diagnostic v1
Workflow run `31837683797`:
1. rebuilt the exact frozen #96 curriculum/public corpus;
2. retrained one ephemeral micro-2m model under the reproducible #97 v2 policy;
3. verified the training trajectory matched the frozen #97 measurements;
4. ran the unchanged `m6-domain-selection-v1` code holdout;
5. compared every free-running generation against the exact oracle token sequence and decoded oracle text;
6. committed aggregate evidence plus eight audit samples;
7. discarded the model weights.

## Measured result
Across all 60 frozen code tasks:
- strict exact answers: **0/60**;
- generations beginning with the exact oracle **token** sequence: **0/60**;
- generations beginning with the decoded oracle **text**: **0/60**;
- correct oracle prefix followed by extra tokens: **0/60**;
- newline immediately after a correct oracle prefix: **0/60**.

Example:
- oracle: `5*x + 1*y + -8`
- generated: `5555*x1*1*y 8*x8*x 8*x55555555y1`
- verifier result: `invalid_syntax`

The evidence therefore **rejects the answer-termination hypothesis**. The model is already wrong before an answer terminator could matter.

## Decision
No termination-aware replacement benchmark will be introduced for this failure.

`m6-domain-selection-v1` and its frozen baseline remain valid and immutable. Changing the evaluator after this result would incorrectly make the benchmark easier for a candidate that did not generate the target answer.

The actual defect is the gap between near-zero teacher-forced target loss and failed free-running generation. That work continues under #115, which will measure first-response-token accuracy and longest-correct-prefix behavior before changing training.

## Frozen record
`research/m6-termination-diagnostic-v1.json` is the accepted negative-result record. The automatic diagnostic has been replaced by a manual-only verifier so the evidence cannot silently rerun or change.
