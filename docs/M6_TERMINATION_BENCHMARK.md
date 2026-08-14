# M6 code answer termination benchmark

Issue: #111.

## Problem
The rejected #97 micro-2m candidate produced:
- frozen code exact accuracy: `0/60`;
- code oracle-target loss: `0.0006654457` from baseline `6.6200023013`;
- reproducible model tensors;
- improved M3 validation loss.

The v1 domain evaluator always generates 32 new tokens and has no EOS/answer-stop contract. The procedural curriculum likewise teaches the answer expression but no terminator.

This makes answer termination the leading hypothesis, but the benchmark will not change until generated-response evidence proves it.

## Diagnostic v1
A one-shot self-hosted workflow:
1. rebuilds the exact frozen #96 curriculum/public corpus;
2. retrains one ephemeral micro-2m model under the reproducible #97 v2 policy;
3. verifies the training trajectory matches the frozen #97 measurements;
4. runs the unchanged `m6-domain-selection-v1` code holdout;
5. records whether each generated token sequence begins with the exact oracle-answer token sequence;
6. separately records strict verifier success and whether correct prefixes are followed by extra generated tokens;
7. commits only compact diagnostic evidence; model weights remain ephemeral.

## Evidence required before changing evaluation
Termination is considered proven only if the diagnostic shows that a substantial portion of strict failures:
- begin with the exact oracle token sequence; and
- contain additional generated tokens after that correct sequence.

Decoded-string prefix evidence is recorded in addition to token-prefix evidence. Sample outputs are project-owned Genesis generations used only for audit, never as training targets.

## Benchmark-change rule
`m6-domain-selection-v1` and its frozen result remain immutable.

If termination is proven, a new versioned suite will be introduced. It must:
- preserve the exact holdout task content/seeds;
- define a deterministic answer terminator independent of verifier feedback;
- terminate generation only on that explicit representation;
- strip only the defined terminator before exact verification;
- be baseline-evaluated and frozen **before** any retraining result can be accepted;
- introduce no proprietary or model-generated training targets.
