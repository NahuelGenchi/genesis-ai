# M6 code curriculum

Issue: #96.

## Purpose
Build the first useful-domain training curriculum after #95 selected `code` by frozen evaluation evidence.

## Curriculum v1
- Domain: restricted integer-expression synthesis.
- Difficulty: 1.
- Training seed: `46001`.
- Unique procedural prompts: **4,096**.
- Target training budget for #97: **2,000,000 tokens**.
- Training mix: **80% procedural / 20% approved public-domain text**.
- Model context: 128 tokens.

Every procedural answer is derived by the deterministic project oracle and must pass the existing restricted-expression verifier before entering the curriculum.

No proprietary model, API, distillation output, or AI-generated answer is used.

## 128-token window policy
Some canonical code prompts plus their answer exceed the model's 128-token context because of the fixed instruction prefix.

#97 must therefore construct each procedural training sequence exactly as follows:
- concatenate `prompt + "\nAnswer:" + response`;
- preserve **all response tokens**;
- when the sequence exceeds the training window, remove tokens only from the **left edge of the prompt prefix**;
- never truncate or mask away any response token;
- use response-only loss for procedural examples.

The curriculum lock records how many examples require prefix truncation and the maximum effective training-window size. A response that cannot itself fit the context fails closed.

## Evaluation isolation
The frozen #95 code holdout is reconstructed from `evals/m6-domain-selection-v1.json` before curriculum generation.

Training fails closed unless:
- the frozen evaluation suite version/hash matches;
- #95 still selects `code`;
- the reconstructed holdout task hash matches the committed #95 result;
- training and evaluation seeds differ;
- **exact training/evaluation prompt overlap is zero**.

Evaluation prompts are excluded explicitly by SHA-256, not merely by using another random seed.

## Public-domain mix
Public text is rebuilt from the existing locked Project Gutenberg bootstrap sources. The curriculum records source/catalog/manifest/content hashes and token counts but does not copy the text into Git.

This text mix exists to reduce catastrophic loss of general-language modeling while #97 trains the specialized capability.

## Artifacts
Generated procedural records are ephemeral under `runs/` and contain only:
- prompt;
- deterministic oracle response;
- source task ID;
- minimal procedural provenance.

Hidden verifier tests/expected values are not copied into training records.

Only the compact hash-bound lock `research/m6-code-curriculum-v1.json` is committed after the one-shot build succeeds.
