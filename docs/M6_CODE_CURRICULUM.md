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

## Frozen measured result
The one-shot runner completed successfully in workflow run `31834675482`.

Measured lock:
- 4,096 unique procedural examples from 5,195 generation attempts;
- 0 exact prompt overlaps with the frozen 60-task code holdout;
- 610,542 raw prompt+response tokens;
- 61,559 response tokens;
- maximum raw prompt+response length: 153 tokens;
- **4,096 / 4,096** examples require left-prefix truncation;
- maximum effective training window: 129 tokens (`x` + next-token targets);
- public-language corpus: 1,082 documents / 274,653 tokens;
- required cash compute: $0.

The automatic builder is frozen after this accepted result. The remaining workflow is manual-only and verifies the committed v1 identities; it does not regenerate the curriculum.

## 128-token window policy
The canonical code prompts plus their answer exceed the model's 128-token context because of the fixed instruction prefix.

#97 must therefore construct each procedural training sequence exactly as follows:
- concatenate `prompt + "\nAnswer:" + response`;
- preserve **all response tokens**;
- when the sequence exceeds the training window, remove tokens only from the **left edge of the prompt prefix**;
- never truncate or mask away any response token;
- use response-only loss for procedural examples.

A response that cannot itself fit the context fails closed.

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

Only the compact hash-bound lock `research/m6-code-curriculum-v1.json` is committed.
