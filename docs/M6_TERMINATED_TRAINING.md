# M6 explicit answer termination

Issue: #128.

## Evidence

#125 reconstructed the generation-aligned #115 candidate and measured free generation on the unchanged 60-task code holdout:

- exact oracle token prefix: `60/60`;
- exact oracle text prefix: `60/60`;
- oracle prefix followed by extra tokens: `60/60`;
- newline immediately after oracle: `0/60`;
- strict v1 exact: `0/60`.

Therefore the aligned model knows the complete oracle answer but has no learned answer-stop protocol.

## Versioned stop protocol

Historical `m6-domain-selection-v1` remains immutable.

`m6-domain-selection-v2` preserves the same:
- base seed;
- task count;
- difficulty;
- domain order;
- task generator;
- generation seed;
- `max_new_tokens=32`;
- greedy `top_k=1` policy.

The only semantic addition is a **required generated newline** (`\n`) answer terminator.

Generation stops only when the model itself emits the delimiter. The verifier receives the decoded text before that delimiter. If no delimiter is generated within 32 tokens, the task cannot pass strict exact.

The currently promoted baseline checkpoint is evaluated on v2 **before candidate training**. Candidate improvement is therefore same-suite baseline → candidate, never a comparison of incompatible v1/v2 scores.

## Training policy

`m6-micro-2m-terminated-training-v1` keeps the accepted M6 scale contract:
- random initialization seed `97001`;
- 1,895,808 parameters;
- context 128;
- learned absolute positions;
- deterministic one-thread CPU execution;
- AdamW with fixed deterministic backend;
- 1,955 steps / 2,001,920 processed tokens;
- exact 80% procedural / 20% public schedule;
- #96 tokenizer, corpus, curriculum, provenance and holdout separation;
- required cash compute `$0`.

Each procedural target sequence is the deterministic oracle response plus a newline. Every token is trained from the exact sliding context used by generation.

## Frozen procedural schedule

The 2M compute budget permits 12,512 procedural target updates.

The deterministic schedule must include, exactly once each:
- all 4,096 first-response targets;
- all 4,096 newline-terminator targets.

The remaining 4,320 updates are unique continuation target contexts. No target context is duplicated.

This changes target allocation, not total training compute.

## Promotion

Promotion remains blocked unless all gates pass:
- stop-aware code exact accuracy gains at least +5 percentage points over the promoted baseline evaluated on the same v2 suite;
- M3 validation-loss regression <=2%;
- zero exact contamination;
- zero frozen-holdout prompt overlap;
- exact parameter count;
- frozen token budget and 80/20 mix;
- 100% first-response and terminator target coverage;
- unique generation-aligned schedule;
- independent semantic reproducibility;
- `$0` required cash compute.

Only a fully promoted candidate may publish `checkpoints/genesis-micro-2m-v1.pt` and its model card.
