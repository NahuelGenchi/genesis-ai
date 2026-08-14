# genesis-tiny-v0

## Purpose
First end-to-end model trained from random weights. **Pipeline baseline only.**

## Architecture
- Parameters: 394,560
- Vocabulary: 512
- Context: 128
- Width: 96
- Heads: 4
- Layers: 3
- FFN: 384

## Training
- Steps: 160
- Intentional resume point: 80
- Train documents: 975
- Train tokens: 247453
- Probe loss: 6.278069019317627 → 3.982328414916992
- Last training loss: 3.9865145683288574
- CPU time: 1.386398927999835 s

## Evaluation
- Split: validation
- Documents: 107
- Tokens: 29360
- Loss: 3.686832940578461
- Perplexity: 39.918223165821594

## Tokenizer
- `genesis-v0`
- Vocabulary: 512
- Bytes/token: 1.621624

## Training sources
- `gutenberg-don-quijote-2000` — language `es` — sample SHA `2cbc9e527f6f3e975f016cd4bab0cf8524b5406ea3327b5e0d79579df8477062`
- `gutenberg-les-miserables-17489` — language `fr` — sample SHA `b6b64be2f61cbd2f380c937ac2244c2665e0060e5bc5c67f8d31f6cda4be6f8a`
- `gutenberg-pride-prejudice-1342` — language `en` — sample SHA `c7246fa08df958e075cf6343b68103cea963fd99deecf6977e016ae0824ecf83`

## Checkpoint
- SHA-256: `4db01f8239cfc28933bc3152b7b698ffe089491ce1acf5b66de8adc53b9f8ed9`
- Bytes: 1594143

## Seeded sample
```text
The hs o cons vas

cert de h que to de sey bo to saial od pral vindécfo,t, de que f, en enma elcpla cauror
que sa
```

## Limitations
- Pipeline-validation model; not a useful general assistant.
- Training corpus is tiny and literature-only.
- Training languages are limited to English, Spanish, and French source samples.
- No instruction tuning, preference tuning, tool training, safety tuning, or factuality training.
- Validation is document-disjoint but comes from the same small source pool.
- Generated text may be incoherent or malformed.
