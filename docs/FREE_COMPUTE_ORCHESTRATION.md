# Continuous free-compute orchestration

Issues: #194, #142, #203.

## Canonical loop

`Autonomous verified improvement` remains the only workflow with checkpoint-promotion authority. It runs on GitHub-hosted `ubuntu-latest`, independently reproduces candidates, evaluates frozen suites, applies the immutable gate, persists the decision, and continues remotely at `$0` cash compute.

The controller now consumes committed cycle metadata as well as aggregate incumbent metrics. Repeated zero-focus-gain rejections are treated as **research stagnation**, not as permission to keep running equivalent seed-only retries. After three zero-gain failures under one high-level strategy, the next canonical plan rotates to a materially different bounded training intervention.

## CPU research farm

The public GitHub-hosted CPU farm runs every three hours and screens bounded architecture, optimizer, tiny-model, tokenizer, filtering, evaluation, and verifier variants. The latest aggregate is persisted at `research/accelerators/cpu-farm-latest.json`. Screening still has no promotion authority.

A farm winner is no longer permanent. Canonical history counts reject+zero-focus-gain outcomes while a hint is applied. After five such end-to-end failures for the current incumbent, that hint is automatically retired from canonical use even if it remains a screening winner. This prevents fixture improvements from consuming unlimited downstream cycles without capability evidence.

## Stagnation-triggered architecture research

When the canonical controller enters a new high-level research strategy, it emits a tracked research escalation. The workflow updates Issue #203 (or creates another M6 research Issue after #203 closes) and can dispatch `M6 architecture tournament v1` when an immutable tournament result is absent.

The architecture tournament runs on public `ubuntu-latest` under an equal 2e12-FLOP budget per candidate and compares the frozen baseline with RoPE-only, RMSNorm-only, and parameter-matched SwiGLU. The result is committed as research evidence only.

A successful tournament triggers `M6 architecture finalist v1`, also on GitHub-hosted CPU. The initial winner is reproduced against the baseline on two fresh deterministic seeds. A challenger is accepted only when mean validation loss improves by at least 0.5% and no seed is more than 1% worse. Neither workflow can write or promote the canonical incumbent checkpoint.

This makes the free-compute ladder:

`canonical failure history -> new research strategy -> CPU architecture tournament -> fresh-seed reproduction -> later scale/canonical experiment`

rather than `reject -> new seed -> reject forever`.

## Kaggle GPU

Every successful CPU-farm completion still triggers the Kaggle dispatcher. Only model-side candidates that survived CPU screening and can be faithfully expressed by the accelerator runner are eligible. Missing credentials/quota or no model-side winner causes a clean skip. Kaggle output has no direct promotion authority.

`Kaggle GPU result collector` retrieves and validates non-promoting accelerator records. A later frozen/reproduced experiment must explicitly consume any useful accelerator evidence before it can affect the incumbent.

## Free Google Colab

Free Colab remains an interactive opportunistic sidecar, never an unattended worker. `accelerators/colab/genesis_colab.ipynb` may consume eligible committed research manifests when an active notebook user explicitly enables execution. Colab availability is not required for autonomy.

## Failure behavior

- Canonical candidate fails a gate: incumbent unchanged; outcome becomes negative research evidence.
- Same strategy reaches three zero-focus-gain rejects: strategy rotates automatically.
- Applied CPU hint reaches five end-to-end zero-focus-gain rejects: hint is retired automatically.
- Research escalation: tracked in M6 and architecture research is dispatched when appropriate.
- Architecture tournament/finalist fail: no model promotion; canonical loop remains safe.
- CPU farm fails: canonical improvement continues without new farm evidence.
- No Kaggle winner/credentials/quota: Kaggle skips.
- Colab unavailable: no effect on canonical autonomy.
- No research lane may weaken frozen evaluation or promote weights directly.
