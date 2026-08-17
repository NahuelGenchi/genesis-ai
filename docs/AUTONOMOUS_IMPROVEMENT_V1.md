# Autonomous Improvement Controller

Issues: #142 and #203.

## Purpose

Genesis must do more than keep a training queue busy. The autonomous loop has two separate responsibilities:

1. **execution autonomy** — run, reproduce, evaluate, reject/promote, persist, and continue without a personal computer;
2. **scientific autonomy** — notice when a high-level intervention is exhausted, stop retrying it with new seeds, choose a materially different bounded intervention, retire downstream-failing research hints, and route independent research work.

The immutable promotion gate remains outside the planner and is unchanged by scientific autonomy.

## Planner input boundary

The controller may consume only frozen suite identity/version, current difficulty, aggregate per-domain exact accuracy/loss/termination, incumbent checkpoint SHA-256, the monotonic cycle index, committed aggregate CPU-farm evidence, and committed autonomous cycle **metadata**. It never consumes private holdout prompt/task/answer/oracle/response/text content; those field names are rejected recursively.

Committed history is read from `research/autonomous/cycles/`. Only cycles whose gate baseline SHA-256 matches the current incumbent contribute to stagnation accounting, so a promotion resets the relevant research history naturally.

## Weakness selection

The weakest domain is still selected deterministically by lowest exact accuracy, then highest aggregate terminated-oracle loss, then domain name. When every domain reaches at least 80% exact accuracy, the controller moves to the next frozen GCI-Ladder difficulty and requires an incumbent baseline on that suite before training.

## Stagnation is a first-class failure mode

Changing only `cycle_index`, generated examples, or training seed is not considered a new high-level hypothesis.

Historical cycles without an explicit research strategy are classified as `legacy-focus-heavy-v1`. After **3 reject decisions with zero focus-domain gain** under the same strategy and incumbent, that strategy is exhausted. The controller then selects a materially different predeclared intervention. Each later intervention is also capped at 3 zero-focus-gain rejections before rotation.

Current bounded intervention catalog:

| Strategy | Focus records | Continuation allocation |
|---|---:|---|
| `sequence-depth-v1` | 1,024 | 65% focus / 25% strongest replay / 10% other replay |
| `anti-forgetting-v1` | 1,536 | 50% focus / 40% strongest replay / 10% other replay |
| `balanced-transfer-v1` | 2,048 | 60% focus / 30% strongest replay / 10% other replay |
| `broad-conservative-v1` | 4,096 | 55% focus / 35% strongest replay / 10% other replay |

The strongest replay domain is chosen from aggregate incumbent metrics. This makes interventions change sequence coverage and anti-forgetting pressure instead of merely changing randomness.

`history_summary`, `research_strategy`, and any `research_escalation` are included in `plan_sha256`, so every scientific decision remains exactly replayable from committed evidence.

## Research-hint retirement

CPU-farm screening is evidence, not authority. A screening winner cannot stay enabled forever merely because it keeps winning its fixture.

The controller tracks end-to-end canonical outcomes for each applied hint. After **5 reject decisions with zero focus gain** under the current incumbent, the hint is retired from canonical application. The retirement and count are written into `research_evidence.retired_eligible_hints` and hash-bound into the plan.

For example, a retired `data-filtering/min-chars-80` hint no longer changes public sampling; `public_min_chars` returns to the unfiltered canonical value while the negative evidence remains visible.

## Curriculum and candidate training

The curriculum remains verifier-backed, deterministic, hash-bound, and holdout-separated. Replay records continue to scale with the 3M/2.5M/2M token budget. Focus record count and continuation allocation now come from the research strategy instead of being frozen forever at 4,096 and 70/15/15.

`autonomous_training.py` accepts either the legacy focus/replay allocation or exact per-domain weights. It still requires:

- mandatory first-answer and terminator anchors;
- unique target contexts only;
- sufficient continuation capacity before training starts;
- the exact incumbent checkpoint and tokenizer;
- deterministic CPU training for the canonical micro-model path;
- the existing 80/20 procedural/public step mix;
- `$0` cash compute.

Primary and independent replica use the identical plan-bound policy.

## Independent architecture research escalation

Entering a new strategy creates a research escalation event. The canonical workflow records it in tracked Issue #203 (or creates a new M6 research Issue if that issue is closed) and can dispatch the fixed-FLOP architecture tournament when no immutable tournament result exists.

`M6 architecture tournament v1` runs on public GitHub-hosted `ubuntu-latest` at `$0`, comparing the frozen baseline against RoPE-only, RMSNorm-only, and parameter-matched SwiGLU under identical training FLOPs. Its result cannot promote weights.

A successful tournament automatically triggers `M6 architecture finalist v1`, which reruns the baseline and initial winner on two fresh deterministic seeds. A challenger is accepted as research evidence only if mean validation loss improves by at least 0.5% and no fresh seed regresses by more than 1%. This result also has no checkpoint-promotion authority.

Architecture evidence is intended to feed later scale work such as #160; it does not bypass canonical frozen evaluation.

## Promotion boundary

`autonomous_gate.py` remains external and unchanged. Promotion still requires all declared gates, including same-suite comparison, positive focus gain, minimum GCI gain, bounded non-focus regression, M3 boundary, zero contamination/holdout overlap, independent semantic reproduction, and `$0` cash compute.

Repeated rejection is therefore safe, but it is no longer allowed to be scientifically inert: repeated zero-gain rejection changes the next research strategy.

## Remote execution and persistence

`.github/workflows/autonomous-improvement.yml` runs only on GitHub-hosted `ubuntu-latest`, synchronizes serialized runs to current `main`, evaluates the incumbent, plans from aggregate metrics plus committed history, trains primary + replica, applies immutable evaluation/gates, and persists authoritative results.

Each completed cycle records its strategy, prior same-incumbent rejection streak, applied/retired research hints, schedule accounting, and gate result under `research/autonomous/cycles/`. `research/autonomous/state.json` records the latest strategy/escalation metadata in addition to the incumbent/difficulty/cycle pointer.

Rejected candidate weights remain ephemeral. A passing candidate alone may be copied to `checkpoints/genesis-autonomous-incumbent.pt` after the immutable gate passes.

## Anti-reward-hacking boundary

The planner cannot inspect or rewrite private holdout answers, graders, or the promotion gate. Research escalation chooses only among versioned repository implementations that are reviewed by CI and committed to `main`. Screening, tournament, Kaggle, and Colab outputs have no direct promotion authority.
