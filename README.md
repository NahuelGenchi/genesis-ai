# Genesis AI

[![CI](https://github.com/NahuelGenchi/genesis-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/NahuelGenchi/genesis-ai/actions/workflows/ci.yml)
[![CPU research farm](https://github.com/NahuelGenchi/genesis-ai/actions/workflows/cpu-research-farm.yml/badge.svg)](https://github.com/NahuelGenchi/genesis-ai/actions/workflows/cpu-research-farm.yml)

Genesis AI is a **$0-required-spend, from-scratch AI research project** building an independent model family and optimizing capability per unit of compute.

**Research loop:** start tiny → measure → screen on free CPU → use free accelerators only for winners → promote only after frozen evaluation gates.

## Current evidence

| Item | Current result |
|---|---:|
| Promoted research model | `genesis-micro-2m-v1` |
| Parameters | 1,895,808 |
| Demonstrated domain | restricted integer-expression synthesis, difficulty 1 |
| Frozen exact accuracy | 95.00% |
| Termination rate | 100.00% |
| General M3 validation loss | 3.355112 |
| Required cash compute for promoted run | $0.00 |

This is a **research checkpoint, not a general-purpose assistant**. The demonstrated domain result does not imply broad coding, factual, agentic, or frontier capability.

→ [Benchmark & progress dashboard](docs/BENCHMARKS.md)  
→ [Public roadmap](docs/ROADMAP.md)  
→ [Current model card](models/genesis-micro-2m-v1/MODEL_CARD.md)

## Free compute ladder

Genesis deliberately separates cheap screening from scarce accelerator time.

1. **GitHub Actions CPU farm** — 19 bounded architecture/tokenizer/filter/eval/verifier/optimizer/tiny-model screens, up to 12 in parallel.
2. **Kaggle GPU** — CPU-screen-gated cloud dispatch to the free Tesla P100 pool. Kaggle currently documents a 30-hour weekly GPU quota or sometimes higher depending on demand/resources.
3. **Google Colab** — optional GPU capacity when available; never a required dependency because quotas/hardware/runtime limits fluctuate.

No accelerator output can promote itself. It must pass the normal frozen evaluation, contamination, regression, and promotion gates.

→ [Free accelerator guide](docs/FREE_ACCELERATORS.md)  
→ [Open the Colab runner](https://colab.research.google.com/github/NahuelGenchi/genesis-ai/blob/main/accelerators/colab/genesis_colab.ipynb)

## Contribute

Outside contributors are welcome. The fastest entry points are deliberately small and reproducible:

- [Good first issues](https://github.com/NahuelGenchi/genesis-ai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [Help wanted](https://github.com/NahuelGenchi/genesis-ai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)
- [Public roadmap](docs/ROADMAP.md)
- [Contribution workflow](CONTRIBUTING.md)
- [Benchmark dashboard](docs/BENCHMARKS.md)

A useful contribution can be a benchmark, verifier, data-quality improvement, tiny CPU experiment, reproducibility fix, documentation improvement, or efficiency idea. **You do not need a GPU to contribute.**

## Hard constraints

- **$0 required spend.** Paid compute, paid APIs, and paid services are not project requirements.
- **Independent weights.** Models start from random initialization.
- **No proprietary-model dependency.** Training/evaluation/production must not require ChatGPT, Claude, or another proprietary model.
- **Evidence before scale.** Larger experiments happen only after smaller experiments justify them.
- **Everything is tracked.** Every task, bug, decision, experiment, documentation change, and enhancement belongs to a GitHub Issue and exactly one Milestone.
- **No hidden capability claims.** Public claims must match committed evaluation evidence.

## Reproduce locally

```bash
./scripts/test_local.sh
```

To refresh the evidence-backed dashboard after changing model metrics:

```bash
python3 scripts/render_progress.py
```

## Tracking rule

GitHub **Milestones + Issues are the source of truth**. No meaningful project work should exist only in chat, commits, pull requests, or documentation.

See [`PROJECT_RULES.md`](PROJECT_RULES.md).
