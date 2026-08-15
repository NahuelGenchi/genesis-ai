# Public roadmap

GitHub Milestones and Issues remain the source of truth. This page is the fast human map.

| Milestone | Outcome | Contributor entry point |
|---|---|---|
| [M0 — Bootstrap & Rules](https://github.com/NahuelGenchi/genesis-ai/milestone/1) | Reproducible repository, tracking, CI, independence rules | CI/docs/process fixes |
| [M1 — Data & Tokenizer](https://github.com/NahuelGenchi/genesis-ai/milestone/2) | Provenance-first corpus + tokenizer | Data validation and tokenizer analysis |
| [M2 — Tiny Baseline Model](https://github.com/NahuelGenchi/genesis-ai/milestone/3) | End-to-end tiny model | Training/evaluation reproducibility |
| [M3 — Evaluation Lab](https://github.com/NahuelGenchi/genesis-ai/milestone/4) | Frozen quality/regression/efficiency measurement | New deterministic benchmarks |
| [M4 — Efficiency Research](https://github.com/NahuelGenchi/genesis-ai/milestone/5) | Better quality per FLOP | CPU-screenable architecture ideas |
| [M5 — Self-Improvement Loop](https://github.com/NahuelGenchi/genesis-ai/milestone/6) | Challenger → verifier → candidate → promotion | Verifiers and anti-reward-hacking tests |
| [M6 — Scale Useful Capability](https://github.com/NahuelGenchi/genesis-ai/milestone/7) | Scale only measured winners | CPU farm + free accelerator experiments |
| [M7 — Free Serving & Community](https://github.com/NahuelGenchi/genesis-ai/milestone/8) | Reproducible demos/releases/contribution workflow | `good first issue` + `help wanted` |
| [M8 — Frontier Efficiency](https://github.com/NahuelGenchi/genesis-ai/milestone/9) | Long-horizon capability-per-dollar research | Hard benchmark and inference-efficiency work |

## Current research loop

```text
Issue
  ↓
cheap deterministic test
  ↓
GitHub CPU research farm
  ↓ only measured model winners
Kaggle / Colab accelerator experiment
  ↓
frozen evaluation + contamination + regression gates
  ↓
promotion decision
```

## Where help matters most

- **Benchmarks:** add hard, deterministic, contamination-resistant tasks.
- **Data:** improve provenance, filtering, deduplication, and multilingual coverage using legally usable sources.
- **Efficiency:** propose architecture/optimizer changes that can be screened cheaply before GPU time.
- **Reproducibility:** make every result one-command repeatable.
- **Community:** improve dashboards, docs, onboarding, and public demonstrations without overstating capability.

## Rules for research claims

A result is not a model improvement merely because training loss moved. Capability claims require the frozen evaluation evidence defined by the relevant Issue. Accelerator runs never bypass promotion gates.

Find starter work: https://github.com/NahuelGenchi/genesis-ai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22

Find help-wanted work: https://github.com/NahuelGenchi/genesis-ai/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22
