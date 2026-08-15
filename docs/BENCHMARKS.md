# Benchmark & progress dashboard

> Generated from committed model metrics and experiment definitions. Run `python3 scripts/render_progress.py` after changing source evidence.

## Promoted progress

| Metric | `genesis-tiny-v0` | `genesis-micro-2m-v1` |
|---|---:|---:|
| Parameters | 394,560 | 1,895,808 |
| General validation loss (M3) | 3.686833 | 3.355112 |
| Restricted expression exact accuracy | — | 95.00% |
| Termination rate | — | 100.00% |
| Processed training tokens | 247,453 corpus tokens | 2,001,920 processed tokens |
| Required cash compute | not recorded | $0.00 |

The promoted micro checkpoint demonstrates **restricted integer-expression synthesis at difficulty 1**. It is not evidence of broad coding, factual, agentic, or frontier capability.

## CPU research farm

- Definition: `cpu-screen-v1`
- Candidate/guard jobs: **19**
- Maximum parallel jobs: **12**
- Per-job timeout: **12 minutes**
- Runner: **ubuntu-latest**
- Paid runners allowed: **no**
- CPU screening has no checkpoint-promotion authority.

## Free accelerator ladder

1. GitHub-hosted CPU farm screens cheap hypotheses.
2. Only model-side CPU winners may enter Kaggle/Colab accelerator jobs.
3. Accelerator outputs remain non-promoted until frozen evaluation, contamination, regression, and promotion gates pass.
4. Kaggle and Colab availability is opportunistic; neither is required for repository correctness.

## Evidence sources

- `models/genesis-tiny-v0/metrics.json`
- `models/genesis-micro-2m-v1/metrics.json`
- `experiments/cpu-farm-v1.json`
