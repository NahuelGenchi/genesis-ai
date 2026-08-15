# Free accelerator strategy

Genesis uses accelerators **after** cheap CPU screening, never instead of it.

## Order of operations

1. GitHub-hosted CPU research farm screens hypotheses.
2. A model-side candidate must appear in `expensive_stage_eligible` in the farm summary.
3. An Issue-backed `accelerators/jobs/*.json` manifest freezes the GPU experiment and budget.
4. Kaggle or Colab runs the exact manifest.
5. GPU output remains **non-promoted** until independent frozen evaluation, contamination, regression, and normal promotion gates pass.

The accelerator runner rejects data-filtering/tokenizer-only winners as GPU jobs and rejects any candidate absent from the referenced CPU shortlist.

## Kaggle

Kaggle currently documents free **NVIDIA Tesla P100** access. Its weekly GPU quota is **30 hours or sometimes higher**, depending on demand and available resources. Treat this as a dynamic free quota, not a permanent capacity guarantee.

Official source: https://www.kaggle.com/docs/efficient-gpu-usage

### Cloud-only dispatch

The repository includes `.github/workflows/kaggle-gpu-dispatch.yml`.

One-time repository setup:

- add `KAGGLE_USERNAME` as a GitHub Actions secret;
- add `KAGGLE_KEY` as a GitHub Actions secret.

Never commit Kaggle credentials.

To submit a run, use **Actions → Kaggle GPU dispatch → Run workflow** and provide:

- the successful CPU research-farm run ID;
- an enabled manifest under `accelerators/jobs/`.

The workflow downloads that exact CPU shortlist, re-validates eligibility, freezes the current Git commit into the Kaggle payload, and submits the kernel with `NvidiaTeslaP100`. The GitHub runner only performs dispatch; model training occurs on Kaggle.

Kaggle CLI reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md

## Google Colab

Colab provides free GPU access, but accelerator type, quotas, availability, and runtime limits are dynamic. Genesis therefore treats Colab as opportunistic capacity, **never as a required autonomous dependency**. We intentionally do not encode a fixed free-session lifetime.

Official entry point: https://colab.research.google.com/

Open the repository notebook:

https://colab.research.google.com/github/NahuelGenchi/genesis-ai/blob/main/accelerators/colab/genesis_colab.ipynb

The notebook:

- clones the public repository;
- detects GPU/TPU/CPU explicitly;
- requires the same CPU-screened job manifest and CPU summary as Kaggle;
- supports periodic training checkpoints through the normal `genesis-train` checkpoint format;
- can store output in Google Drive so a later Colab session can resume from `latest.pt`.

TPUs may be detected, but the current Genesis training stack does not enable PyTorch/XLA training. Select a GPU runtime for accelerator jobs rather than introducing an untested TPU path.

## Job manifest contract

Copy `accelerators/job.example.json` to `accelerators/jobs/<job-id>.json` only after opening an Issue. A runnable manifest must:

- set `enabled: true`;
- reference a model-side CPU-screen lane and surviving variant;
- specify deterministic seed/model/training parameters;
- bound total steps and checkpoint interval;
- set `promotion_authority: false`.

No free accelerator can directly promote a checkpoint.

## Current promoted checkpoint

`genesis-micro-2m-v1` keeps its existing frozen CPU training policy. This accelerator layer is for future experiments that survive CPU screening; it does not retroactively change the reproducibility contract of the promoted model.
