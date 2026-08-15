# Continuous free-compute orchestration

Issue: #194. Parent autonomy contract: #142.

## Production loop

`Autonomous verified improvement` remains the only workflow with checkpoint-promotion authority. A successful completed run triggers `Autonomous cycle continuation`, which immediately dispatches the next run on `main`. `Autonomous continuity watchdog` checks every 15 minutes and restarts the chain only when no canonical run is queued or executing. The watchdog also runs when its continuity wiring lands on `main`, so deployment bootstraps the first canonical cycle immediately instead of waiting for cron.

Canonical training remains GitHub-hosted `ubuntu-latest`, deterministic CPU, independently reproduced, frozen-suite evaluated, and promotion-gated. Cash compute cost is `$0`.

## CPU research farm

The public GitHub-hosted CPU farm runs every three hours. It screens bounded architecture, optimizer, tiny-model, tokenizer, filtering, evaluation, and verifier variants. Results are screening-only and cannot promote checkpoints. The latest aggregate evidence is persisted at `research/accelerators/cpu-farm-latest.json` and also uploaded as a workflow artifact.

## Kaggle GPU

Every successful CPU-farm completion triggers the Kaggle dispatcher. It selects the strongest committed, enabled `tiny-model` manifest that survived CPU screening. The current generic accelerator runner is intentionally restricted to this lane because it can faithfully express model geometry; architecture/optimizer winners are not silently translated into unsupported GPU jobs.

Kaggle dispatch is opportunistic. It requires repository secret `KAGGLE_API_TOKEN` and repository variable `KAGGLE_USERNAME`. If either is absent, the lane skips without blocking the canonical loop. The Kaggle runtime rebuilds the provenance-locked public corpus before training. GPU output has no promotion authority.

`Kaggle GPU result collector` checks hourly and, when available, downloads only `accelerator-record.json`, validates the non-promotion contract, and persists the latest record at `research/accelerators/kaggle-latest.json`. Model weights remain outside the incumbent path until a separately defined frozen evaluation/reproduction experiment authorizes their use.

## Free Google Colab

Free Colab is an interactive sidecar, not an unattended worker. `accelerators/colab/genesis_colab.ipynb` reads the latest persisted CPU shortlist, auto-selects an eligible committed tiny-model manifest, rebuilds the locked public corpus, and remains dry until the active notebook user sets `RUN=True`.

This restriction is intentional: free managed Colab prioritizes interactive notebook use, has dynamic/non-guaranteed resource limits, and is not used here as a headless distributed-compute worker.

## Failure behavior

- Candidate/reproduction/evaluation/gate failure: no promotion; watchdog can restart the canonical chain.
- CPU farm failure: canonical improvement continues; no accelerator dispatch from that farm run.
- No CPU winner: Kaggle skips.
- Missing Kaggle credentials/quota: Kaggle skips.
- Colab unavailable: no effect on canonical autonomy.
- No research lane may weaken frozen evaluation or promote weights directly.
