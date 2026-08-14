# Self-Hosted Runner

Genesis AI does not require paid GitHub-hosted compute.

## Setup

1. Open **Repository → Settings → Actions → Runners**.
2. Select **New self-hosted runner**.
3. Choose **Linux / x64**.
4. Run GitHub's generated setup commands on a trusted local machine.
5. Ensure `python` and PyTorch are available.
6. Start the runner.

## What it runs

- `Bootstrap tracking` — reconciles labels, Milestones, and Issues from `tracking/`.
- `CI` — runs `./scripts/test_local.sh`.

## Security

Use only a trusted machine. Keep the repository private while this runner is attached. Do not execute untrusted fork PRs on it.

## Cost

Required cloud spend: **$0**.
