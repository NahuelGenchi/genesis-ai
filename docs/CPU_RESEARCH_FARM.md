# Public CPU research farm

Genesis uses standard public GitHub-hosted `ubuntu-latest` runners to reject weak ideas before expensive training.

## Contract

- Public repository only.
- Standard `ubuntu-latest` only; no larger/paid runner labels.
- At most 20 screening jobs, 12 concurrent, 12 minutes/job.
- Fixed seeds and bounded per-candidate compute.
- No repository secrets and no pull-request trigger.
- Artifacts are tiny JSON records; per-job records expire after 1 day.
- Screening results are always `promotion_eligible: false`.
- A CPU winner may only become **eligible for a later expensive/GPU experiment**. It still needs independent reproduction, frozen evaluation, contamination checks, regression gates, and the normal promotion process.

## Lanes

`architecture`, `optimizer`, and `tiny-model` use equal estimated training-FLOP budgets on a deterministic synthetic screening stream. `tokenizer` compares reversible compression on a fixed multilingual/code fixture. `data-filtering` scores quality classification on a labelled fixture. `evaluation` and `verifier` are guard lanes and must score 100%.

The frozen matrix is `experiments/cpu-farm-v1.json`. Changing that definition or relevant implementation files triggers the farm automatically. A weekly run checks reproducibility/environment drift without turning identical work into a continuous busy-loop.

The aggregate artifact `cpu-farm-summary` contains lane winners and `expensive_stage_eligible`. It has no checkpoint-promotion authority.
