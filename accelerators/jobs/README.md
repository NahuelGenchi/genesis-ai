# Accelerator jobs

Only **Issue-backed model candidates that survived the CPU research farm** belong here.

Workflow:

1. Copy `../job.example.json` to `<job-id>.json`.
2. Set the exact CPU-screen `lane` + `variant`.
3. Freeze seed, model, training, and budget values.
4. Set `enabled: true` only when the referenced CPU summary lists that pair in `expensive_stage_eligible`.
5. Run `genesis-accelerator validate --job ... --cpu-summary ...` before dispatch.

The validator accepts only model-side lanes (`architecture`, `optimizer`, `tiny-model`). Tokenizer/data-filter/evaluation/verifier results do not justify GPU training by themselves.

Every manifest must keep `promotion_authority: false`.
