from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "BENCHMARKS.md"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def render() -> str:
    tiny = load("models/genesis-tiny-v0/metrics.json")
    micro = load("models/genesis-micro-2m-v1/metrics.json")
    farm = load("experiments/cpu-farm-v1.json")

    tiny_loss = tiny["evaluation"]["loss"]
    micro_loss = micro["general_language"]["candidate_validation_loss"]
    capability = micro["capability"]
    training = micro["training"]
    candidates = farm["candidates"]

    lines = [
        "# Benchmark & progress dashboard",
        "",
        "> Generated from committed model metrics and experiment definitions. Run `python3 scripts/render_progress.py` after changing source evidence.",
        "",
        "## Promoted progress",
        "",
        "| Metric | `genesis-tiny-v0` | `genesis-micro-2m-v1` |",
        "|---|---:|---:|",
        f"| Parameters | {tiny['parameter_count']:,} | {micro['parameter_count']:,} |",
        f"| General validation loss (M3) | {tiny_loss:.6f} | {micro_loss:.6f} |",
        f"| Restricted expression exact accuracy | — | {capability['candidate_exact_accuracy'] * 100:.2f}% |",
        f"| Termination rate | — | {capability['candidate_termination_rate'] * 100:.2f}% |",
        f"| Processed training tokens | {tiny['training']['train_tokens']:,} corpus tokens | {training['processed_tokens']:,} processed tokens |",
        f"| Required cash compute | not recorded | ${training['cash_compute_cost_usd']:.2f} |",
        "",
        "The promoted micro checkpoint demonstrates **restricted integer-expression synthesis at difficulty 1**. It is not evidence of broad coding, factual, agentic, or frontier capability.",
        "",
        "## CPU research farm",
        "",
        f"- Definition: `{farm['farm_version']}`",
        f"- Candidate/guard jobs: **{len(candidates)}**",
        f"- Maximum parallel jobs: **{farm['max_parallel']}**",
        f"- Per-job timeout: **{farm['timeout_minutes']} minutes**",
        f"- Runner: **{farm['runner']}**",
        f"- Paid runners allowed: **{'yes' if farm['paid_runners_allowed'] else 'no'}**",
        "- CPU screening has no checkpoint-promotion authority.",
        "",
        "## Free accelerator ladder",
        "",
        "1. GitHub-hosted CPU farm screens cheap hypotheses.",
        "2. Only model-side CPU winners may enter Kaggle/Colab accelerator jobs.",
        "3. Accelerator outputs remain non-promoted until frozen evaluation, contamination, regression, and promotion gates pass.",
        "4. Kaggle and Colab availability is opportunistic; neither is required for repository correctness.",
        "",
        "## Evidence sources",
        "",
        "- `models/genesis-tiny-v0/metrics.json`",
        "- `models/genesis-micro-2m-v1/metrics.json`",
        "- `experiments/cpu-farm-v1.json`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the evidence-backed Genesis benchmark dashboard.")
    parser.add_argument("--check", action="store_true", help="Fail if docs/BENCHMARKS.md is stale")
    args = parser.parse_args()
    rendered = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            raise SystemExit("docs/BENCHMARKS.md is stale; run python3 scripts/render_progress.py")
        print("benchmark dashboard is current")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
