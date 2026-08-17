from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .architecture_tournament import _candidate_config, _load_json, run_candidate
from .ingest import sha256_file
from .tokenizer import ByteBPETokenizer

FINALIST_VERSION = "m6-architecture-finalist-v1"
BASELINE_NAME = "baseline-learned-layernorm-gelu"
FRESH_SEEDS = (102002, 102003)
MIN_MEAN_RELATIVE_IMPROVEMENT = 0.005
MAX_PER_SEED_RELATIVE_REGRESSION = 0.01


def _definition_candidate(definition: dict[str, Any], name: str) -> dict[str, Any]:
    for candidate in definition["candidates"]:
        if candidate.get("name") == name:
            return candidate
    raise ValueError(f"candidate not found in frozen definition: {name}")


def run_finalist(
    *,
    tournament_path: str | Path,
    definition_path: str | Path,
    public_data: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    tournament_path = Path(tournament_path)
    definition_path = Path(definition_path)
    public_data = Path(public_data)
    tokenizer_path = Path(tokenizer_path)
    tournament = _load_json(tournament_path)
    definition = _load_json(definition_path)
    if tournament.get("tournament_version") != "m6-architecture-tournament-v1":
        raise ValueError("unsupported tournament result")
    if tournament.get("definition_sha256") != sha256_file(definition_path):
        raise ValueError("tournament result does not bind the frozen definition")
    winner = str(tournament.get("winner"))
    baseline = str(tournament.get("baseline"))
    if baseline != BASELINE_NAME:
        raise ValueError("unexpected tournament baseline")

    baseline_definition = _definition_candidate(definition, baseline)
    winner_definition = _definition_candidate(definition, winner)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    flop_budget = int(definition["training_flop_budget"])
    target_tokens_per_step = int(definition["target_tokens_per_step"])
    learning_rate = float(definition["learning_rate"])
    validation_batches = int(definition["validation_batches"])

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    repetitions: list[dict[str, Any]] = []
    names = [baseline] if winner == baseline else [baseline, winner]
    for seed in FRESH_SEEDS:
        for name in names:
            candidate_definition = baseline_definition if name == baseline else winner_definition
            repetitions.append(
                run_candidate(
                    name=name,
                    config=_candidate_config(candidate_definition["config"]),
                    public_data=public_data,
                    tokenizer=tokenizer,
                    flop_budget=flop_budget,
                    target_tokens_per_step=target_tokens_per_step,
                    learning_rate=learning_rate,
                    validation_batches=validation_batches,
                    seed=seed,
                    device=device,
                )
            )

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for result in repetitions:
        grouped[str(result["name"])].append(result)
    summary = {
        name: {
            "mean_final_validation_loss": sum(float(item["final_validation_loss"]) for item in items) / len(items),
            "mean_tokens_per_second": sum(float(item["training_tokens_per_second"]) for item in items) / len(items),
            "seeds": list(FRESH_SEEDS),
            "final_validation_losses": [float(item["final_validation_loss"]) for item in items],
        }
        for name, items in grouped.items()
    }

    if winner == baseline:
        accepted = baseline
        decision = {
            "reason": "baseline won the initial fixed-FLOP tournament",
            "mean_relative_improvement": 0.0,
            "per_seed_relative_deltas": [0.0 for _ in FRESH_SEEDS],
            "passed": True,
        }
    else:
        baseline_losses = summary[baseline]["final_validation_losses"]
        winner_losses = summary[winner]["final_validation_losses"]
        baseline_mean = float(summary[baseline]["mean_final_validation_loss"])
        winner_mean = float(summary[winner]["mean_final_validation_loss"])
        mean_relative_improvement = 1.0 - winner_mean / baseline_mean
        per_seed_relative_deltas = [
            candidate_loss / baseline_loss - 1.0
            for candidate_loss, baseline_loss in zip(winner_losses, baseline_losses)
        ]
        passed = (
            mean_relative_improvement >= MIN_MEAN_RELATIVE_IMPROVEMENT
            and all(delta <= MAX_PER_SEED_RELATIVE_REGRESSION for delta in per_seed_relative_deltas)
        )
        accepted = winner if passed else baseline
        decision = {
            "reason": "fresh-seed reproduction gate",
            "mean_relative_improvement": mean_relative_improvement,
            "per_seed_relative_deltas": per_seed_relative_deltas,
            "requirement": {
                "minimum_mean_relative_improvement": MIN_MEAN_RELATIVE_IMPROVEMENT,
                "maximum_per_seed_relative_regression": MAX_PER_SEED_RELATIVE_REGRESSION,
            },
            "passed": passed,
        }

    return {
        "format_version": "1.0",
        "finalist_version": FINALIST_VERSION,
        "tournament_sha256": sha256_file(tournament_path),
        "definition_sha256": sha256_file(definition_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "public_manifest_sha256": sha256_file(public_data / "manifest.json"),
        "initial_winner": winner,
        "baseline": baseline,
        "accepted_architecture": accepted,
        "fresh_seeds": list(FRESH_SEEDS),
        "decision": decision,
        "summary": summary,
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the fixed-FLOP architecture finalist on fresh seeds.")
    parser.add_argument("--tournament", type=Path, required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_finalist(
        tournament_path=args.tournament,
        definition_path=args.definition,
        public_data=args.public_data,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
