from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .improvement_controller import (
    LEGACY_STRATEGY_ID,
    MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY,
    RESEARCH_STRATEGIES,
)

DISPATCH_POLICY_VERSION = "autonomy-dispatch-policy-v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dispatch_decision(*, state_path: str | Path, history_root: str | Path) -> dict[str, Any]:
    state = _load(state_path)
    if state.get("state_version") != "autonomous-state-v1" or state.get("cash_compute_cost_usd") != 0.0:
        raise ValueError("invalid autonomous state for dispatch policy")
    incumbent_path = Path(str(state.get("incumbent_checkpoint", "")))
    if not incumbent_path.is_file():
        raise ValueError(f"autonomous incumbent checkpoint missing: {incumbent_path}")
    incumbent_sha = _sha256_file(incumbent_path)
    root = Path(history_root)
    if not root.is_dir():
        raise ValueError("autonomous history root is missing")

    strategy_zero_gain: dict[str, int] = {}
    cycles_considered = 0
    last_cycle_dir: str | None = None
    for cycle_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        plan_path = cycle_dir / "plan.json"
        gate_path = cycle_dir / "gate.json"
        if not plan_path.is_file() or not gate_path.is_file():
            continue
        plan = _load(plan_path)
        gate = _load(gate_path)
        if gate.get("baseline_checkpoint_sha256") != incumbent_sha:
            continue
        cycles_considered += 1
        last_cycle_dir = cycle_dir.as_posix()
        strategy = plan.get("research_strategy")
        strategy_id = LEGACY_STRATEGY_ID
        if isinstance(strategy, dict) and isinstance(strategy.get("id"), str):
            strategy_id = str(strategy["id"])
        focus_gain = gate.get("focus_absolute_gain", 0.0)
        if not isinstance(focus_gain, (int, float)) or isinstance(focus_gain, bool):
            raise ValueError(f"invalid focus gain in committed autonomous history: {cycle_dir}")
        if gate.get("decision") == "reject" and float(focus_gain) <= 0.0:
            strategy_zero_gain[strategy_id] = strategy_zero_gain.get(strategy_id, 0) + 1

    catalog = [str(strategy["id"]) for strategy in RESEARCH_STRATEGIES]
    legacy_exhausted = strategy_zero_gain.get(LEGACY_STRATEGY_ID, 0) >= MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY
    unexhausted = [
        strategy_id
        for strategy_id in catalog
        if strategy_zero_gain.get(strategy_id, 0) < MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY
    ]
    all_exhausted = legacy_exhausted and not unexhausted
    return {
        "format_version": "1.0",
        "dispatch_policy_version": DISPATCH_POLICY_VERSION,
        "canonical_training_allowed": not all_exhausted,
        "reason": (
            "all same-incumbent predeclared research strategies exhausted; wait for new strategy code or promoted incumbent"
            if all_exhausted
            else "at least one same-incumbent canonical strategy remains available"
        ),
        "incumbent_checkpoint": incumbent_path.as_posix(),
        "incumbent_checkpoint_sha256": incumbent_sha,
        "state_cycle_index": int(state.get("cycle_index", 0)),
        "cycles_considered_for_incumbent": cycles_considered,
        "maximum_zero_gain_attempts_per_strategy": MAX_ZERO_GAIN_ATTEMPTS_PER_STRATEGY,
        "legacy_strategy_id": LEGACY_STRATEGY_ID,
        "strategy_catalog": catalog,
        "strategy_zero_focus_gain_rejections": dict(sorted(strategy_zero_gain.items())),
        "unexhausted_research_strategies": unexhausted,
        "all_predeclared_strategies_exhausted": all_exhausted,
        "last_matching_cycle_dir": last_cycle_dir,
        "cash_compute_cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide whether canonical autonomous training may be dispatched.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = dispatch_decision(state_path=args.state, history_root=args.history_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    if args.check and not result["canonical_training_allowed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
