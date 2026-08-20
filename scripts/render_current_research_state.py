from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def render_body(root: Path = ROOT) -> str:
    state = _load(root / "research/autonomous/state.json")
    if state is None:
        raise ValueError("missing research/autonomous/state.json")
    diagnostic = _load(root / "research/m6-incumbent-structured-diagnostic/latest.json")
    funnel = _load(root / "research/weak-domain-funnel-v1/latest.json")

    incumbent = str(state["incumbent_checkpoint"])
    gci = float(state["incumbent_gci_v1"])
    status = str(state.get("autonomy_status", "unknown"))
    cycle = int(state.get("cycle_index", 0))
    breaker = state.get("circuit_breaker") if isinstance(state.get("circuit_breaker"), dict) else {}
    breaker_active = bool(breaker.get("active"))
    breaker_reason = str(breaker.get("reason", "none"))

    lines = [
        "## Purpose",
        "Canonical human-facing research dashboard generated from committed machine state. Do not edit capability numbers here by hand.",
        "",
        "## Promoted incumbent",
        f"- checkpoint: `{incumbent}`",
        f"- GCI-v1: **{gci:.4f}**",
        f"- difficulty: **{int(state.get('difficulty', 1))}**",
        f"- autonomous cycle index: **{cycle}**",
        f"- autonomy status: **`{status}`**",
        f"- research circuit breaker: **{'active' if breaker_active else 'inactive'}**",
    ]
    if breaker_active:
        lines.append(f"- hold reason: {breaker_reason}")

    if diagnostic is not None:
        block = diagnostic.get("diagnostic") if isinstance(diagnostic.get("diagnostic"), dict) else {}
        oracle = block.get("oracle_context_greedy") if isinstance(block.get("oracle_context_greedy"), dict) else {}
        free = block.get("free_generation") if isinstance(block.get("free_generation"), dict) else {}
        semantics = block.get("structured_semantics") if isinstance(block.get("structured_semantics"), dict) else {}
        quartiles = oracle.get("quartile_position_accuracy") if isinstance(oracle.get("quartile_position_accuracy"), list) else []
        late = float(quartiles[-1]) if quartiles else 0.0
        lines.extend(
            [
                "",
                "## Measured bottleneck",
                "Replicated incumbent diagnostic (#153):",
                f"- structured strict exact: **{int(free.get('strict_correct', 0))}/{int(block.get('task_count', 0))}**",
                f"- oracle-context next-token accuracy: **{100.0 * float(oracle.get('token_accuracy', 0.0)):.2f}%**",
                f"- late-quartile oracle accuracy: **{100.0 * late:.2f}%**",
                f"- termination: **{100.0 * float(free.get('termination_rate', 0.0)):.2f}%**",
                f"- valid JSON: **{100.0 * float(semantics.get('valid_json_rate', 0.0)):.2f}%**",
                "- conclusion: transformation/sequence learning is the bottleneck; formatting-only work is not the active hypothesis.",
            ]
        )

    lines.extend(
        [
            "",
            "## Active research lane",
            "#246 — **Weak-domain successive-halving research funnel**.",
            "",
            "Resource order:",
            "`diagnostic -> tiny 5% screens -> medium 25% survivors -> one full candidate -> independent replica -> immutable promotion gate`",
            "",
            "Tiny/medium screens have zero promotion authority. Full training is reserved for the strongest measured survivor.",
            "",
            "## Scaling policy",
            "#160 — scaling is paused until incumbent-scale math/structured learning shows positive capability per compute. The rejected first ~5M experiment remains historical evidence in #210.",
        ]
    )

    if funnel is not None:
        lines.extend(["", "## Latest weak-domain funnel result"])
        lines.append(f"- workflow run: `{funnel.get('workflow_run_id', 'unknown')}`")
        lines.append(f"- decision: **`{funnel.get('gate_decision', funnel.get('decision', 'unknown'))}`**")
        if funnel.get("winner_variant"):
            lines.append(f"- winning screened variant: `{funnel['winner_variant']}`")
        if isinstance(funnel.get("candidate_gci_v1"), (int, float)):
            lines.append(
                f"- GCI-v1: `{float(funnel.get('baseline_gci_v1', gci)):.4f} -> {float(funnel['candidate_gci_v1']):.4f}`"
            )
        lines.append(f"- adopted: **{bool(funnel.get('adopted', False))}**")
        lines.append(f"- screening processed tokens: `{int(funnel.get('screening_processed_tokens', 0)):,}`")

    lines.extend(
        [
            "",
            "## Tracking contract",
            "1. One durable Issue per hypothesis, blocker, or deliverable.",
            "2. Autonomous cycles are machine-readable `research/` records, not new Issues.",
            "3. #203 records only high-level strategy transitions.",
            "4. #248 is regenerated from committed state whenever tracked research state changes.",
            "5. Promotion claims require committed immutable-gate evidence.",
            "",
            "Refs #142 #153 #160 #194 #203 #210 #246",
            "",
            "_Generated by `scripts/render_current_research_state.py`._",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    print(render_body(), end="")


if __name__ == "__main__":
    main()
