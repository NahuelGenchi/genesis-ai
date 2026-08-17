from pathlib import Path


CONTINUATION = Path(".github/workflows/autonomous-continuation.yml")
WATCHDOG = Path(".github/workflows/autonomous-watchdog.yml")


def test_continuation_checks_strategy_capacity_before_redispatch() -> None:
    workflow = CONTINUATION.read_text(encoding="utf-8")
    assert "genesis_ai.autonomy_dispatch_policy" in workflow
    assert "research/autonomous/cycles" in workflow
    assert "canonical candidate lane paused" in workflow
    assert "if: steps.policy.outputs.allowed == 'true'" in workflow
    assert "autonomous-improvement.yml/dispatches" in workflow


def test_watchdog_pauses_only_candidate_lane_and_keeps_cpu_research_alive() -> None:
    workflow = WATCHDOG.read_text(encoding="utf-8")
    assert "genesis_ai.autonomy_dispatch_policy" in workflow
    assert "research/autonomous/research-pause.json" in workflow
    assert "all same-incumbent strategies exhausted" in workflow
    assert "cpu-research-farm.yml" in workflow
    assert "timedelta(hours=4)" in workflow
    assert "canonical_allowed" in workflow
    assert "canonical candidate lane intentionally paused" in workflow


def test_watchdog_persists_pause_once_and_clears_it_after_new_capacity() -> None:
    workflow = WATCHDOG.read_text(encoding="utf-8")
    assert 'if [ "$EXHAUSTED" = "true" ]; then' in workflow
    assert 'elif [ -f "$PAUSE_MARKER" ]; then' in workflow
    assert 'rm "$PAUSE_MARKER"' in workflow
    assert "pause exhausted canonical research lane (#212)" in workflow
    assert "resume canonical research lane (#212)" in workflow


def test_new_strategy_catalog_or_new_incumbent_can_resume_without_manual_pc() -> None:
    policy = Path("src/genesis_ai/autonomy_dispatch_policy.py").read_text(encoding="utf-8")
    assert "RESEARCH_STRATEGIES" in policy
    assert 'gate.get("baseline_checkpoint_sha256") != incumbent_sha' in policy
    assert "unexhausted_research_strategies" in policy
    assert "all_predeclared_strategies_exhausted" in policy
