from pathlib import Path


def test_canonical_cycle_consumes_history_and_can_route_research_escalation() -> None:
    workflow = Path(".github/workflows/autonomous-improvement.yml").read_text(encoding="utf-8")

    assert "actions: write" in workflow
    assert "issues: write" in workflow
    assert '--history-root "$RESULT_ROOT"' in workflow
    assert "Track and route research escalation" in workflow
    assert 'gh issue view 203 --repo "$GITHUB_REPOSITORY"' in workflow
    assert "m6-architecture-tournament.yml" in workflow
    assert "research/m6-architecture-tournament-v1.json" in workflow
    assert 'research_strategy' in workflow
    assert 'retired_research_hints' in workflow


def test_architecture_tournament_is_public_zero_cash_research_only() -> None:
    workflow = Path(".github/workflows/m6-architecture-tournament.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in workflow
    assert "self-hosted" not in workflow
    assert "timeout-minutes: 90" in workflow
    assert 'test "$RUNNER_ENVIRONMENT" = "github-hosted"' in workflow
    assert "genesis_ai.architecture_tournament" in workflow
    assert "research/m6-architecture-tournament-v1.json" in workflow
    assert "genesis_ai.autonomous_gate" not in workflow
    assert "checkpoints/genesis-autonomous-incumbent.pt" not in workflow
    assert "promotion" not in workflow.lower()


def test_architecture_finalist_chains_from_tournament_without_promotion_authority() -> None:
    workflow = Path(".github/workflows/m6-architecture-finalist.yml").read_text(encoding="utf-8")

    assert 'workflows: ["M6 architecture tournament v1"]' in workflow
    assert "workflow_run:" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "self-hosted" not in workflow
    assert "timeout-minutes: 120" in workflow
    assert 'test "$RUNNER_ENVIRONMENT" = "github-hosted"' in workflow
    assert "genesis_ai.architecture_finalist" in workflow
    assert "research/m6-architecture-finalist-v1.json" in workflow
    assert "genesis_ai.autonomous_gate" not in workflow
    assert "checkpoints/genesis-autonomous-incumbent.pt" not in workflow


def test_research_workflows_do_not_accept_pull_request_code() -> None:
    for path in (
        ".github/workflows/m6-architecture-tournament.yml",
        ".github/workflows/m6-architecture-finalist.yml",
    ):
        workflow = Path(path).read_text(encoding="utf-8")
        assert "pull_request:" not in workflow
        assert "pull_request_target:" not in workflow
