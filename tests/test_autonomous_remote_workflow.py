from pathlib import Path


def test_remote_autonomous_workflow_contract() -> None:
    workflow = Path(".github/workflows/autonomous-improvement.yml").read_text(encoding="utf-8")

    assert 'cron: "17 5 * * 2,5"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "self-hosted" not in workflow
    assert "timeout-minutes: 300" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "contents: write" in workflow
    assert 'test "$RUNNER_ENVIRONMENT" = "github-hosted"' in workflow

    required_modules = (
        "genesis_ai.terminated_eval",
        "genesis_ai.improvement_controller",
        "genesis_ai.autonomous_curriculum",
        "genesis_ai.autonomous_training",
        "genesis_ai.scale_repro",
        "genesis_ai.eval_lab",
        "genesis_ai.autonomous_gate",
    )
    for module in required_modules:
        assert module in workflow

    assert "research/accelerators/cpu-farm-latest.json" in workflow
    assert "--research-evidence" in workflow
    assert "checkpoints/genesis-autonomous-incumbent.pt" in workflow
    assert "if [ \"$PROMOTED\" = \"true\" ]" in workflow
    assert 'git push origin HEAD:main' in workflow


def test_autonomous_result_commits_do_not_retrigger_cycle() -> None:
    workflow = Path(".github/workflows/autonomous-improvement.yml").read_text(encoding="utf-8")
    push_block = workflow.split("  push:\n", 1)[1].split("\n\npermissions:", 1)[0]

    assert "research/autonomous" not in push_block
    assert "checkpoints/genesis-autonomous-incumbent.pt" not in push_block
    assert "models/genesis-autonomous-incumbent" not in push_block
