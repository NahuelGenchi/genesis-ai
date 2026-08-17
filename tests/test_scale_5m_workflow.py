from pathlib import Path


WORKFLOW = Path(".github/workflows/m6-scale-5m-rope.yml")


def test_full_scale_workflow_is_remote_bounded_and_zero_cash() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in workflow
    assert "self-hosted" not in workflow
    assert "timeout-minutes: 350" in workflow
    assert 'test "$RUNNER_ENVIRONMENT" = "github-hosted"' in workflow
    assert workflow.count("python -m genesis_ai.scale_5m_training") == 2
    assert "python -m genesis_ai.scale_repro" in workflow
    assert "python -m genesis_ai.scale_5m_gate" in workflow
    assert "python -m genesis_ai.gci_ladder score" in workflow


def test_scale_promotion_requires_incumbent_freshness_and_updates_autonomy_state() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'INCUMBENT_UNCHANGED="${FRESHNESS[1]}"' in workflow
    assert 'if [ "$GATE_PROMOTED" = "true" ] && [ "$INCUMBENT_UNCHANGED" = "true" ]; then' in workflow
    assert "export CURRENT_SHA INCUMBENT_UNCHANGED ADOPTED" in workflow
    assert 'state["incumbent_checkpoint"] = "checkpoints/genesis-scale-5m-rope-v1.pt"' in workflow
    assert "last_scale_promotion" in workflow
    assert "git push origin HEAD:main" in workflow
    assert "force" not in workflow.lower()


def test_scale_result_does_not_commit_rejected_candidate_weights() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [ "$ADOPTED" = "true" ]; then' in workflow
    assert "git add -f checkpoints/genesis-scale-5m-rope-v1.pt" in workflow
    assert "cp \"$WORK/primary.pt\" checkpoints/genesis-scale-5m-rope-v1.pt" in workflow
    assert "pull_request:" not in workflow
    assert "pull_request_target:" not in workflow
