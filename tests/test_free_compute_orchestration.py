import json
from pathlib import Path

from genesis_ai.accelerator_job import validate_job


def test_successful_autonomous_cycles_chain_immediately() -> None:
    workflow = Path(".github/workflows/autonomous-continuation.yml").read_text(encoding="utf-8")
    assert 'workflows: ["Autonomous verified improvement"]' in workflow
    assert "types: [completed]" in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "actions: write" in workflow
    assert "autonomous-improvement.yml/dispatches" in workflow
    assert '{"ref": "main"}' in workflow
    assert "self-hosted" not in workflow


def test_watchdog_recovers_an_idle_chain() -> None:
    workflow = Path(".github/workflows/autonomous-watchdog.yml").read_text(encoding="utf-8")
    assert 'cron: "7 * * * *"' in workflow
    assert "actions: write" in workflow
    assert "autonomous-improvement.yml/runs?per_page=20" in workflow
    assert '"queued", "in_progress", "waiting", "pending", "requested"' in workflow
    assert "autonomous-improvement.yml/dispatches" in workflow
    assert "self-hosted" not in workflow


def test_cpu_farm_runs_repeatedly_and_persists_research_evidence() -> None:
    workflow = Path(".github/workflows/cpu-research-farm.yml").read_text(encoding="utf-8")
    assert 'cron: "23 */3 * * *"' in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "contents: write" in workflow
    assert "research/accelerators/cpu-farm-latest.json" in workflow
    assert "promotion authority" in workflow.lower()
    assert "self-hosted" not in workflow


def test_kaggle_is_automatic_downstream_of_cpu_screening() -> None:
    workflow = Path(".github/workflows/kaggle-gpu-dispatch.yml").read_text(encoding="utf-8")
    assert 'workflows: ["CPU research farm"]' in workflow
    assert "github.event.workflow_run.conclusion == 'success'" in workflow
    assert "cpu-farm-summary" in workflow
    assert "tiny-model" in workflow
    assert "KAGGLE_API_TOKEN" in workflow
    assert "KAGGLE_USERNAME" in workflow
    assert "NvidiaTeslaP100" in workflow
    assert "--timeout 7200" in workflow
    assert "promotion_authority" in workflow


def test_kaggle_results_are_collected_without_promotion_authority() -> None:
    workflow = Path(".github/workflows/kaggle-gpu-collect.yml").read_text(encoding="utf-8")
    assert 'cron: "47 * * * *"' in workflow
    assert "accelerator-record.json" in workflow
    assert "promotion_authority" in workflow
    assert "promotion_eligible" in workflow
    assert "research/accelerators/kaggle-latest.json" in workflow


def test_enabled_accelerator_jobs_are_bounded_and_non_promoting() -> None:
    paths = sorted(Path("accelerators/jobs").glob("tiny-*.json"))
    assert paths
    for path in paths:
        job = json.loads(path.read_text(encoding="utf-8"))
        validate_job(job)
        assert job["cpu_screen"]["lane"] == "tiny-model"
        assert job["promotion_authority"] is False
        assert job["training"]["data"] == "runs/accelerator-public-corpus/filtered"
        assert job["training"]["steps"] <= 2000


def test_free_colab_remains_interactive_but_consumes_cpu_screening() -> None:
    notebook = json.loads(Path("accelerators/colab/genesis_colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "cpu-farm-latest.json" in source
    assert "RUN = False" in source
    assert "tiny-model" in source
    assert "promotion" in source.lower()
    assert "accelerator-public-corpus" in source
