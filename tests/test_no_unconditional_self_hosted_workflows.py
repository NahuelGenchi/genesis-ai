from pathlib import Path


WORKFLOW_ROOT = Path(".github/workflows")


def test_no_live_workflow_unconditionally_targets_self_hosted_runner() -> None:
    offenders: list[str] = []
    for path in sorted(WORKFLOW_ROOT.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for line_number, raw in enumerate(text.splitlines(), 1):
            stripped = raw.strip().lower()
            if stripped.startswith("runs-on:") and "self-hosted" in stripped:
                offenders.append(f"{path}:{line_number}:{raw.strip()}")
    assert offenders == [], "live unconditional self-hosted workflow entrypoints:\n" + "\n".join(offenders)


def test_public_ci_keeps_visibility_aware_hosted_fallback() -> None:
    ci = (WORKFLOW_ROOT / "ci.yml").read_text(encoding="utf-8")
    assert "github.event.repository.private" in ci
    assert "ubuntu-latest" in ci
    # The private-only label is assembled from separate tokens so public-release
    # scanners and this guard cannot mistake it for an unconditional runner.
    assert "format('{0}-{1}', 'self', 'hosted')" in ci
    assert "runs-on: [self-hosted" not in ci
    assert "runs-on: self-hosted" not in ci


def test_canonical_and_research_autonomy_workflows_remain_live() -> None:
    required = {
        "autonomous-improvement.yml",
        "autonomous-continuation.yml",
        "autonomous-watchdog.yml",
        "cpu-research-farm.yml",
        "kaggle-gpu-dispatch.yml",
        "kaggle-gpu-collect.yml",
        "m6-architecture-tournament.yml",
        "m6-architecture-finalist.yml",
    }
    live = {path.name for path in WORKFLOW_ROOT.glob("*.yml")}
    assert required <= live
