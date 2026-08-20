"""Idempotently reconcile project labels, milestones, and durable Issues.

Uses only Python stdlib + the repository-scoped GITHUB_TOKEN supplied by
GitHub Actions. Tracking must work with the user's PC offline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"
CYCLE_ISSUE_PREFIX = "Autonomy research escalation from cycle "
CURRENT_RESEARCH_TITLE = "Current Research State — capability per compute"


def request(method: str, path: str, token: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "genesis-ai-bootstrap",
        },
    )
    try:
        with urllib.request.urlopen(req) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc


def get_all(path: str, token: str) -> list[dict]:
    """Fetch every page so reconciliation cannot duplicate older tracking Issues."""
    results: list[dict] = []
    page = 1
    while True:
        separator = "&" if "?" in path else "?"
        batch = request("GET", f"{path}{separator}per_page=100&page={page}", token) or []
        if not isinstance(batch, list):
            raise RuntimeError(f"GitHub list endpoint returned non-list: {path}")
        results.extend(batch)
        if len(batch) < 100:
            return results
        page += 1


def _load_issue_specs() -> list[dict]:
    specs = json.loads((ROOT / "tracking/issues.json").read_text())
    research_path = ROOT / "tracking/research_issues.json"
    if research_path.is_file():
        specs.extend(json.loads(research_path.read_text()))
    seen: set[str] = set()
    for spec in specs:
        title = spec["title"]
        if title in seen:
            raise RuntimeError(f"duplicate managed Issue title across manifests: {title}")
        seen.add(title)
    return specs


def _close_cycle_issues(repo: str, token: str, issues: list[dict]) -> None:
    """Cycle telemetry belongs in research/ records, not one Issue per attempt."""
    for issue in issues:
        if "pull_request" in issue:
            continue
        title = str(issue.get("title", ""))
        if issue.get("state") != "open" or not title.startswith(CYCLE_ISSUE_PREFIX):
            continue
        request(
            "PATCH",
            f"/repos/{repo}/issues/{issue['number']}",
            token,
            {"state": "closed", "state_reason": "not_planned"},
        )
        print(f"closed noisy cycle Issue #{issue['number']}; durable state lives in {CURRENT_RESEARCH_TITLE}")


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise SystemExit("GITHUB_TOKEN and GITHUB_REPOSITORY are required")

    labels_manifest = json.loads((ROOT / "tracking/labels.json").read_text())
    milestones_manifest = json.loads((ROOT / "tracking/milestones.json").read_text())
    issues_manifest = _load_issue_specs()

    existing_labels = {item["name"] for item in get_all(f"/repos/{repo}/labels", token)}
    for label in labels_manifest:
        if label["name"] not in existing_labels:
            request("POST", f"/repos/{repo}/labels", token, label)
            print(f"created label: {label['name']}")

    milestones = get_all(f"/repos/{repo}/milestones?state=all", token)
    by_title = {item["title"]: item for item in milestones}
    for item in milestones_manifest:
        existing = by_title.get(item["title"])
        if existing is None:
            created = request("POST", f"/repos/{repo}/milestones", token, item)
            by_title[item["title"]] = created
            print(f"created milestone: {item['title']}")
            continue
        patch = {}
        if existing.get("description") != item.get("description"):
            patch["description"] = item.get("description")
        if patch:
            updated = request("PATCH", f"/repos/{repo}/milestones/{existing['number']}", token, patch)
            by_title[item["title"]] = updated
            print(f"reconciled milestone: {item['title']}")

    existing_issues = get_all(f"/repos/{repo}/issues?state=all", token)
    _close_cycle_issues(repo, token, existing_issues)
    issue_by_title = {
        item["title"]: item for item in existing_issues if "pull_request" not in item
    }
    for spec in issues_manifest:
        title = spec["title"]
        milestone_number = by_title[spec["milestone"]]["number"]
        labels = spec.get("labels", [])
        desired_state = spec.get("state", "open")
        preserve_body = bool(spec.get("preserve_body", False))
        existing = issue_by_title.get(title)
        if existing is not None:
            patch: dict = {}
            if not preserve_body and existing.get("body") != spec["body"]:
                patch["body"] = spec["body"]
            if (existing.get("milestone") or {}).get("number") != milestone_number:
                patch["milestone"] = milestone_number
            existing_labels = sorted(item["name"] for item in existing.get("labels", []))
            if existing_labels != sorted(labels):
                patch["labels"] = labels
            if existing.get("state") != desired_state:
                patch["state"] = desired_state
                patch["state_reason"] = "completed" if desired_state == "closed" else "reopened"
            if patch:
                updated = request(
                    "PATCH", f"/repos/{repo}/issues/{existing['number']}", token, patch
                )
                issue_by_title[title] = updated
                print(f"reconciled issue #{existing['number']}: {title}")
            continue
        if preserve_body:
            raise RuntimeError(
                f"managed preserve_body Issue is missing and must be created explicitly first: {title}"
            )
        payload = {
            "title": title,
            "body": spec["body"],
            "milestone": milestone_number,
            "labels": labels,
        }
        created = request("POST", f"/repos/{repo}/issues", token, payload)
        issue_by_title[title] = created
        print(f"created issue #{created['number']}: {title}")
        if desired_state == "closed":
            request(
                "PATCH",
                f"/repos/{repo}/issues/{created['number']}",
                token,
                {"state": "closed", "state_reason": "completed"},
            )
            print(f"closed issue #{created['number']}")


if __name__ == "__main__":
    main()
