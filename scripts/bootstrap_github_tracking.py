"""Idempotently create project labels, milestones, and bootstrap issues.

Uses only Python stdlib + the repository-scoped GITHUB_TOKEN supplied by
GitHub Actions. This is intentionally zero-dependency and zero-cost.
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
    separator = "&" if "?" in path else "?"
    return request("GET", f"{path}{separator}per_page=100", token) or []


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise SystemExit("GITHUB_TOKEN and GITHUB_REPOSITORY are required")

    labels_manifest = json.loads((ROOT / "tracking/labels.json").read_text())
    milestones_manifest = json.loads((ROOT / "tracking/milestones.json").read_text())
    issues_manifest = json.loads((ROOT / "tracking/issues.json").read_text())

    existing_labels = {item["name"] for item in get_all(f"/repos/{repo}/labels", token)}
    for label in labels_manifest:
        if label["name"] not in existing_labels:
            request("POST", f"/repos/{repo}/labels", token, label)
            print(f"created label: {label['name']}")

    milestones = get_all(f"/repos/{repo}/milestones?state=all", token)
    by_title = {item["title"]: item for item in milestones}
    for item in milestones_manifest:
        if item["title"] not in by_title:
            created = request("POST", f"/repos/{repo}/milestones", token, item)
            by_title[item["title"]] = created
            print(f"created milestone: {item['title']}")

    existing_issues = get_all(f"/repos/{repo}/issues?state=all", token)
    issue_by_title = {
        item["title"]: item for item in existing_issues if "pull_request" not in item
    }
    for spec in issues_manifest:
        title = spec["title"]
        if title in issue_by_title:
            continue
        milestone_number = by_title[spec["milestone"]]["number"]
        payload = {
            "title": title,
            "body": spec["body"],
            "milestone": milestone_number,
            "labels": spec.get("labels", []),
        }
        created = request("POST", f"/repos/{repo}/issues", token, payload)
        issue_by_title[title] = created
        print(f"created issue #{created['number']}: {title}")
        if spec.get("state") == "closed":
            request(
                "PATCH",
                f"/repos/{repo}/issues/{created['number']}",
                token,
                {"state": "closed", "state_reason": "completed"},
            )
            print(f"closed issue #{created['number']}")


if __name__ == "__main__":
    main()
