import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_issues() -> tuple[list[dict], int]:
    legacy = json.loads((ROOT / "tracking/issues.json").read_text())
    research_path = ROOT / "tracking/research_issues.json"
    research = json.loads(research_path.read_text()) if research_path.is_file() else []
    return [*legacy, *research], len(research)


def main() -> None:
    milestones = json.loads((ROOT / "tracking/milestones.json").read_text())
    issues, research_count = _load_issues()
    labels = json.loads((ROOT / "tracking/labels.json").read_text())
    milestone_titles = {item["title"] for item in milestones}
    label_names = {item["name"] for item in labels}
    if len(milestone_titles) != len(milestones):
        raise SystemExit("duplicate milestone title")
    if len(label_names) != len(labels):
        raise SystemExit("duplicate label name")
    issue_titles: set[str] = set()
    for index, issue in enumerate(issues, start=1):
        missing = {"milestone", "title"} - issue.keys()
        if missing:
            raise SystemExit(f"issue {index} missing fields: {sorted(missing)}")
        preserve_body = bool(issue.get("preserve_body", False))
        if not preserve_body and "body" not in issue:
            raise SystemExit(f"issue {index} missing body")
        if issue["title"] in issue_titles:
            raise SystemExit(f"duplicate issue title across tracking manifests: {issue['title']}")
        issue_titles.add(issue["title"])
        if issue["milestone"] not in milestone_titles:
            raise SystemExit(f"issue {index} references unknown milestone: {issue['milestone']}")
        unknown_labels = set(issue.get("labels", [])) - label_names
        if unknown_labels:
            raise SystemExit(f"issue {index} references unknown labels: {sorted(unknown_labels)}")
        if issue.get("state", "open") not in {"open", "closed"}:
            raise SystemExit(f"issue {index} has invalid state")
        if preserve_body:
            if "body" in issue:
                raise SystemExit(f"preserve_body issue {index} must not shadow its live body")
            continue
        for required in ("**Goal**", "**Why**", "**Done when**"):
            if required not in issue["body"]:
                raise SystemExit(f"issue {index} missing {required}")
    print(
        f"tracking ok: {len(milestones)} milestones, {len(issues)} managed issues "
        f"({research_count} durable research), {len(labels)} labels"
    )


if __name__ == "__main__":
    main()
