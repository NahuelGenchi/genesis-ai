import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    milestones = json.loads((ROOT / "tracking/milestones.json").read_text())
    issues = json.loads((ROOT / "tracking/issues.json").read_text())
    labels = json.loads((ROOT / "tracking/labels.json").read_text())
    milestone_titles = {item["title"] for item in milestones}
    label_names = {item["name"] for item in labels}
    if len(milestone_titles) != len(milestones):
        raise SystemExit("duplicate milestone title")
    if len(label_names) != len(labels):
        raise SystemExit("duplicate label name")
    issue_titles: set[str] = set()
    for index, issue in enumerate(issues, start=1):
        missing = {"milestone", "title", "body"} - issue.keys()
        if missing:
            raise SystemExit(f"issue {index} missing fields: {sorted(missing)}")
        if issue["title"] in issue_titles:
            raise SystemExit(f"duplicate issue title: {issue['title']}")
        issue_titles.add(issue["title"])
        if issue["milestone"] not in milestone_titles:
            raise SystemExit(f"issue {index} references unknown milestone")
        unknown_labels = set(issue.get("labels", [])) - label_names
        if unknown_labels:
            raise SystemExit(f"issue {index} references unknown labels: {sorted(unknown_labels)}")
        if issue.get("state", "open") not in {"open", "closed"}:
            raise SystemExit(f"issue {index} has invalid state")
        for required in ("**Goal**", "**Why**", "**Done when**"):
            if required not in issue["body"]:
                raise SystemExit(f"issue {index} missing {required}")
    print(f"tracking ok: {len(milestones)} milestones, {len(issues)} issues, {len(labels)} labels")


if __name__ == "__main__":
    main()
