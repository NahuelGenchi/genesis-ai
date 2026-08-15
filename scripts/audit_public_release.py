from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

AUDIT_VERSION = "public-release-v1"
MAX_TEXT_BLOB_BYTES = 2_000_000

SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b")),
    ("github_fine_grained_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[0-9A-Za-z_-]{20,}\b")),
    ("openai_style_api_key", re.compile(r"\bsk-[0-9A-Za-z_-]{24,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?im)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|secret[_-]?key)\b"
            r"\s*[:=]\s*[\"']?([A-Za-z0-9_./+\-=]{16,})"
        ),
    ),
)

SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
DATABASE_SUFFIXES = {".sqlite", ".sqlite3", ".db", ".dump"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".onnx"}
DATASET_SUFFIXES = {".csv", ".jsonl", ".parquet", ".arrow"}
THIRD_PARTY_MODEL_TERMS = {"claude", "anthropic", "openai", "gpt", "gemma", "qwen", "llama", "mistral"}


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], text=text)


def _finding(*, severity: str, rule: str, location: str, detail: str) -> dict[str, str]:
    return {"severity": severity, "rule": rule, "location": location, "detail": detail}


def _secret_findings(text: str, *, location: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for rule, pattern in SECRET_RULES:
        if pattern.search(text):
            findings.append(
                _finding(
                    severity="blocker",
                    rule=rule,
                    location=location,
                    detail="secret-like value matched; value intentionally omitted",
                )
            )
    return findings


def _path_findings(path: str, *, size: int, oid: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    lower = path.lower()
    suffix = Path(path).suffix.lower()
    findings: list[dict[str, str]] = []
    model_artifacts: list[dict[str, Any]] = []

    name = Path(path).name.lower()
    if name == ".env" or name.startswith(".env.") or suffix in SENSITIVE_SUFFIXES:
        findings.append(_finding(severity="blocker", rule="sensitive_path", location=path, detail="sensitive credential/key path exists in Git history"))
    if suffix in DATABASE_SUFFIXES:
        findings.append(_finding(severity="blocker", rule="database_artifact", location=path, detail="database/dump artifact exists in Git history"))
    if path.startswith("data/raw/") and size > 0:
        findings.append(_finding(severity="blocker", rule="raw_training_data", location=path, detail="non-empty raw training artifact exists in Git history"))
    if path.startswith("data/") and suffix in DATASET_SUFFIXES and size > 0:
        findings.append(_finding(severity="review", rule="dataset_artifact", location=path, detail="dataset-like artifact requires explicit redistribution review"))
    if suffix in MODEL_SUFFIXES:
        model_artifacts.append({"path": path, "oid": oid, "size_bytes": size})
        allowed_genesis = path.startswith("checkpoints/genesis-") and suffix == ".pt"
        if not allowed_genesis:
            findings.append(_finding(severity="review", rule="model_artifact", location=path, detail="model artifact is not an explicitly allowed Genesis checkpoint path"))
        if any(term in lower for term in THIRD_PARTY_MODEL_TERMS):
            findings.append(_finding(severity="blocker", rule="third_party_model_name", location=path, detail="model artifact path references an external/proprietary model family"))
    return findings, model_artifacts


def scan_git_history() -> dict[str, Any]:
    raw = str(_git("rev-list", "--objects", "--all"))
    object_paths: dict[str, set[str]] = {}
    for line in raw.splitlines():
        oid, _, path = line.partition(" ")
        if path:
            object_paths.setdefault(oid, set()).add(path)

    findings: list[dict[str, str]] = []
    model_artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    scanned_text_blobs = 0
    scanned_blobs = 0
    seen_blobs: set[str] = set()

    for oid, paths in object_paths.items():
        if oid in seen_blobs:
            continue
        try:
            obj_type = str(_git("cat-file", "-t", oid)).strip()
        except subprocess.CalledProcessError:
            continue
        if obj_type != "blob":
            continue
        seen_blobs.add(oid)
        scanned_blobs += 1
        size = int(str(_git("cat-file", "-s", oid)).strip())
        for path in sorted(paths):
            path_results, models = _path_findings(path, size=size, oid=oid)
            findings.extend(path_results)
            for model in models:
                model_artifacts[(model["path"], model["oid"])] = model

        if size > MAX_TEXT_BLOB_BYTES:
            continue
        content = bytes(_git("cat-file", "blob", oid, text=False))
        if b"\x00" in content[:8192]:
            continue
        text = content.decode("utf-8", errors="replace")
        scanned_text_blobs += 1
        for path in sorted(paths):
            findings.extend(_secret_findings(text, location=f"git:{path}@{oid[:12]}"))

    return {
        "objects_with_paths": len(object_paths),
        "scanned_blobs": scanned_blobs,
        "scanned_text_blobs": scanned_text_blobs,
        "model_artifacts": sorted(model_artifacts.values(), key=lambda item: (item["path"], item["oid"])),
        "findings": findings,
    }


def _github_get(url: str, token: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "genesis-ai-public-release-audit",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _paged(url: str, token: str) -> list[Any]:
    values: list[Any] = []
    page = 1
    separator = "&" if "?" in url else "?"
    while True:
        batch = _github_get(f"{url}{separator}per_page=100&page={page}", token)
        if not isinstance(batch, list):
            raise ValueError(f"expected list from {url}")
        values.extend(batch)
        if len(batch) < 100:
            return values
        page += 1


def scan_github_text(repo: str, token: str) -> dict[str, Any]:
    base = f"https://api.github.com/repos/{repo}"
    findings: list[dict[str, str]] = []
    issues = _paged(f"{base}/issues?state=all", token)
    issue_comments = 0
    review_comments = 0
    reviews = 0

    for item in issues:
        number = int(item["number"])
        kind = "pr" if "pull_request" in item else "issue"
        findings.extend(_secret_findings(str(item.get("title") or ""), location=f"{kind}:#{number}:title"))
        findings.extend(_secret_findings(str(item.get("body") or ""), location=f"{kind}:#{number}:body"))

        comments = _paged(f"{base}/issues/{number}/comments", token)
        issue_comments += len(comments)
        for comment in comments:
            findings.extend(_secret_findings(str(comment.get("body") or ""), location=f"{kind}:#{number}:comment:{comment.get('id')}"))

        if kind == "pr":
            inline = _paged(f"{base}/pulls/{number}/comments", token)
            review_comments += len(inline)
            for comment in inline:
                findings.extend(_secret_findings(str(comment.get("body") or ""), location=f"pr:#{number}:review-comment:{comment.get('id')}"))
            submitted = _paged(f"{base}/pulls/{number}/reviews", token)
            reviews += len(submitted)
            for review in submitted:
                findings.extend(_secret_findings(str(review.get("body") or ""), location=f"pr:#{number}:review:{review.get('id')}"))

    secret_scanning_status: dict[str, Any]
    try:
        alerts = _paged(f"{base}/secret-scanning/alerts?state=open", token)
        secret_scanning_status = {"accessible": True, "open_alert_count": len(alerts)}
        if alerts:
            findings.append(_finding(severity="blocker", rule="github_secret_scanning_alert", location="GitHub secret scanning", detail=f"{len(alerts)} open alert(s); secret values intentionally omitted"))
    except urllib.error.HTTPError as exc:
        secret_scanning_status = {"accessible": False, "http_status": exc.code}

    return {
        "issues_and_prs_scanned": len(issues),
        "issue_comments_scanned": issue_comments,
        "review_comments_scanned": review_comments,
        "reviews_scanned": reviews,
        "secret_scanning": secret_scanning_status,
        "findings": findings,
    }


def current_tree_checks() -> dict[str, Any]:
    tracked = set(str(_git("ls-tree", "-r", "--name-only", "HEAD")).splitlines())
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    raw_files = [path for path in tracked if path.startswith("data/raw/") and path != "data/raw/.gitkeep"]
    if raw_files:
        blockers.append(_finding(severity="blocker", rule="current_raw_data", location="data/raw", detail=f"{len(raw_files)} non-placeholder raw file(s) tracked"))

    if "LICENSE" not in tracked and "LICENSE.md" not in tracked and "LICENSE.txt" not in tracked:
        warnings.append(_finding(severity="warning", rule="missing_license", location="repository root", detail="no explicit repository license found; this is a collaboration/reuse issue, not a secret leak"))

    try:
        ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    except FileNotFoundError:
        ci = ""
    if "pull_request:" in ci and "self-hosted" in ci:
        blockers.append(_finding(severity="blocker", rule="public_pr_self_hosted_runner", location=".github/workflows/ci.yml", detail="pull_request jobs execute on a self-hosted runner; harden before public visibility"))

    return {"tracked_paths": len(tracked), "blockers": blockers, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Genesis AI Git history and GitHub text before public release without printing secret values.")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required for Issue/PR text audit")

    git_result = scan_git_history()
    github_result = scan_github_text(args.repo, token)
    current = current_tree_checks()
    all_findings = [*git_result["findings"], *github_result["findings"], *current["blockers"]]
    blockers = [item for item in all_findings if item["severity"] == "blocker"]
    review = [item for item in all_findings if item["severity"] == "review"]

    result = {
        "format_version": "1.0",
        "audit_version": AUDIT_VERSION,
        "repository": args.repo,
        "git": git_result,
        "github_text": github_result,
        "current_tree": current,
        "summary": {
            "blocker_count": len(blockers),
            "review_count": len(review),
            "warning_count": len(current["warnings"]),
            "content_clear": len([item for item in blockers if item["rule"] != "public_pr_self_hosted_runner"]) == 0 and not review,
            "public_ready": len(blockers) == 0 and not review,
        },
        "privacy": "Secret values are never included in this report; findings contain rule IDs and locations only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if blockers or review:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
