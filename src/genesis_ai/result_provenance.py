from __future__ import annotations

import argparse
import json
from pathlib import Path


def bind_result(path: Path, source_commit: str, workflow_run_id: str) -> None:
    if not source_commit or not workflow_run_id:
        raise ValueError("source_commit and workflow_run_id are required")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("result must be a JSON object")
    value["result_provenance"] = {
        "source_commit": source_commit,
        "workflow_run_id": workflow_run_id,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind a generated research result to its triggering source commit.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    args = parser.parse_args()
    bind_result(args.path, args.source_commit, args.workflow_run_id)


if __name__ == "__main__":
    main()
