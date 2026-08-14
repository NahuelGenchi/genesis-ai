from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GCI_VERSION = "gci-v1"
GCI_DOMAINS = ("code", "math", "structured")


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evaluation result must be a JSON object")
    return value


def score_result(result: dict[str, Any]) -> dict[str, Any]:
    domains = result.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("evaluation result is missing domains")
    exact: dict[str, float] = {}
    for domain in GCI_DOMAINS:
        block = domains.get(domain)
        if not isinstance(block, dict):
            raise ValueError(f"evaluation result is missing {domain}")
        value = block.get("exact_accuracy")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid exact_accuracy for {domain}")
        exact[domain] = float(value)
    score = 100.0 * sum(exact.values()) / len(GCI_DOMAINS)
    return {
        "metric_version": GCI_VERSION,
        "suite_version": result.get("suite_version"),
        "suite_sha256": result.get("suite_sha256"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "domain_exact_accuracy": exact,
        "score": score,
    }


def compare_results(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    before = score_result(baseline)
    after = score_result(candidate)
    if before["suite_version"] != after["suite_version"] or before["suite_sha256"] != after["suite_sha256"]:
        raise ValueError("GCI comparison requires identical frozen suite version and hash")
    absolute = float(after["score"]) - float(before["score"])
    baseline_score = float(before["score"])
    relative = None if baseline_score == 0.0 else 100.0 * absolute / baseline_score
    return {
        "format_version": "1.0",
        "metric_version": GCI_VERSION,
        "baseline": before,
        "candidate": after,
        "absolute_point_change": absolute,
        "relative_percent_change": relative,
        "relative_percent_note": "N/A (zero baseline)" if relative is None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the auditable Genesis Capability Index (GCI-v1).")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_results(_load(args.baseline), _load(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
