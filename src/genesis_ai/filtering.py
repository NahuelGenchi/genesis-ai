from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator

from genesis_ai.ingest import IngestError, sha256_file

WHITESPACE_RE = re.compile(r"\s+")


def canonical_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def exact_fingerprint(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def quality_reason(text: str, min_chars: int = 40) -> str | None:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return "too_short"
    if "\x00" in text:
        return "null_byte"

    control = sum(1 for char in text if unicodedata.category(char) == "Cc" and char not in "\n\r\t")
    if control / max(1, len(text)) > 0.01:
        return "control_chars"

    longest_run = 1
    current_run = 1
    previous = ""
    for char in text:
        if char == previous:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
            previous = char
    if longest_run >= 128:
        return "repeated_character_run"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.4:
        return "repeated_lines"
    return None


def _load_manifest(input_dir: Path) -> dict:
    path = input_dir / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise IngestError(f"cannot read input manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid input manifest: {exc.msg}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("shards"), list):
        raise IngestError("input manifest must contain a shards array")
    return value


def iter_input_documents(input_dir: Path) -> Iterator[dict]:
    manifest = _load_manifest(input_dir)
    observed = 0
    for shard in manifest["shards"]:
        if not isinstance(shard, dict) or not isinstance(shard.get("file"), str):
            raise IngestError("invalid shard record in input manifest")
        path = input_dir / shard["file"]
        if not path.is_file():
            raise IngestError(f"missing input shard: {path.name}")
        expected_size = shard.get("size_bytes")
        if isinstance(expected_size, int) and path.stat().st_size != expected_size:
            raise IngestError(f"size mismatch for input shard: {path.name}")
        expected_hash = shard.get("sha256")
        if isinstance(expected_hash, str) and sha256_file(path) != expected_hash:
            raise IngestError(f"checksum mismatch for input shard: {path.name}")

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IngestError(f"invalid JSONL in {path.name}:{line_number}") from exc
                if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                    raise IngestError(f"invalid document in {path.name}:{line_number}")
                observed += 1
                yield document

    expected_documents = manifest.get("documents")
    if isinstance(expected_documents, int) and observed != expected_documents:
        raise IngestError(f"document count mismatch: expected {expected_documents}, got {observed}")


def _json_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def filter_corpus(
    input_dir: Path,
    output_dir: Path,
    *,
    min_chars: int = 40,
    docs_per_shard: int = 10_000,
) -> dict[str, object]:
    if min_chars < 0:
        raise IngestError("min_chars must be non-negative")
    if docs_per_shard <= 0:
        raise IngestError("docs_per_shard must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("shard-*.jsonl"):
        stale.unlink()
    for stale_name in ("manifest.json", "metrics.json"):
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()

    seen: set[str] = set()
    reasons: Counter[str] = Counter()
    input_documents = 0
    kept_documents = 0
    shard_index = 0
    shard_docs = 0
    shard_records: list[dict[str, object]] = []
    handle = None
    shard_path: Path | None = None

    def open_shard() -> None:
        nonlocal handle, shard_path, shard_docs
        shard_path = output_dir / f"shard-{shard_index:05d}.jsonl"
        handle = shard_path.open("wb")
        shard_docs = 0

    def close_shard() -> None:
        nonlocal handle, shard_path
        if handle is None or shard_path is None:
            return
        handle.close()
        shard_records.append({
            "file": shard_path.name,
            "documents": shard_docs,
            "sha256": sha256_file(shard_path),
            "size_bytes": shard_path.stat().st_size,
        })
        handle = None
        shard_path = None

    try:
        for document in iter_input_documents(input_dir):
            input_documents += 1
            text = document["text"]
            reason = quality_reason(text, min_chars=min_chars)
            if reason is None:
                fingerprint = exact_fingerprint(text)
                if fingerprint in seen:
                    reason = "exact_duplicate"
                else:
                    seen.add(fingerprint)
            if reason is not None:
                reasons[reason] += 1
                continue

            if handle is None:
                open_shard()
            if shard_docs == docs_per_shard:
                close_shard()
                shard_index += 1
                open_shard()
            assert handle is not None
            handle.write(_json_line(document))
            shard_docs += 1
            kept_documents += 1
    finally:
        close_shard()

    metrics: dict[str, object] = {
        "format_version": "1.0",
        "input_documents": input_documents,
        "kept_documents": kept_documents,
        "dropped_documents": input_documents - kept_documents,
        "drop_reasons": dict(sorted(reasons.items())),
        "retention_ratio": round(kept_documents / input_documents, 6) if input_documents else 0.0,
        "min_chars": min_chars,
    }
    manifest: dict[str, object] = {
        "format_version": "1.0",
        "documents": kept_documents,
        "shards": shard_records,
        "filter_metrics": "metrics.json",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and conservatively filter Genesis AI corpus shards.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--docs-per-shard", type=int, default=10_000)
    args = parser.parse_args()
    try:
        metrics = filter_corpus(
            args.input_dir,
            args.output_dir,
            min_chars=args.min_chars,
            docs_per_shard=args.docs_per_shard,
        )
    except IngestError as exc:
        parser.error(str(exc))
    print(
        f"kept {metrics['kept_documents']}/{metrics['input_documents']} documents; "
        f"dropped {metrics['dropped_documents']}"
    )


if __name__ == "__main__":
    main()
