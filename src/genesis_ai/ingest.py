from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_FORMATS = {"text", "code", "math", "structured-data"}


class IngestError(ValueError):
    """Raised when corpus metadata or content fails ingestion checks."""


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    path: Path
    sha256: str
    size_bytes: int
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    formats: tuple[str, ...]


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise IngestError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid JSON in {path}: {exc.msg}") from exc


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IngestError(f"{field} must be a non-empty string")
    return value


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise IngestError(f"{field} must be a non-empty array")
    result = tuple(_nonempty_string(item, field) for item in value)
    if len(set(result)) != len(result):
        raise IngestError(f"{field} must not contain duplicates")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: object, field: str) -> Path:
    raw = _nonempty_string(value, field)
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _load_source(metadata_path: Path, content_path: Path) -> Source:
    raw = _read_json(metadata_path)
    if not isinstance(raw, dict):
        raise IngestError(f"metadata must be an object: {metadata_path}")

    required = {
        "schema_version",
        "id",
        "name",
        "origin",
        "rights",
        "retrieval",
        "content",
        "filtering_steps",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise IngestError(f"metadata missing fields: {', '.join(missing)}")
    if raw["schema_version"] != "1.0":
        raise IngestError("unsupported metadata schema_version")

    source_id = _nonempty_string(raw["id"], "id")
    if not ID_RE.fullmatch(source_id):
        raise IngestError("id must use lowercase kebab-case")
    name = _nonempty_string(raw["name"], "name")
    _nonempty_string(raw["origin"], "origin")

    rights = raw["rights"]
    if not isinstance(rights, dict):
        raise IngestError("rights must be an object")
    _nonempty_string(rights.get("basis"), "rights.basis")
    if rights.get("training_allowed") is not True:
        raise IngestError(f"training_allowed must be true for {source_id}")
    if not isinstance(rights.get("redistribution_allowed"), bool):
        raise IngestError("rights.redistribution_allowed must be boolean")

    retrieval = raw["retrieval"]
    if not isinstance(retrieval, dict):
        raise IngestError("retrieval must be an object")
    expected_hash = _nonempty_string(retrieval.get("sha256"), "retrieval.sha256")
    if not SHA256_RE.fullmatch(expected_hash):
        raise IngestError("retrieval.sha256 must be 64 lowercase hex characters")
    expected_size = retrieval.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise IngestError("retrieval.size_bytes must be a non-negative integer")
    _nonempty_string(retrieval.get("retrieved_at"), "retrieval.retrieved_at")

    content = raw["content"]
    if not isinstance(content, dict):
        raise IngestError("content must be an object")
    languages = _string_list(content.get("languages"), "content.languages")
    domains = _string_list(content.get("domains"), "content.domains")
    formats = _string_list(content.get("formats"), "content.formats")
    unknown_formats = sorted(set(formats) - ALLOWED_FORMATS)
    if unknown_formats:
        raise IngestError(f"unsupported content formats: {', '.join(unknown_formats)}")

    if not isinstance(raw["filtering_steps"], list):
        raise IngestError("filtering_steps must be an array")
    for step in raw["filtering_steps"]:
        _nonempty_string(step, "filtering_steps")

    if not content_path.is_file():
        raise IngestError(f"source file not found: {content_path}")
    actual_size = content_path.stat().st_size
    if actual_size != expected_size:
        raise IngestError(f"size mismatch for {source_id}: expected {expected_size}, got {actual_size}")
    actual_hash = sha256_file(content_path)
    if actual_hash != expected_hash:
        raise IngestError(f"checksum mismatch for {source_id}")

    return Source(
        source_id=source_id,
        name=name,
        path=content_path,
        sha256=expected_hash,
        size_bytes=expected_size,
        languages=languages,
        domains=domains,
        formats=formats,
    )


def load_manifest(manifest_path: Path) -> list[Source]:
    raw = _read_json(manifest_path)
    if not isinstance(raw, list) or not raw:
        raise IngestError("manifest must be a non-empty JSON array")

    sources: list[Source] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise IngestError(f"manifest entry {index} must be an object")
        if set(entry) != {"metadata", "local_path"}:
            raise IngestError(f"manifest entry {index} must contain only metadata and local_path")
        metadata_path = _resolve(manifest_path.parent, entry["metadata"], "metadata")
        content_path = _resolve(manifest_path.parent, entry["local_path"], "local_path")
        source = _load_source(metadata_path, content_path)
        if source.source_id in seen_ids:
            raise IngestError(f"duplicate source id: {source.source_id}")
        seen_ids.add(source.source_id)
        sources.append(source)
    return sorted(sources, key=lambda source: source.source_id)


def iter_documents(source: Source) -> Iterator[dict[str, object]]:
    try:
        with source.path.open("r", encoding="utf-8", newline=None) as handle:
            paragraph: list[str] = []
            index = 0
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.strip():
                    paragraph.append(line)
                    continue
                if paragraph:
                    text = "\n".join(paragraph).strip()
                    if text:
                        yield _document(source, index, text)
                        index += 1
                    paragraph = []
            if paragraph:
                text = "\n".join(paragraph).strip()
                if text:
                    yield _document(source, index, text)
    except UnicodeDecodeError as exc:
        raise IngestError(f"source is not valid UTF-8: {source.source_id}") from exc


def _document(source: Source, index: int, text: str) -> dict[str, object]:
    return {
        "id": f"{source.source_id}:{index:08d}",
        "text": text,
        "source_id": source.source_id,
        "source_name": source.name,
        "source_sha256": source.sha256,
        "languages": list(source.languages),
        "domains": list(source.domains),
        "formats": list(source.formats),
    }


def _json_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def ingest_manifest(manifest_path: Path, output_dir: Path, docs_per_shard: int = 10_000) -> dict[str, object]:
    if docs_per_shard <= 0:
        raise IngestError("docs_per_shard must be greater than zero")
    sources = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("shard-*.jsonl"):
        stale.unlink()
    output_manifest = output_dir / "manifest.json"
    if output_manifest.exists():
        output_manifest.unlink()

    shard_records: list[dict[str, object]] = []
    shard_index = 0
    shard_docs = 0
    total_docs = 0
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
        for source in sources:
            for document in iter_documents(source):
                if handle is None:
                    open_shard()
                if shard_docs == docs_per_shard:
                    close_shard()
                    shard_index += 1
                    open_shard()
                assert handle is not None
                handle.write(_json_line(document))
                shard_docs += 1
                total_docs += 1
    finally:
        close_shard()

    if total_docs == 0:
        raise IngestError("no documents produced")

    result: dict[str, object] = {
        "format_version": "1.0",
        "documents": total_docs,
        "sources": [source.source_id for source in sources],
        "shards": shard_records,
    }
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic training shards from approved local corpora.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--docs-per-shard", type=int, default=10_000)
    args = parser.parse_args()
    try:
        result = ingest_manifest(args.manifest, args.output_dir, args.docs_per_shard)
    except IngestError as exc:
        parser.error(str(exc))
    print(f"ingested {result['documents']} documents into {len(result['shards'])} shard(s)")


if __name__ == "__main__":
    main()
