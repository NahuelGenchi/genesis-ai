from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from genesis_ai.filtering import filter_corpus
from genesis_ai.ingest import IngestError, ingest_manifest, sha256_file
from genesis_ai.tokenizer import save_tokenizer, train_byte_bpe

START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK .*\*\*\*$", re.IGNORECASE | re.MULTILINE)
END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK .*\*\*\*$", re.IGNORECASE | re.MULTILINE)
USER_AGENT = "genesis-ai/0.0.1 tokenizer-bootstrap"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_text(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise IngestError(f"download failed for {url}: {exc}") from exc
    if not payload:
        raise IngestError(f"empty download: {url}")
    return payload


def strip_gutenberg_wrapper(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start = START_RE.search(normalized)
    if start is None:
        raise IngestError("Project Gutenberg START marker not found")
    end = END_RE.search(normalized, start.end())
    if end is None:
        raise IngestError("Project Gutenberg END marker not found")
    body = normalized[start.end() : end.start()].strip()
    if not body:
        raise IngestError("Project Gutenberg body is empty")
    return body


def systematic_paragraph_sample(text: str, max_utf8_bytes: int) -> tuple[str, int]:
    if max_utf8_bytes <= 0:
        raise IngestError("sample byte limit must be positive")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        raise IngestError("source has no paragraphs")
    sizes = [len(part.encode("utf-8")) + 2 for part in paragraphs]
    total = sum(sizes)
    if total <= max_utf8_bytes:
        return "\n\n".join(paragraphs) + "\n", 1

    stride = max(1, math.ceil(total / max_utf8_bytes))
    selected: list[str] = []
    used = 0
    for index in range(0, len(paragraphs), stride):
        paragraph = paragraphs[index]
        size = len(paragraph.encode("utf-8")) + (2 if selected else 1)
        if used + size > max_utf8_bytes:
            continue
        selected.append(paragraph)
        used += size
    if not selected:
        raise IngestError("sampling produced no text")
    return "\n\n".join(selected) + "\n", stride


def _load_catalog(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestError(f"invalid bootstrap source catalog: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format_version") != "1.0":
        raise IngestError("unsupported bootstrap source catalog")
    if not isinstance(raw.get("sample_bytes_per_source"), int) or raw["sample_bytes_per_source"] <= 0:
        raise IngestError("sample_bytes_per_source must be positive")
    if not isinstance(raw.get("sources"), list) or not raw["sources"]:
        raise IngestError("bootstrap source catalog must contain sources")
    return raw


def _existing_lock(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestError(f"invalid source lock: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise IngestError("invalid source lock structure")
    result: dict[str, dict] = {}
    for source in raw["sources"]:
        if isinstance(source, dict) and isinstance(source.get("id"), str):
            result[source["id"]] = source
    return result


def bootstrap_tokenizer(
    catalog_path: Path,
    workspace: Path,
    output_path: Path,
    lock_path: Path,
    *,
    vocab_size: int = 512,
) -> dict[str, object]:
    catalog = _load_catalog(catalog_path)
    existing_lock = _existing_lock(lock_path)
    sample_limit = catalog["sample_bytes_per_source"]
    raw_dir = workspace / "raw"
    metadata_dir = workspace / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, str]] = []
    locked_sources: list[dict[str, object]] = []
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    for raw_source in catalog["sources"]:
        if not isinstance(raw_source, dict):
            raise IngestError("bootstrap source entry must be an object")
        required = {"id", "name", "author", "ebook_id", "language", "text_url", "rights_page"}
        missing = sorted(required - raw_source.keys())
        if missing:
            raise IngestError(f"bootstrap source missing fields: {', '.join(missing)}")
        source_id = str(raw_source["id"])
        payload = download_text(str(raw_source["text_url"]))
        upstream_sha = _sha256_bytes(payload)
        upstream_size = len(payload)

        old = existing_lock.get(source_id)
        if old:
            if old.get("upstream_sha256") != upstream_sha or old.get("upstream_size_bytes") != upstream_size:
                raise IngestError(f"upstream source changed since lock: {source_id}")

        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestError(f"bootstrap source is not UTF-8: {source_id}") from exc
        body = strip_gutenberg_wrapper(text)
        sampled, stride = systematic_paragraph_sample(body, sample_limit)
        sampled_bytes = sampled.encode("utf-8")
        sample_sha = _sha256_bytes(sampled_bytes)

        if old and old.get("sample_sha256") != sample_sha:
            raise IngestError(f"sample changed since lock: {source_id}")

        content_path = raw_dir / f"{source_id}.txt"
        content_path.write_bytes(sampled_bytes)
        metadata_path = metadata_dir / f"{source_id}.json"
        metadata = {
            "schema_version": "1.0",
            "id": source_id,
            "name": str(raw_source["name"]),
            "origin": str(raw_source["text_url"]),
            "rights": {
                "basis": "Project Gutenberg eBook page marks this edition public domain in the USA; raw redistribution is disabled and jurisdiction must be reviewed before release.",
                "evidence": [str(raw_source["rights_page"])],
                "training_allowed": True,
                "redistribution_allowed": False,
            },
            "retrieval": {
                "retrieved_at": retrieved_at,
                "sha256": sample_sha,
                "size_bytes": len(sampled_bytes),
            },
            "content": {
                "languages": [str(raw_source["language"])],
                "domains": ["literature"],
                "formats": ["text"],
            },
            "filtering_steps": [
                "Project Gutenberg header and license footer removed using canonical markers.",
                f"Systematic paragraph sample with stride {stride}; UTF-8 cap {sample_limit} bytes.",
            ],
            "notes": f"Bootstrap tokenizer source; author: {raw_source['author']}; Project Gutenberg eBook #{raw_source['ebook_id']}.",
        }
        _write_json(metadata_path, metadata)
        manifest_entries.append({
            "metadata": str(metadata_path.relative_to(workspace)),
            "local_path": str(content_path.relative_to(workspace)),
        })
        locked_sources.append({
            "id": source_id,
            "ebook_id": raw_source["ebook_id"],
            "language": raw_source["language"],
            "text_url": raw_source["text_url"],
            "rights_page": raw_source["rights_page"],
            "upstream_sha256": upstream_sha,
            "upstream_size_bytes": upstream_size,
            "sample_sha256": sample_sha,
            "sample_size_bytes": len(sampled_bytes),
            "sample_stride": stride,
        })

    manifest_path = workspace / "manifest.json"
    _write_json(manifest_path, manifest_entries)
    lock = {
        "format_version": "1.0",
        "catalog_sha256": sha256_file(catalog_path),
        "sample_bytes_per_source": sample_limit,
        "sources": sorted(locked_sources, key=lambda item: str(item["id"])),
    }
    _write_json(lock_path, lock)

    ingested_dir = workspace / "ingested"
    filtered_dir = workspace / "filtered"
    ingest_manifest(manifest_path, ingested_dir, docs_per_shard=5000)
    filter_corpus(ingested_dir, filtered_dir, min_chars=80, docs_per_shard=5000)
    tokenizer, metrics = train_byte_bpe(filtered_dir, vocab_size=vocab_size, min_pair_count=2)
    metrics["source_lock_sha256"] = sha256_file(lock_path)
    metrics["source_catalog_sha256"] = sha256_file(catalog_path)
    metrics["corpus_source_count"] = len(locked_sources)
    save_tokenizer(tokenizer, output_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch approved public-domain bootstrap text and train genesis-v0 tokenizer.")
    parser.add_argument("--catalog", type=Path, default=Path("data/bootstrap-tokenizer-sources.json"))
    parser.add_argument("--workspace", type=Path, default=Path("runs/tokenizer-v0"))
    parser.add_argument("--output", type=Path, default=Path("tokenizers/genesis-v0.json"))
    parser.add_argument("--lock", type=Path, default=Path("data/bootstrap-tokenizer-lock.json"))
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()
    try:
        metrics = bootstrap_tokenizer(args.catalog, args.workspace, args.output, args.lock, vocab_size=args.vocab_size)
    except IngestError as exc:
        parser.error(str(exc))
    print(
        f"genesis-v0 trained: vocab={metrics['vocab_size']} bytes/token={metrics['bytes_per_token']} "
        f"documents={metrics['documents']}"
    )


if __name__ == "__main__":
    main()
