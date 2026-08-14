from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from genesis_ai.filtering import iter_input_documents
from genesis_ai.ingest import IngestError, sha256_file

PIECE_RE = re.compile(r"\s+|\S+")
BASE_VOCAB_SIZE = 256


def _pieces(text: str) -> Iterable[bytes]:
    for match in PIECE_RE.finditer(text):
        yield match.group(0).encode("utf-8")


def _merge_pair(tokens: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    if len(tokens) < 2:
        return tokens
    output: list[int] = []
    index = 0
    left, right = pair
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
            output.append(new_id)
            index += 2
        else:
            output.append(tokens[index])
            index += 1
    return tuple(output)


@dataclass(frozen=True)
class ByteBPETokenizer:
    merges: tuple[tuple[int, int, int], ...]

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for piece in _pieces(text):
            tokens = tuple(piece)
            for left, right, new_id in self.merges:
                tokens = _merge_pair(tokens, (left, right), new_id)
            encoded.extend(tokens)
        return encoded

    def token_bytes(self) -> list[bytes]:
        vocab = [bytes([value]) for value in range(BASE_VOCAB_SIZE)]
        for left, right, new_id in self.merges:
            if new_id != len(vocab) or left >= len(vocab) or right >= len(vocab):
                raise IngestError("invalid tokenizer merge table")
            vocab.append(vocab[left] + vocab[right])
        return vocab

    def decode(self, token_ids: Iterable[int]) -> str:
        vocab = self.token_bytes()
        chunks: list[bytes] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 or token_id >= len(vocab):
                raise IngestError(f"invalid token id: {token_id}")
            chunks.append(vocab[token_id])
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestError("token sequence does not decode to valid UTF-8") from exc

    def to_dict(self, training: dict[str, object] | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "format_version": "1.0",
            "kind": "byte-bpe",
            "base_vocab_size": BASE_VOCAB_SIZE,
            "vocab_size": self.vocab_size,
            "merges": [list(merge) for merge in self.merges],
        }
        if training is not None:
            result["training"] = training
        return result

    @classmethod
    def from_dict(cls, raw: object) -> "ByteBPETokenizer":
        if not isinstance(raw, dict) or raw.get("format_version") != "1.0" or raw.get("kind") != "byte-bpe":
            raise IngestError("unsupported tokenizer artifact")
        merges_raw = raw.get("merges")
        if not isinstance(merges_raw, list):
            raise IngestError("tokenizer merges must be an array")
        merges: list[tuple[int, int, int]] = []
        next_id = BASE_VOCAB_SIZE
        for merge in merges_raw:
            if (
                not isinstance(merge, list)
                or len(merge) != 3
                or any(not isinstance(value, int) or isinstance(value, bool) for value in merge)
            ):
                raise IngestError("invalid tokenizer merge")
            left, right, new_id = merge
            if new_id != next_id or left < 0 or right < 0 or left >= new_id or right >= new_id:
                raise IngestError("invalid tokenizer merge ordering")
            merges.append((left, right, new_id))
            next_id += 1
        return cls(tuple(merges))

    @classmethod
    def load(cls, path: Path) -> "ByteBPETokenizer":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise IngestError(f"cannot read tokenizer: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise IngestError(f"invalid tokenizer JSON: {exc.msg}") from exc
        return cls.from_dict(raw)


def _count_pieces(input_dir: Path) -> tuple[Counter[tuple[int, ...]], int, int]:
    counts: Counter[tuple[int, ...]] = Counter()
    documents = 0
    utf8_bytes = 0
    for document in iter_input_documents(input_dir):
        text = document["text"]
        documents += 1
        utf8_bytes += len(text.encode("utf-8"))
        for piece in _pieces(text):
            counts[tuple(piece)] += 1
    if not counts:
        raise IngestError("tokenizer input contains no text")
    return counts, documents, utf8_bytes


def train_byte_bpe(
    input_dir: Path,
    *,
    vocab_size: int = 1024,
    min_pair_count: int = 2,
) -> tuple[ByteBPETokenizer, dict[str, object]]:
    if vocab_size < BASE_VOCAB_SIZE:
        raise IngestError(f"vocab_size must be at least {BASE_VOCAB_SIZE}")
    if min_pair_count < 1:
        raise IngestError("min_pair_count must be at least 1")

    piece_counts, documents, utf8_bytes = _count_pieces(input_dir)
    merges: list[tuple[int, int, int]] = []

    while BASE_VOCAB_SIZE + len(merges) < vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for tokens, frequency in piece_counts.items():
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += frequency
        if not pair_counts:
            break
        best_count = max(pair_counts.values())
        if best_count < min_pair_count:
            break
        best_pair = min(pair for pair, count in pair_counts.items() if count == best_count)
        new_id = BASE_VOCAB_SIZE + len(merges)
        merged_counts: Counter[tuple[int, ...]] = Counter()
        for tokens, frequency in piece_counts.items():
            merged_counts[_merge_pair(tokens, best_pair, new_id)] += frequency
        piece_counts = merged_counts
        merges.append((best_pair[0], best_pair[1], new_id))

    tokenizer = ByteBPETokenizer(tuple(merges))
    token_count = 0
    round_trip_failures = 0
    for document in iter_input_documents(input_dir):
        text = document["text"]
        encoded = tokenizer.encode(text)
        token_count += len(encoded)
        if tokenizer.decode(encoded) != text:
            round_trip_failures += 1
    if round_trip_failures:
        raise IngestError(f"round-trip verification failed for {round_trip_failures} document(s)")

    input_manifest = input_dir / "manifest.json"
    metrics: dict[str, object] = {
        "documents": documents,
        "utf8_bytes": utf8_bytes,
        "tokens": token_count,
        "bytes_per_token": round(utf8_bytes / token_count, 6) if token_count else 0.0,
        "token_reduction_vs_bytes": round(1.0 - token_count / utf8_bytes, 6) if utf8_bytes else 0.0,
        "requested_vocab_size": vocab_size,
        "vocab_size": tokenizer.vocab_size,
        "merge_count": len(merges),
        "min_pair_count": min_pair_count,
        "round_trip_failures": round_trip_failures,
        "input_manifest_sha256": sha256_file(input_manifest),
    }
    filter_metrics = input_dir / "metrics.json"
    if filter_metrics.is_file():
        metrics["filter_metrics_sha256"] = sha256_file(filter_metrics)
    return tokenizer, metrics


def save_tokenizer(tokenizer: ByteBPETokenizer, path: Path, training: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tokenizer.to_dict(training), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Genesis AI deterministic byte-level BPE tokenizer.")
    parser.add_argument("input_dir", type=Path, help="Filtered corpus directory")
    parser.add_argument("output", type=Path, help="Tokenizer JSON artifact")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-pair-count", type=int, default=2)
    args = parser.parse_args()
    try:
        tokenizer, metrics = train_byte_bpe(
            args.input_dir,
            vocab_size=args.vocab_size,
            min_pair_count=args.min_pair_count,
        )
        save_tokenizer(tokenizer, args.output, metrics)
    except IngestError as exc:
        parser.error(str(exc))
    print(
        f"trained vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} "
        f"bytes/token={metrics['bytes_per_token']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
