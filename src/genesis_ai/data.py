from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .filtering import iter_input_documents
from .tokenizer import ByteBPETokenizer


class TokenDataset(Dataset):
    """Deterministic token dataset over integrity-checked filtered shards."""

    def __init__(
        self,
        root: str | Path,
        tokenizer: ByteBPETokenizer,
        context_length: int,
        *,
        split: str = "train",
        validation_fraction: float = 0.1,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")

        root = Path(root)
        threshold = int(validation_fraction * 10_000)
        separator = tokenizer.encode("\n\n")
        token_ids: list[int] = []
        document_ids: list[str] = []

        for document in iter_input_documents(root):
            document_id = document.get("id")
            text = document.get("text")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("filtered document must have a non-empty id")
            if not isinstance(text, str):
                raise ValueError(f"document {document_id} must have text")
            bucket = int.from_bytes(hashlib.sha256(document_id.encode("utf-8")).digest()[:4], "big") % 10_000
            is_validation = bucket < threshold
            if split == "train" and is_validation:
                continue
            if split == "validation" and not is_validation:
                continue
            encoded = tokenizer.encode(text)
            if not encoded:
                continue
            if token_ids:
                token_ids.extend(separator)
            token_ids.extend(encoded)
            document_ids.append(document_id)

        if not document_ids:
            raise ValueError(f"no documents selected for split={split}")
        if len(token_ids) <= context_length:
            raise ValueError("dataset must contain more tokens than context_length")

        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.document_ids = tuple(document_ids)
        self.split = split

    @property
    def token_count(self) -> int:
        return int(self.tokens.numel())

    @property
    def document_count(self) -> int:
        return len(self.document_ids)

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk = self.tokens[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]


def sample_batch(
    dataset: TokenDataset,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    starts = torch.randint(0, len(dataset), (batch_size,), generator=generator)
    pairs = [dataset[int(index)] for index in starts]
    x = torch.stack([pair[0] for pair in pairs])
    y = torch.stack([pair[1] for pair in pairs])
    return x, y
