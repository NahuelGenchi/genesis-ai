from pathlib import Path

import torch
from torch.utils.data import Dataset


class ByteDataset(Dataset):
    """Simple bootstrap dataset over UTF-8/raw bytes.

    This is intentionally basic. It validates the training pipeline before the
    project's provenance-aware tokenizer and shard pipeline are implemented.
    """

    def __init__(self, root: str | Path, context_length: int) -> None:
        root = Path(root)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(root)
        if not files:
            raise ValueError(f"no data files found under {root}")
        payload = b"\n".join(path.read_bytes() for path in files)
        if len(payload) <= context_length:
            raise ValueError("dataset must contain more bytes than context_length")
        self.tokens = torch.tensor(list(payload), dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.context_length + 1]
        return chunk[:-1], chunk[1:]
