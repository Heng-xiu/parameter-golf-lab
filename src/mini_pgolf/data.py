from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from mini_pgolf.tokenizer import ByteTokenizer


def read_tokens(path: str | Path) -> Tensor:
    text = Path(path).read_text(encoding="utf-8")
    ids = ByteTokenizer().encode(text)
    if len(ids) < 2:
        raise ValueError(f"{path} must contain at least two bytes")
    return torch.tensor(ids, dtype=torch.long)


class ByteBatcher:
    def __init__(self, tokens: Tensor, batch_size: int, seed: int = 0) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D tensor")
        self.tokens = tokens
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def batch(self, seq_len: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        if self.tokens.numel() <= seq_len:
            raise ValueError("corpus is too short for requested seq_len")
        starts = torch.randint(
            0,
            self.tokens.numel() - seq_len - 1,
            (self.batch_size,),
            generator=self.generator,
        )
        rows = [self.tokens[start : start + seq_len + 1] for start in starts]
        packed = torch.stack(rows).to(device)
        return packed[:, :-1], packed[:, 1:]


def split_documents(tokens: Tensor, chunk_len: int, max_documents: int) -> list[Tensor]:
    docs: list[Tensor] = []
    cursor = 0
    min_len = chunk_len + 1
    while cursor + min_len <= tokens.numel() and len(docs) < max_documents:
        docs.append(tokens[cursor : cursor + min_len * 2])
        cursor += min_len * 2
    if not docs and tokens.numel() >= min_len:
        docs.append(tokens[:min_len])
    return docs
