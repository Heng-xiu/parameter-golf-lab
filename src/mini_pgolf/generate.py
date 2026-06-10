from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from mini_pgolf.model import TinyCausalTransformer
from mini_pgolf.tokenizer import ByteTokenizer


@torch.no_grad()
def generate_bytes(
    model: TinyCausalTransformer,
    prompt: str,
    max_new_bytes: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device | str = "cpu",
    allowed_token_ids: set[int] | None = None,
) -> str:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [10]
    model.eval()
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_bytes):
        window = tokens[:, -model.config.max_seq_len :]
        logits: Tensor = model(window)[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        if allowed_token_ids is not None:
            mask = torch.full_like(logits, float("-inf"))
            allowed = torch.tensor(
                sorted(allowed_token_ids), dtype=torch.long, device=logits.device
            )
            mask[:, allowed] = logits[:, allowed]
            logits = mask
        if top_k > 0:
            values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            cutoff = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens.squeeze(0).tolist())
