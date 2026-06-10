from __future__ import annotations

import torch
from torch import Tensor, nn

from mini_pgolf.attention import CausalSelfAttention
from mini_pgolf.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        out: Tensor = x + self.mlp(self.norm2(x))
        return out


class TinyCausalTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input sequence exceeds model max_seq_len")
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits: Tensor = self.lm_head(self.norm(x))
        return logits
