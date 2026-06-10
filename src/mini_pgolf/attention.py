from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def apply_rope(x: Tensor) -> Tensor:
    batch, heads, seq, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    positions = torch.arange(seq, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim))
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).reshape(batch, heads, seq, dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        out: Tensor = self.o_proj(y.transpose(1, 2).contiguous().view(batch, seq, self.d_model))
        return out
