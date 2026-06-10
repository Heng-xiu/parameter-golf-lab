from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mini_pgolf.quantization import dequantize, quantize_per_channel


def lowrank_residual(weight: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    q_weight = dequantize(quantize_per_channel(weight, axis=0))
    residual = weight - q_weight
    u, s, vh = torch.linalg.svd(residual.float(), full_matrices=False)
    used = min(rank, s.numel())
    a = u[:, :used] * s[:used].sqrt()
    b = s[:used].sqrt().unsqueeze(1) * vh[:used, :]
    return q_weight.to(weight.dtype), a.to(weight.dtype), b.to(weight.dtype)


class LowRankCorrectedLinear(nn.Module):
    def __init__(
        self, base_weight: Tensor, a: Tensor, b: Tensor, bias: Tensor | None = None
    ) -> None:
        super().__init__()
        self.register_buffer("base_weight", base_weight)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.base_weight + self.a @ self.b, self.bias)
