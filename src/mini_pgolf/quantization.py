from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class QuantizedTensor:
    values: Tensor
    scale: Tensor
    axis: int | None = None
    bits: int = 8


def quantize_symmetric(x: Tensor, bits: int = 8) -> QuantizedTensor:
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().clamp_min(1e-8) / qmax
    values = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return QuantizedTensor(values=values, scale=scale, bits=bits)


def quantize_per_channel(x: Tensor, axis: int = 0, bits: int = 8) -> QuantizedTensor:
    qmax = 2 ** (bits - 1) - 1
    reduce_dims = [dim for dim in range(x.ndim) if dim != axis]
    scale = x.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / qmax
    values = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return QuantizedTensor(values=values, scale=scale, axis=axis, bits=bits)


def dequantize(q: QuantizedTensor) -> Tensor:
    return q.values.float() * q.scale


def quantized_state_dict(
    state: dict[str, Tensor],
    method: str,
) -> tuple[dict[str, Tensor], dict[str, int | str]]:
    out: dict[str, Tensor] = {}
    for name, tensor in state.items():
        if tensor.is_floating_point() and tensor.ndim >= 2:
            q = quantize_per_channel(tensor, bits=4 if method == "int4" else 8)
            out[name] = dequantize(q).to(tensor.dtype)
        else:
            out[name] = tensor
    return out, {"method": method}
