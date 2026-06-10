from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            if base.bias is not None
            else None
        )
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * (self.alpha / self.rank)
        return base + delta

    def reset_lora(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)


def _target_names(mask: str) -> set[str]:
    if mask == "all":
        return {"q_proj", "k_proj", "v_proj", "o_proj", "0", "2"}
    if mask == "qv_only":
        return {"q_proj", "v_proj"}
    if mask == "no_qv":
        return {"k_proj", "o_proj", "0", "2"}
    if mask == "ko_mlp":
        return {"k_proj", "o_proj", "0", "2"}
    raise ValueError(f"unknown LoRA target mask: {mask}")


def inject_lora(module: nn.Module, rank: int, alpha: float, target_mask: str) -> int:
    targets = _target_names(target_mask)
    inserted = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in targets:
            setattr(module, child_name, LoRALinear(child, rank, alpha))
            inserted += 1
        else:
            inserted += inject_lora(child, rank, alpha, target_mask)
    return inserted


def freeze_non_lora(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name


def reset_lora(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.reset_lora()


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for name, param in module.named_parameters() if "lora_" in name]
