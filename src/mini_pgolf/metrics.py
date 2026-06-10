from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def bpb_from_loss(loss_nats: float) -> float:
    return loss_nats / math.log(2.0)


def perplexity_from_loss(loss_nats: float) -> float:
    return math.exp(min(loss_nats, 20.0))


@torch.no_grad()
def accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().item())
