from __future__ import annotations

import torch

from mini_pgolf.data import ByteBatcher
from mini_pgolf.metrics import bpb_from_loss, cross_entropy_loss
from mini_pgolf.model import TinyCausalTransformer


@torch.no_grad()
def evaluate_model(
    model: TinyCausalTransformer,
    batcher: ByteBatcher,
    seq_len: int,
    batches: int = 5,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = batcher.batch(seq_len, device)
        loss = cross_entropy_loss(model(x), y)
        losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "bpb": bpb_from_loss(mean_loss)}
