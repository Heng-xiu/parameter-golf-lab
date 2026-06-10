from __future__ import annotations

from pathlib import Path

from torch import nn


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def size_report(model: nn.Module, checkpoint: str | Path | None = None) -> dict[str, int]:
    params = parameter_count(model)
    report = {
        "parameters": params,
        "fp32_bytes": params * 4,
        "fp16_bytes": params * 2,
    }
    if checkpoint is not None:
        report["checkpoint_bytes"] = checkpoint_size(checkpoint)
    return report
