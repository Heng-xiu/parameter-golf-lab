from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mini_pgolf.config import ModelConfig
from mini_pgolf.model import TinyCausalTransformer


def save_checkpoint(path: str | Path, model: TinyCausalTransformer, step: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": asdict(model.config), "model": model.state_dict(), "step": step}, path
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> TinyCausalTransformer:
    raw: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    model = TinyCausalTransformer(ModelConfig(**raw["model_config"]))
    state = raw["model"] if "model" in raw else raw["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device)
