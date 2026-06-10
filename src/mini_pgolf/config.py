from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 256
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 4
    mlp_ratio: int = 4
    max_seq_len: int = 128
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclass(slots=True)
class TrainConfig:
    data_path: str = "data/tiny.txt"
    output_dir: str = "outputs/baseline"
    batch_size: int = 4
    seq_len: int = 64
    max_steps: int = 100
    lr: float = 3e-3
    seed: int = 7
    device: str = "auto"
    log_every: int = 10
    progressive_schedule: str | None = None


@dataclass(slots=True)
class TTTConfig:
    data_path: str = "data/tiny.txt"
    chunk_len: int = 64
    lr: float = 1e-2
    steps_per_chunk: int = 1
    lora_rank: int = 4
    lora_alpha: float = 8.0
    target_mask: str = "no_qv"
    max_documents: int = 4
    seed: int = 7
    device: str = "auto"


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    train: TrainConfig


def _coerce_model_config(raw: dict[str, Any]) -> ModelConfig:
    allowed = {item.name for item in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in raw.items() if key in allowed})


def _coerce_train_config(raw: dict[str, Any]) -> TrainConfig:
    allowed = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in raw.items() if key in allowed})


def _coerce_ttt_config(raw: dict[str, Any]) -> TTTConfig:
    allowed = {item.name for item in fields(TTTConfig)}
    return TTTConfig(**{key: value for key, value in raw.items() if key in allowed})


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text()) or {}
    return cast(dict[str, Any], loaded)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _load_yaml_mapping(path)
    return ExperimentConfig(
        model=_coerce_model_config(cast(dict[str, Any], raw.get("model", {}))),
        train=_coerce_train_config(cast(dict[str, Any], raw.get("train", {}))),
    )


def load_ttt_config(path: str | Path) -> TTTConfig:
    raw = _load_yaml_mapping(path)
    return _coerce_ttt_config(cast(dict[str, Any], raw.get("ttt", raw)))
