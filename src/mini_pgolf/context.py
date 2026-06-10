from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextStage:
    seq_len: int
    until_progress: float


def parse_context_schedule(text: str) -> list[ContextStage]:
    stages: list[ContextStage] = []
    for part in text.split(","):
        length_text, progress_text = part.strip().split("@", maxsplit=1)
        stages.append(ContextStage(int(length_text), float(progress_text)))
    if not stages:
        raise ValueError("schedule must not be empty")
    previous = 0.0
    for stage in stages:
        if stage.seq_len <= 0:
            raise ValueError("sequence lengths must be positive")
        if stage.until_progress <= previous:
            raise ValueError("progress boundaries must increase")
        previous = stage.until_progress
    return stages


def active_seq_len(stages: list[ContextStage], progress: float) -> int:
    for stage in stages:
        if progress < stage.until_progress or stage.until_progress >= 1.0:
            return stage.seq_len
    return stages[-1].seq_len
