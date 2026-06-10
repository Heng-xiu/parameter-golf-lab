from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_pgolf.config import TTTConfig
from mini_pgolf.data import split_documents
from mini_pgolf.lora import freeze_non_lora, inject_lora, lora_parameters, reset_lora
from mini_pgolf.metrics import bpb_from_loss, cross_entropy_loss
from mini_pgolf.model import TinyCausalTransformer
from mini_pgolf.utils import pick_device, set_seed


@dataclass(slots=True)
class TTTResult:
    no_ttt_loss: float
    ttt_loss: float
    chunks_scored_before_update: int


def _chunk_pairs(doc: torch.Tensor, chunk_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, doc.numel() - chunk_len, chunk_len):
        piece = doc[start : start + chunk_len + 1]
        if piece.numel() == chunk_len + 1:
            pairs.append((piece[:-1], piece[1:]))
    return pairs


def run_score_first_ttt(
    model: TinyCausalTransformer,
    tokens: torch.Tensor,
    cfg: TTTConfig,
) -> TTTResult:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = model.to(device)
    inserted = inject_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.target_mask)
    if inserted == 0:
        raise ValueError("no LoRA modules were inserted")
    freeze_non_lora(model)
    docs = split_documents(tokens, cfg.chunk_len, cfg.max_documents)
    no_ttt_losses: list[float] = []
    ttt_losses: list[float] = []
    scored_chunks = 0
    for doc_idx, doc in enumerate(docs):
        reset_lora(model)
        opt = torch.optim.AdamW(lora_parameters(model), lr=cfg.lr)
        pairs = _chunk_pairs(doc.to(device), cfg.chunk_len)
        for x_cpu, y_cpu in pairs:
            x = x_cpu.unsqueeze(0)
            y = y_cpu.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                no_ttt_losses.append(float(cross_entropy_loss(model(x), y).item()))
            model.train()
            logits = model(x)
            score_loss = cross_entropy_loss(logits, y)
            ttt_losses.append(float(score_loss.item()))
            scored_chunks += 1
            for _ in range(cfg.steps_per_chunk):
                opt.zero_grad(set_to_none=True)
                update_loss = cross_entropy_loss(model(x), y)
                update_loss.backward()
                opt.step()
            print(
                f"doc={doc_idx} chunk={scored_chunks} scored_before_update=true "
                f"loss={score_loss.item():.4f} bpb={bpb_from_loss(float(score_loss.item())):.4f}"
            )
    no_ttt = sum(no_ttt_losses) / len(no_ttt_losses)
    ttt = sum(ttt_losses) / len(ttt_losses)
    return TTTResult(no_ttt_loss=no_ttt, ttt_loss=ttt, chunks_scored_before_update=scored_chunks)
