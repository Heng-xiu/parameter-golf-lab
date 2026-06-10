from __future__ import annotations

import torch

from mini_pgolf.checkpoint import save_checkpoint
from mini_pgolf.config import ExperimentConfig
from mini_pgolf.context import active_seq_len, parse_context_schedule
from mini_pgolf.data import ByteBatcher, read_tokens
from mini_pgolf.evaluate import evaluate_model
from mini_pgolf.metrics import bpb_from_loss, cross_entropy_loss
from mini_pgolf.model import TinyCausalTransformer
from mini_pgolf.utils import ensure_dir, pick_device, set_seed


def train_model(
    cfg: ExperimentConfig, max_steps_override: int | None = None
) -> TinyCausalTransformer:
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    tokens = read_tokens(cfg.train.data_path)
    batcher = ByteBatcher(tokens, cfg.train.batch_size, cfg.train.seed)
    model = TinyCausalTransformer(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    max_steps = max_steps_override or cfg.train.max_steps
    stages = (
        parse_context_schedule(cfg.train.progressive_schedule)
        if cfg.train.progressive_schedule
        else None
    )
    model.train()
    for step in range(1, max_steps + 1):
        progress = step / max_steps
        seq_len = active_seq_len(stages, progress) if stages else cfg.train.seq_len
        seq_len = min(seq_len, cfg.model.max_seq_len)
        x, y = batcher.batch(seq_len, device)
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy_loss(model(x), y)
        loss.backward()
        opt.step()
        if step == 1 or step % cfg.train.log_every == 0 or step == max_steps:
            print(
                f"step={step} seq_len={seq_len} loss={loss.item():.4f} "
                f"bpb={bpb_from_loss(float(loss.item())):.4f}"
            )
    out = ensure_dir(cfg.train.output_dir)
    save_checkpoint(out / "checkpoint.pt", model, max_steps)
    metrics = evaluate_model(
        model, batcher, min(cfg.train.seq_len, cfg.model.max_seq_len), device=device
    )
    print(
        f"saved={out / 'checkpoint.pt'} eval_loss={metrics['loss']:.4f} eval_bpb={metrics['bpb']:.4f}"
    )
    return model
