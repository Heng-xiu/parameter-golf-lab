from __future__ import annotations

import argparse

from mini_pgolf.checkpoint import load_checkpoint
from mini_pgolf.data import ByteBatcher, read_tokens
from mini_pgolf.evaluate import evaluate_model
from mini_pgolf.utils import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/tiny.txt")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    batcher = ByteBatcher(read_tokens(args.data), batch_size=4, seed=11)
    metrics = evaluate_model(
        model, batcher, min(args.seq_len, model.config.max_seq_len), device=device
    )
    print(f"cross_entropy={metrics['loss']:.4f} bpb={metrics['bpb']:.4f}")


if __name__ == "__main__":
    main()
