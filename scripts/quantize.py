from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from mini_pgolf.checkpoint import load_checkpoint
from mini_pgolf.data import ByteBatcher, read_tokens
from mini_pgolf.evaluate import evaluate_model
from mini_pgolf.lowrank import lowrank_residual
from mini_pgolf.quantization import dequantize, quantize_per_channel, quantized_state_dict
from mini_pgolf.size_report import checkpoint_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--method", choices=["int8", "int8-channel", "int4", "int8-lowrank"], default="int8"
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--data", default="data/tiny.txt")
    args = parser.parse_args()
    raw: dict[str, Any] = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_checkpoint(args.checkpoint)
    before = checkpoint_size(args.checkpoint)
    before_metrics = None
    if Path(args.data).exists():
        batcher = ByteBatcher(read_tokens(args.data), batch_size=4, seed=22)
        before_metrics = evaluate_model(model, batcher, min(64, model.config.max_seq_len))
    if args.method == "int8-lowrank":
        state: dict[str, torch.Tensor] = {}
        lowrank_layers = 0
        for name, tensor in raw["model"].items():
            if tensor.is_floating_point() and tensor.ndim == 2:
                q_weight, a, b = lowrank_residual(tensor, args.rank)
                int8_err = torch.linalg.matrix_norm(
                    tensor - dequantize(quantize_per_channel(tensor))
                ).item()
                corrected_err = torch.linalg.matrix_norm(tensor - (q_weight + a @ b)).item()
                state[name] = q_weight + a @ b
                raw[f"{name}.lowrank_a"] = a
                raw[f"{name}.lowrank_b"] = b
                lowrank_layers += 1
            else:
                state[name] = tensor
        raw["model"] = state
        raw["quantization"] = {
            "method": args.method,
            "rank": args.rank,
            "lowrank_layers": lowrank_layers,
        }
        print(f"example_reconstruction_error int8={int8_err:.6f} corrected={corrected_err:.6f}")
    else:
        raw["model"], raw["quantization"] = quantized_state_dict(raw["model"], args.method)
    out = Path(args.checkpoint).with_suffix(f".{args.method}.pt")
    torch.save(raw, out)
    after = checkpoint_size(out)
    print(f"wrote={out} original_bytes={before} quantized_artifact_bytes={after}")
    print("estimated_quantized_weight_storage is smaller than this teaching checkpoint artifact")
    if before_metrics is not None:
        q_model = load_checkpoint(out)
        after_metrics = evaluate_model(
            q_model, ByteBatcher(read_tokens(args.data), 4, 22), min(64, q_model.config.max_seq_len)
        )
        print(
            f"loss_before={before_metrics['loss']:.4f} loss_after={after_metrics['loss']:.4f} "
            f"loss_delta={after_metrics['loss'] - before_metrics['loss']:.4f}"
        )


if __name__ == "__main__":
    main()
