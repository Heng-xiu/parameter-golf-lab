from __future__ import annotations

import argparse

from mini_pgolf.checkpoint import load_checkpoint
from mini_pgolf.config import load_ttt_config
from mini_pgolf.data import read_tokens
from mini_pgolf.metrics import bpb_from_loss
from mini_pgolf.ttt import run_score_first_ttt
from mini_pgolf.utils import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/ttt_lora.yaml")
    args = parser.parse_args()
    cfg = load_ttt_config(args.config)
    model = load_checkpoint(args.checkpoint, pick_device(cfg.device))
    result = run_score_first_ttt(model, read_tokens(cfg.data_path), cfg)
    print(
        f"no_ttt_loss={result.no_ttt_loss:.4f} no_ttt_bpb={bpb_from_loss(result.no_ttt_loss):.4f} "
        f"ttt_loss={result.ttt_loss:.4f} ttt_bpb={bpb_from_loss(result.ttt_loss):.4f} "
        f"chunks_scored_before_update={result.chunks_scored_before_update}"
    )


if __name__ == "__main__":
    main()
