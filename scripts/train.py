from __future__ import annotations

import argparse

from mini_pgolf.config import load_experiment_config
from mini_pgolf.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="Override config device, e.g. auto, mps, cpu"
    )
    parser.add_argument("--output-dir", default=None, help="Override config output directory")
    parser.add_argument("--data-path", default=None, help="Override config data path")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    if args.device is not None:
        cfg.train.device = args.device
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.data_path is not None:
        cfg.train.data_path = args.data_path
    train_model(cfg, args.max_steps)


if __name__ == "__main__":
    main()
