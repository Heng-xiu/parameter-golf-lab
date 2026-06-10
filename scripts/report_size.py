from __future__ import annotations

import argparse

from mini_pgolf.checkpoint import load_checkpoint
from mini_pgolf.size_report import size_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    report = size_report(model, args.checkpoint)
    for key, value in report.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
