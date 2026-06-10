from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tiny.txt")
    args = parser.parse_args()
    text = (
        "mini parameter golf teaches tiny language models.\n"
        "a byte model predicts the next byte from previous bytes.\n"
        "context grows from short examples to longer examples.\n"
        "quantization trades precision for artifact size.\n"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text * 80, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
