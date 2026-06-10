from __future__ import annotations

import argparse
from pathlib import Path

EXAMPLES = [
    (
        "Hi there",
        "Hello! I am a tiny teaching model. I can answer simple questions from my training text.",
    ),
    (
        "What is mini parameter golf?",
        "Mini parameter golf is a small project for learning how compact language models are trained and measured.",
    ),
    (
        "What is BPB?",
        "BPB means bits per byte. It measures language-model compression quality without depending on a tokenizer.",
    ),
    (
        "Why use MPS?",
        "MPS lets Apple Silicon Macs train PyTorch models faster than CPU for many small experiments.",
    ),
    (
        "Why is my model bad at chat?",
        "A model learns the style of its training data. Web text teaches continuation, while dialogue data teaches replies.",
    ),
    (
        "How do I improve chat quality?",
        "Use more dialogue examples, train a larger model for longer, and format prompts consistently.",
    ),
    (
        "What is quantization?",
        "Quantization stores weights with fewer bits. It can reduce size, but it may add prediction error.",
    ),
    (
        "What is LoRA?",
        "LoRA adds small trainable low-rank adapters while keeping the base model frozen.",
    ),
    (
        "What is score first TTT?",
        "Score-first TTT measures a chunk before adapting on it, so the metric does not use future information.",
    ),
    (
        "Give me a short summary.",
        "This repo teaches small language models, context growth, quantization, low-rank correction, and LoRA TTT.",
    ),
]


def render_example(user: str, assistant: str) -> str:
    return f"User: {user}\nAssistant: {assistant}\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tiny_chat.txt")
    parser.add_argument("--repeat", type=int, default=250)
    args = parser.parse_args()
    text = "".join(render_example(user, assistant) for user, assistant in EXAMPLES)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text * args.repeat, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
