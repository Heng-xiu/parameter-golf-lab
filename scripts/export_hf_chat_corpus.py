from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def normalize_role(role: str) -> str:
    lowered = role.lower()
    if lowered in {"assistant", "ai", "gpt"}:
        return "Assistant"
    if lowered in {"user", "human"}:
        return "User"
    return role.title()


def render_messages(messages: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = normalize_role(str(message.get("role", "")))
        content = str(message.get("content", "")).strip()
        if role and content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\n\n" if parts else ""


def row_to_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        return render_messages(messages)
    completion = row.get("completion")
    if isinstance(completion, str) and completion.strip():
        return completion.replace("\nAI:", "\nAssistant:").strip() + "\n\n"
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip() + "\n\n"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceTB/everyday-conversations-llama3.1-2k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--out", default="data/hf_chat.txt")
    parser.add_argument("--max-examples", type=int, default=1200)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency. Run: .venv/bin/python -m pip install -r requirements-hf.txt"
        ) from exc

    dataset = load_dataset(args.dataset, split=args.split)
    rendered: list[str] = []
    for index, row in enumerate(dataset):
        if index >= args.max_examples:
            break
        text = row_to_text(dict(row))
        if text:
            rendered.append(text)
    if not rendered:
        raise SystemExit("No usable chat text was exported; inspect dataset columns and format.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rendered), encoding="utf-8")
    print(f"wrote {out} examples={len(rendered)} bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
