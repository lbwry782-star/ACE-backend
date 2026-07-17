"""Shared Builder1 test helpers."""

from __future__ import annotations


def marketing_text_words(count: int = 50, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(1, count + 1))


def marketing_text_hebrew(count: int = 50) -> str:
    return " ".join(f"מילה{i}" for i in range(1, count + 1))


def marketing_text_with_punctuation() -> str:
    words = [f"word{i}" for i in range(1, 50)]
    words[0] = "Hello,"
    words.append("finished.")
    return " ".join(words)
