"""Shared Builder1 test helpers."""

from __future__ import annotations


def marketing_text_words(count: int = 50, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(1, count + 1))


def marketing_text_hebrew(count: int = 50) -> str:
    return " ".join(f"מילה{i}" for i in range(1, count + 1))


def marketing_text_hebrew_with_brand(count: int = 50, brand: str = "TestBrand") -> str:
    words: list[str] = []
    brand_at = max(1, count // 2)
    word_num = 1
    for i in range(1, count + 1):
        if i == brand_at:
            words.append(brand)
        else:
            words.append(f"מילה{word_num}")
            word_num += 1
    return " ".join(words)


def marketing_text_english_with_hebrew_brand(count: int = 50, brand: str = "מותג") -> str:
    words: list[str] = []
    brand_at = 5
    word_num = 1
    for i in range(1, count + 1):
        if i == brand_at:
            words.append(brand)
        else:
            words.append(f"word{word_num}")
            word_num += 1
    return " ".join(words)


def marketing_text_with_punctuation() -> str:
    words = [f"word{i}" for i in range(1, 50)]
    words[0] = "Hello,"
    words.append("finished.")
    return " ".join(words)
