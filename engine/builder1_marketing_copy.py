"""
Builder1 per-ad marketing copy validation (exactly 50 words).
"""
from __future__ import annotations

import re
from typing import Optional

MARKETING_TEXT_WORD_COUNT = 50


def normalize_marketing_text(text: object) -> str:
    """Trim, collapse whitespace, and remove wrapping quotation marks."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def count_marketing_words(text: object) -> int:
    normalized = normalize_marketing_text(text)
    if not normalized:
        return 0
    return len(normalized.split())


def validate_marketing_text_50_words(text: object) -> None:
    count = count_marketing_words(text)
    if count != MARKETING_TEXT_WORD_COUNT:
        raise ValueError(f"marketing_text_word_count_{count}")


def marketing_text_word_count_error(text: object) -> Optional[str]:
    count = count_marketing_words(text)
    if count == MARKETING_TEXT_WORD_COUNT:
        return None
    return f"marketing_text_word_count_{count}"


def trim_marketing_text_to_50_words_if_usable(text: object) -> Optional[str]:
    """
    Final safety trim only when the first 50 words end on a sentence boundary.
    Never pad short text or cut mid-sentence.
    """
    normalized = normalize_marketing_text(text)
    words = normalized.split()
    if len(words) <= MARKETING_TEXT_WORD_COUNT:
        return None
    candidate = " ".join(words[:MARKETING_TEXT_WORD_COUNT]).strip()
    if candidate and candidate[-1] in ".!?":
        return candidate
    trimmed = candidate.rstrip(".,;:")
    if trimmed and trimmed[-1] in ".!?":
        return trimmed
    return None
