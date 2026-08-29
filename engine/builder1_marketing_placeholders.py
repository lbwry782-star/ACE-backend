"""
Builder1 marketing text placeholder hygiene — deterministic before paid repair.
"""
from __future__ import annotations

import re
from typing import Optional

from engine.builder1_marketing_copy import MARKETING_TEXT_WORD_COUNT, count_marketing_words

_PLACEHOLDER_RESIDUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\(\s*המוצר\s+הזה\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*this\s+product\s*\)", re.IGNORECASE),
    re.compile(r"\[\s*product\s*\]", re.IGNORECASE),
    re.compile(r"\{\s*product\s*\}", re.IGNORECASE),
    re.compile(r"\(\s*product\s*\)", re.IGNORECASE),
    re.compile(r"שם\s+המוצר"),
)


def has_builder1_marketing_placeholder_residue(text: object) -> bool:
    token = str(text or "").strip()
    if not token:
        return False
    return any(pattern.search(token) for pattern in _PLACEHOLDER_RESIDUE_PATTERNS)


def sanitize_builder1_marketing_placeholder_residue(text: object) -> str:
    s = " ".join(str(text or "").split()).strip()
    for pattern in _PLACEHOLDER_RESIDUE_PATTERNS:
        s = pattern.sub("", s)
    return " ".join(s.split()).strip()


def marketing_placeholder_error(text: object) -> Optional[str]:
    if has_builder1_marketing_placeholder_residue(text):
        return "marketing_text_placeholder_residue"
    return None


def validate_builder1_marketing_text_hygiene(text: object) -> None:
    err = marketing_placeholder_error(text)
    if err:
        raise ValueError(err)
    count = count_marketing_words(text)
    if count != MARKETING_TEXT_WORD_COUNT:
        raise ValueError(f"marketing_text_word_count_{count}")
