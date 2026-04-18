"""
Image / Builder1 pipeline: marketing copy language helpers (he | en).

Separated from engine.video_language so image code does not depend on video modules.
Logic mirrors the subset previously imported from video_language for side_by_side only.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_CONTENT_LANGUAGES: Final[frozenset[str]] = frozenset({"he", "en"})


def detect_text_language(product_description: str) -> str:
    """
    Deterministic marketing-language tag from product description only (he | en).

    Counts Hebrew letters (Unicode U+0590–U+05FF) vs ASCII Latin (A–Z, a–z).
    Ignores digits, whitespace, and all other characters (including punctuation).
    If Hebrew count > English count → "he"; else → "en". If both zero → "en".
    """
    hebrew_count = 0
    english_count = 0
    for ch in product_description or "":
        if ch.isspace() or ch.isdigit():
            continue
        o = ord(ch)
        if 0x0590 <= o <= 0x05FF:
            hebrew_count += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            english_count += 1

    if hebrew_count == 0 and english_count == 0:
        lang = "en"
    elif hebrew_count > english_count:
        lang = "he"
    else:
        lang = "en"

    logger.info(
        "MARKETING_LANGUAGE_DETECTED lang=%s heb=%s en=%s",
        lang,
        hebrew_count,
        english_count,
    )
    return lang


def normalize_content_language(code: str) -> str:
    """Allowed: he | en. Legacy ru/ar/unknown → he."""
    c = (code or "he").strip().lower()
    if c in ALLOWED_IMAGE_CONTENT_LANGUAGES:
        return c
    return "he"


def content_language_display_name(code: str) -> str:
    return {"he": "Hebrew", "en": "English"}.get(
        normalize_content_language(code), "Hebrew"
    )
