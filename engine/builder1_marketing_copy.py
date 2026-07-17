"""
Builder1 per-ad marketing copy validation (exactly 50 words).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

MARKETING_TEXT_WORD_COUNT = 50

_EN_WORD_HIT_RATIO = 0.22
_EN_WORD_HIT_MIN = 5
_EN_COMMON_WORDS = frozenset(
    x.lower()
    for x in """
    the a an and or but if as of at to in on for with from by about into through during before
    after above below between under over out up down off than then so such only own same other
    another each every both few more most some no nor not any all can will just should now discover
    learn try start today free help make take come go see know think look want need feel seem
    keep let put mean set say ask work hear play run move live believe hold bring happen write
    provide sit stand lose pay meet include continue change lead understand watch follow stop
    create speak read allow add spend grow open walk win offer remember love consider appear
    buy wait serve die send expect build stay fall cut reach kill remain suggest raise pass sell
    require report decide pull return explain carry develop thank agree support hit produce eat
    cover catch draw choose refer close act support product quality best new get use your our
    their its this that these those what which who whom whose when where why how much many
    """.split()
)


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


def normalize_campaign_language(code: object) -> str:
    raw = str(code or "en").strip().lower()
    if raw.startswith("he"):
        return "he"
    return "en"


def _is_hebrew_letter(ch: str) -> bool:
    return 0x0590 <= ord(ch) <= 0x05FF


def _is_latin_letter(ch: str) -> bool:
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return True
    o = ord(ch)
    return 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF


def count_marketing_script_characters(text: object) -> Tuple[int, int]:
    normalized = normalize_marketing_text(text)
    hebrew = latin = 0
    for ch in normalized:
        if ch.isspace() or ch.isdigit():
            continue
        if _is_hebrew_letter(ch):
            hebrew += 1
        elif _is_latin_letter(ch):
            latin += 1
        else:
            cat = unicodedata.category(ch)
            if cat in ("Lu", "Ll", "Lt", "Lo", "Lm"):
                if _is_hebrew_letter(ch):
                    hebrew += 1
                elif _is_latin_letter(ch):
                    latin += 1
    return hebrew, latin


def _latin_portion_looks_english(text: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿĀ-ž]+", text)
    if not words:
        return False
    lowered = [w.lower() for w in words]
    hits = sum(1 for w in lowered if w in _EN_COMMON_WORDS)
    if hits >= _EN_WORD_HIT_MIN:
        return True
    return hits / len(lowered) >= _EN_WORD_HIT_RATIO


def _dominant_language(hebrew: int, latin: int, text: str, *, fallback: str) -> str:
    if hebrew > latin:
        return "he"
    if latin > hebrew:
        return "en"
    if hebrew == 0 and latin == 0:
        return fallback
    return "en" if _latin_portion_looks_english(text) else "he"


def validate_marketing_text_language(text: object, detected_language: str) -> Dict[str, Any]:
    """Deterministic dominant-language check tolerant of brand names and technical tokens."""
    target = normalize_campaign_language(detected_language)
    normalized = normalize_marketing_text(text)
    hebrew, latin = count_marketing_script_characters(normalized)
    dominant = _dominant_language(hebrew, latin, normalized, fallback=target)
    valid = dominant == target
    return {
        "valid": valid,
        "targetLanguage": target,
        "dominantLanguage": dominant,
        "hebrewCharacterCount": hebrew,
        "latinCharacterCount": latin,
    }


def marketing_text_language_error(text: object, detected_language: str) -> Optional[str]:
    result = validate_marketing_text_language(text, detected_language)
    if result["valid"]:
        return None
    return "marketing_copy_wrong_language"
