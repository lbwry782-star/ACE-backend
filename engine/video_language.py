"""
Video pipeline: strict language detection from product description (no LLM guessing).

Allowed output languages: Hebrew (he), English (en), Russian (ru), Arabic (ar).
Default when unclear or unsupported: Hebrew.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Tuple

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_LANGUAGES = frozenset({"he", "en", "ru", "ar"})


def normalize_video_content_language(code: str) -> str:
    """Allowed: he | en | ru | ar. Anything invalid or missing → he (fixed default)."""
    c = (code or "he").strip().lower()
    return c if c in ALLOWED_VIDEO_LANGUAGES else "he"

# Dominant script must reach this share of letter characters to be "clear"
_DOMINANCE_MIN = 0.55
# Second-place script must stay at or below this share for a clear winner (avoids mixed)
_SECOND_MAX_FOR_CLEAR = 0.28
# Latin text: classify as English only if enough common English tokens appear
_EN_WORD_HIT_RATIO = 0.22
_EN_WORD_HIT_MIN = 5

# Deterministic English lexicon for Latin-only classification (no external langid)
_EN_COMMON_WORDS = frozenset(
    x.lower()
    for x in """
    the a an and or but if as of at to in on for with from by about into through during before
    after above below between under over out up down off than then so such only own same other
    another each every both few more most some no nor not any all any both each few more most
    other some such no nor not only own same so than too very can will just should now discover
    learn try start today free help make take come go see know think look want need feel seem
    keep let put mean set say ask work hear play run move live believe hold bring happen write
    provide sit stand lose pay meet include continue change lead understand watch follow stop
    create speak read allow add spend grow open walk win offer remember love consider appear
    buy wait serve die send expect build stay fall cut reach kill remain suggest raise pass sell
    require report decide pull return explain carry develop thank agree support hit produce eat
    cover catch draw choose refer close act support product quality best new get use your our
    their its this that these those what which who whom whose when where why how all each both
    either neither one two first next last much many little few lot great good bad big small high
    low long short new old right left real sure such same different next early late young important
    public private available local global digital online easy hard fast slow full empty
    """.split()
)


def _is_hebrew_letter(ch: str) -> bool:
    o = ord(ch)
    return 0x0590 <= o <= 0x05FF


def _is_arabic_letter(ch: str) -> bool:
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF:
        return True
    if 0x0750 <= o <= 0x077F:
        return True
    if 0x08A0 <= o <= 0x08FF:
        return True
    if 0xFB50 <= o <= 0xFDFF:
        return True
    if 0xFE70 <= o <= 0xFEFF:
        return True
    return False


def _is_cyrillic_letter(ch: str) -> bool:
    o = ord(ch)
    return 0x0400 <= o <= 0x04FF


def _is_latin_letter(ch: str) -> bool:
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return True
    o = ord(ch)
    # Latin-1 supplement + Latin Extended-A/B (covers é, ñ, etc. — treated as Latin script, not English)
    if 0x00C0 <= o <= 0x024F:
        return True
    if 0x1E00 <= o <= 0x1EFF:
        return True
    return False


def _letter_category_counts(text: str) -> Tuple[int, int, int, int, int, int]:
    """Returns (he, ar, ru, latin, other_letters, total_letters)."""
    h = a = r = lat = other = 0
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        if _is_hebrew_letter(ch):
            h += 1
        elif _is_arabic_letter(ch):
            a += 1
        elif _is_cyrillic_letter(ch):
            r += 1
        elif _is_latin_letter(ch):
            lat += 1
        elif cat in ("Lu", "Ll", "Lt", "Lo", "Lm"):
            # Other alphabets (e.g. Greek) — not an allowed video language
            other += 1
    total = h + a + r + lat + other
    return h, a, r, lat, other, total


def _latin_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-zÀ-ÿĀ-ž]+", text)
    return [w.lower() for w in words if w]


def _latin_looks_english(text: str) -> bool:
    words = _latin_words(text)
    if not words:
        return False
    hits = sum(1 for w in words if w in _EN_COMMON_WORDS)
    if hits >= _EN_WORD_HIT_MIN:
        return True
    return hits / len(words) >= _EN_WORD_HIT_RATIO


def strip_hebrew_niqqud(s: str) -> str:
    """Remove Hebrew niqqud / cantillation (no vowel marks in overlay)."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x0591 <= o <= 0x05C7:
            continue
        if ch == "\u05BF":  # rafe
            continue
        out.append(ch)
    return "".join(out)


def strip_arabic_diacritics(s: str) -> str:
    """Remove Arabic harakat / tatweel (no vowel marks in overlay)."""
    out = []
    for ch in s:
        o = ord(ch)
        if o == 0x0640:  # tatweel
            continue
        if 0x064B <= o <= 0x065F:
            continue
        if 0x0610 <= o <= 0x061A:
            continue
        if 0x06D6 <= o <= 0x06ED:
            continue
        out.append(ch)
    return "".join(out)


def normalize_video_overlay_text(headline: str, language: str) -> str:
    """Strip vocalization marks; language is he|en|ru|ar."""
    lang = (language or "he").strip().lower()
    t = headline
    if lang == "he":
        t = strip_hebrew_niqqud(t)
    elif lang == "ar":
        t = strip_arabic_diacritics(t)
    return t


def video_language_display_name(code: str) -> str:
    return {
        "he": "Hebrew",
        "en": "English",
        "ru": "Russian",
        "ar": "Arabic",
    }.get((code or "he").strip().lower(), "Hebrew")


def detect_product_description_language(product_description: str) -> Tuple[str, str, float, bool]:
    """
    Classify product description into allowed video output language.

    Returns:
        detected_label: "he" | "en" | "ru" | "ar" | "mixed" | "unsupported_latin" | "empty"
        applied_code: always one of he|en|ru|ar (Hebrew when defaulting)
        confidence: 0.0–1.0 (heuristic, deterministic per input)
        used_default_hebrew: True if applied Hebrew due to unclear/unsupported/empty
    """
    raw = (product_description or "").strip()
    if not raw:
        return "empty", "he", 0.0, True

    h, a, r, lat, other, total = _letter_category_counts(raw)
    if total == 0:
        return "empty", "he", 0.0, True

    # Non-allowed scripts (e.g. Greek) present → unclear → default Hebrew
    if other > 0:
        return "mixed", "he", 0.42, True

    shares = [
        ("he", h / total),
        ("ar", a / total),
        ("ru", r / total),
        ("en", lat / total),  # provisional label for Latin bucket
    ]
    shares.sort(key=lambda x: x[1], reverse=True)
    top_name, top_s = shares[0]
    second_s = shares[1][1]

    clear_dominance = top_s >= _DOMINANCE_MIN and second_s <= _SECOND_MAX_FOR_CLEAR

    if not clear_dominance:
        return "mixed", "he", 0.45, True

    if top_name == "he":
        return "he", "he", 0.92, False
    if top_name == "ar":
        return "ar", "ar", 0.92, False
    if top_name == "ru":
        return "ru", "ru", 0.92, False

    # Latin-dominant clear: English only if lexicon heuristic passes; else unsupported → Hebrew
    if _latin_looks_english(raw):
        return "en", "en", 0.88, False
    return "unsupported_latin", "he", 0.5, True


def log_video_language_decision(
    product_description: str,
) -> Tuple[str, float, bool]:
    """
    Run detection, emit required logs, return (applied_code, confidence, default_hebrew).
    """
    det_label, applied, conf, default_he = detect_product_description_language(product_description)
    logger.info("VIDEO_LANGUAGE_DETECTED=%s", det_label)
    logger.info("VIDEO_LANGUAGE_CONFIDENCE=%s", round(conf, 3))
    logger.info("VIDEO_LANGUAGE_APPLIED=%s", applied)
    logger.info("VIDEO_LANGUAGE_DEFAULT_HEBREW=%s", default_he)
    return applied, conf, default_he
