"""
Video pipeline: language detection from product description for headline + marketing copy.

Allowed output languages: Hebrew (he), English (en), Russian (ru), Arabic (ar).
Dominant language = plurality among letter counts (Hebrew / Arabic / Cyrillic / Latin).
Short Latin loanwords (e.g. AI, SaaS) do not flip the dominant script when Hebrew/Russian/Arabic letters lead.
Default Hebrew only when empty, disallowed script dominates, or Latin-plurality without English-like text.
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

# Latin-plurality descriptions: treat as English output only if lexicon suggests English
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


def text_predominantly_matches_language(text: str, lang_code: str) -> bool:
    """
    True if the text's letter-count plurality matches lang_code (he|en|ru|ar).
    Ignores disallowed 'other' letters for plurality among the four supported buckets.
    Empty or no supported letters → True (nothing to contradict).
    """
    lang = normalize_video_content_language(lang_code)
    h, a, r, lat, _other, _total = _letter_category_counts(text)
    supported = h + a + r + lat
    if supported == 0:
        return True
    counts_map = {"he": h, "ar": a, "ru": r, "en": lat}
    max_v = max(counts_map.values())
    winners = [k for k, v in counts_map.items() if v == max_v]
    tie_break = ("he", "ar", "ru", "en")
    top = next(k for k in tie_break if k in winners)
    return top == lang


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
        detected_label: winning script code, or mixed_plurality (tie), unsupported_latin, unsupported_script, empty
        applied_code: always one of he|en|ru|ar (Hebrew when defaulting)
        confidence: 0.0–1.0 (top script share among letter counts)
        used_default_hebrew: True if applied Hebrew due to empty / unsupported_script / unsupported_latin
    """
    raw = (product_description or "").strip()
    if not raw:
        return "empty", "he", 0.0, True

    h, a, r, lat, other, total = _letter_category_counts(raw)
    if total == 0:
        return "empty", "he", 0.0, True

    # Disallowed script (e.g. Greek) is the largest bucket → cannot pick he/ar/ru/en reliably
    if other > max(h, a, r, lat):
        return "unsupported_script", "he", 0.4, True

    counts_map = {"he": h, "ar": a, "ru": r, "en": lat}
    max_v = max(counts_map.values())
    winners = [k for k, v in counts_map.items() if v == max_v]
    tie_break = ("he", "ar", "ru", "en")
    top_name = next(k for k in tie_break if k in winners)
    det_label = top_name if len(winners) == 1 else "mixed_plurality"

    sorted_counts = sorted(counts_map.values(), reverse=True)
    second_v = sorted_counts[1] if len(sorted_counts) > 1 else 0
    confidence = max_v / total if total else 0.0

    if top_name == "he":
        return det_label, "he", confidence, False
    if top_name == "ar":
        return det_label, "ar", confidence, False
    if top_name == "ru":
        return det_label, "ru", confidence, False

    # Latin has plurality among letter buckets
    if _latin_looks_english(raw):
        return det_label, "en", confidence, False
    return "unsupported_latin", "he", confidence, True


def log_video_language_decision(
    product_description: str,
) -> Tuple[str, float, bool]:
    """
    Run detection, emit required logs, return (applied_code, confidence, default_hebrew).
    """
    det_label, applied, conf, default_he = detect_product_description_language(product_description)
    raw = (product_description or "").strip()
    h, a, r, lat, _o, tot = _letter_category_counts(raw) if raw else (0, 0, 0, 0, 0, 0)
    counts_map = {"he": h, "ar": a, "ru": r, "en": lat}
    sorted_counts = sorted(counts_map.values(), reverse=True)
    second_v = sorted_counts[1] if tot and len(sorted_counts) > 1 else 0
    second_share = (second_v / tot) if tot else 0.0
    # Policy: allow short foreign/loan terms in outputs when description had secondary script activity
    mixed_terms_allowed = second_share >= 0.03

    logger.info("VIDEO_LANGUAGE_DETECTED=%s", det_label)
    logger.info("VIDEO_LANGUAGE_CONFIDENCE=%s", round(conf, 3))
    logger.info("VIDEO_LANGUAGE_APPLIED=%s", applied)
    logger.info("VIDEO_LANGUAGE_DEFAULT_HEBREW=%s", default_he)
    logger.info("VIDEO_MIXED_LANGUAGE_TERMS_ALLOWED=%s", mixed_terms_allowed)
    return applied, conf, default_he
