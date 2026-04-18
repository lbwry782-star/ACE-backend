"""
Video pipeline: language detection from product description only (headline + marketing copy).

Supported content languages: Hebrew (he) and English (en) only.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Tuple

logger = logging.getLogger(__name__)

ALLOWED_VIDEO_LANGUAGES = frozenset({"he", "en"})


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


def normalize_video_content_language(code: str) -> str:
    """Allowed: he | en. Legacy ru/ar/unknown → he."""
    c = (code or "he").strip().lower()
    if c in ALLOWED_VIDEO_LANGUAGES:
        return c
    return "he"


# Latin text: classify as English only if enough common English tokens appear
_EN_WORD_HIT_RATIO = 0.22
_EN_WORD_HIT_MIN = 5

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
    if 0x00C0 <= o <= 0x024F:
        return True
    if 0x1E00 <= o <= 0x1EFF:
        return True
    return False


def is_hebrew_or_english_product_name_script(s: str) -> bool:
    """
    True if every letter in s is Hebrew or Latin (no Arabic, Cyrillic, or other scripts).
    Non-letters (digits, punctuation, spaces) are allowed. Empty → False.
    """
    raw = (s or "").strip()
    if not raw:
        return False
    for ch in raw:
        cat = unicodedata.category(ch)
        if cat in ("Lu", "Ll", "Lt", "Lo", "Lm"):
            if _is_hebrew_letter(ch) or _is_latin_letter(ch):
                continue
            return False
    return True


def _letter_buckets_for_video(text: str) -> Tuple[int, int, int, int]:
    """
    Returns (hebrew_count, latin_count, foreign_letter_count, total_letters).
    foreign = Arabic, Cyrillic, and other alphabetic letters (not Hebrew/Latin).
    """
    h = lat = foreign = 0
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        if _is_hebrew_letter(ch):
            h += 1
        elif _is_latin_letter(ch):
            lat += 1
        elif _is_arabic_letter(ch) or _is_cyrillic_letter(ch):
            foreign += 1
        elif cat in ("Lu", "Ll", "Lt", "Lo", "Lm"):
            foreign += 1
    total = h + lat + foreign
    return h, lat, foreign, total


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
        if ch == "\u05BF":
            continue
        out.append(ch)
    return "".join(out)


def normalize_video_overlay_text(headline: str, language: str) -> str:
    """Strip vocalization for Hebrew overlay; English unchanged."""
    lang = normalize_video_content_language(language)
    t = headline
    if lang == "he":
        t = strip_hebrew_niqqud(t)
    return t


def is_english_only_product_name_script(s: str) -> bool:
    """
    True if the string has no Hebrew, Arabic, Cyrillic, or non-Latin letters.
    Empty / whitespace-only → False.
    """
    raw = (s or "").strip()
    if not raw:
        return False
    for ch in raw:
        if _is_hebrew_letter(ch) or _is_arabic_letter(ch) or _is_cyrillic_letter(ch):
            return False
        cat = unicodedata.category(ch)
        if cat in ("Lu", "Ll", "Lt", "Lo", "Lm") and not _is_latin_letter(ch):
            return False
    return True


def _hebrew_letter_in(s: str) -> bool:
    return any(0x0590 <= ord(ch) <= 0x05FF for ch in (s or ""))


def _has_ascii_latin_letter(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def product_name_is_latin_only_for_bilingual_headline(pn: str) -> bool:
    """
    True when the resolved product name is English (Latin letters only, ≥1 letter).
    Used with Hebrew headline language: headline format ``<NAME> <Hebrew tail>`` (one ASCII space, no comma).
    """
    raw = (pn or "").strip()
    if not raw:
        return False
    h, lat, foreign, _ = _letter_buckets_for_video(raw)
    return foreign == 0 and h == 0 and lat >= 1


# Punctuation / marks the planner must not put in the Hebrew tail (structural gate only).
_HEADLINE_EN_HE_TAIL_FORBIDDEN = frozenset("•.:;-–—−\u2212\u00b7")
# No explicit bidi / direction overrides in stored headline — frontend owns display direction.
_HEADLINE_EN_HE_TAIL_INVISIBLE = frozenset("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def bilingual_en_he_headline_tail_struct_ok(tail: str) -> bool:
    """
    After '<productNameResolved> ' (one space; no comma), the remainder must be Hebrew-only (no Latin letters),
    contain at least one Hebrew letter, use no forbidden punctuation, no comma,
    and no bidi/isolate marks.
    """
    t = (tail or "").strip()
    if not t or not _hebrew_letter_in(t):
        return False
    if _has_ascii_latin_letter(t):
        return False
    if "," in t:
        return False
    for ch in t:
        if ch in _HEADLINE_EN_HE_TAIL_FORBIDDEN or ch in _HEADLINE_EN_HE_TAIL_INVISIBLE:
            return False
    return True


def hebrew_headline_allows_embedded_english_product_name(headline: str, canonical_name: str) -> bool:
    """
    Hebrew job: the headline may contain the resolved English product name as the only Latin segment.
    True when, after removing that name (case-insensitive, word-boundary safe), the remainder
    contains Hebrew and contains no Latin letters (no extra English beyond the product name).
    """
    cn = (canonical_name or "").strip()
    if not cn or not is_english_only_product_name_script(cn):
        return False
    h = unicodedata.normalize("NFC", (headline or "").strip())
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(cn) + r"(?![A-Za-z0-9])", re.I)
    remainder = pat.sub("", h)
    remainder = re.sub(r"[\s·\u00b7\-–—:|]+", " ", remainder).strip()
    if not _hebrew_letter_in(remainder):
        return False
    if _has_ascii_latin_letter(remainder):
        return False
    return True


def evaluate_headline_overlay_language(
    headline: str,
    required_lang: str,
    canonical_name: str,
) -> Tuple[bool, str, bool]:
    """
    Headline language check for ffmpeg overlay. Returns (passes, log_label, allowed_latin_product_exception).
    """
    req = normalize_video_content_language(required_lang)
    h = (headline or "").strip()
    if not h:
        return True, "empty", False
    if req == "en":
        ok = text_predominantly_matches_language(h, "en")
        return ok, ("plurality_english_ok" if ok else "plurality_english_failed"), False
    if text_predominantly_matches_language(h, "he"):
        return True, "plurality_hebrew_ok", False
    if hebrew_headline_allows_embedded_english_product_name(h, canonical_name):
        return True, "hebrew_with_only_canonical_english_product_name", True
    # Hebrew job: headline may be exactly the Latin product name only (allowed loanword/brand).
    cn = (canonical_name or "").strip()
    hn = unicodedata.normalize("NFC", h).strip()
    if (
        cn
        and is_english_only_product_name_script(cn)
        and is_english_only_product_name_script(hn)
        and hn.casefold() == unicodedata.normalize("NFC", cn).strip().casefold()
    ):
        return True, "hebrew_job_latin_product_name_headline_only", True
    return False, "headline_language_rules_failed", False


def text_predominantly_matches_language(text: str, lang_code: str) -> bool:
    """
    True if letter plurality matches the required video language (he | en).
    Hebrew: Hebrew letters strictly outrank Latin (ties favor Hebrew unless text looks English).
    English: Latin outranks Hebrew, or tie with English-like Latin vocabulary.
    Ignores foreign-script letters for the he vs en decision unless they dominate.
    """
    lang = normalize_video_content_language(lang_code)
    h, lat, foreign, total = _letter_buckets_for_video(text)
    if total == 0:
        return True
    if foreign > max(h, lat):
        return False
    if lang == "he":
        if h > lat:
            return True
        if lat > h:
            return False
        return not _latin_looks_english(text)
    # en
    if lat > h:
        return True
    if h > lat:
        return False
    return _latin_looks_english(text)


def video_language_display_name(code: str) -> str:
    return {"he": "Hebrew", "en": "English"}.get(
        normalize_video_content_language(code), "Hebrew"
    )


def detect_product_description_language(product_description: str) -> Tuple[str, str, float, bool]:
    """
    Classify product description into he | en (from description text only).

    Returns:
        detected_label: he | en | empty | unsupported_script | unsupported_latin | mixed_tie
        applied_code: always he or en
        confidence: top bucket share among Hebrew vs Latin (foreign excluded from share)
        used_default_hebrew: True if applied Hebrew due to empty / unsupported / non-English Latin
    """
    raw = (product_description or "").strip()
    if not raw:
        return "empty", "he", 0.0, True

    h, lat, foreign, total = _letter_buckets_for_video(raw)
    if total == 0:
        return "empty", "he", 0.0, True

    if foreign > max(h, lat):
        return "unsupported_script", "he", 0.4, True

    denom = h + lat
    if denom == 0:
        return "unsupported_script", "he", 0.4, True

    if h > lat:
        conf = h / denom
        return "he", "he", conf, False
    if lat > h:
        conf = lat / denom
        if _latin_looks_english(raw):
            return "en", "en", conf, False
        return "unsupported_latin", "he", conf, True

    # tie h == lat
    conf = h / denom if denom else 0.0
    if _latin_looks_english(raw):
        return "mixed_tie", "en", conf, False
    return "mixed_tie", "he", conf, False


def log_video_language_decision(
    product_description: str,
) -> Tuple[str, float, bool]:
    """
    Run detection, emit VIDEO_LANGUAGE_DETECTED and VIDEO_LANGUAGE_APPLIED, return (applied_code, confidence, default_hebrew).
    """
    det_label, applied, conf, default_he = detect_product_description_language(product_description)
    logger.info("VIDEO_LANGUAGE_DETECTED=%s", det_label)
    logger.info("VIDEO_LANGUAGE_APPLIED=%s", applied)
    return applied, conf, default_he
