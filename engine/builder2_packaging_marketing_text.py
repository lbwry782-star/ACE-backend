"""
Builder2 packaging marketing text — ~50-word delivery copy outside media isolation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

_PACKAGING_MIN_WORDS = 45
_PACKAGING_MAX_WORDS = 55
_DETERMINISTIC_FALLBACK_MARKERS = (
    " Video delivery.",
    " וידאו.",
)

# Known template / meta placeholders — not a general parenthesis ban.
_PLACEHOLDER_RESIDUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\(\s*המוצר\s+הזה\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*this\s+product\s*\)", re.IGNORECASE),
    re.compile(r"\[\s*product\s*\]", re.IGNORECASE),
    re.compile(r"\{\s*product\s*\}", re.IGNORECASE),
    re.compile(r"\(\s*product\s*\)", re.IGNORECASE),
    re.compile(r"שם\s+המוצר"),
)


def count_packaging_marketing_words(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))


def has_builder2_packaging_placeholder_residue(text: str) -> bool:
    token = (text or "").strip()
    if not token:
        return False
    return any(pattern.search(token) for pattern in _PLACEHOLDER_RESIDUE_PATTERNS)


def sanitize_builder2_packaging_marketing_text(text: str) -> str:
    """Remove known template placeholders without rejecting legitimate parentheses."""
    s = " ".join((text or "").split()).strip()
    if not s:
        return s
    for pattern in _PLACEHOLDER_RESIDUE_PATTERNS:
        s = pattern.sub("", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?":
        s = f"{s}."
    return s


def finalize_builder2_packaging_paragraph(text: str, *, lang: str = "en") -> str:
    s = " ".join((text or "").split()).strip()
    if not s:
        return s
    words = s.split()
    if len(words) > _PACKAGING_MAX_WORDS:
        truncated = " ".join(words[:_PACKAGING_MAX_WORDS])
        cut = max(truncated.rfind(". "), truncated.rfind("! "), truncated.rfind("? "))
        s = truncated[: cut + 1].strip() if cut > 0 else truncated.strip()
    if count_packaging_marketing_words(s) < _PACKAGING_MIN_WORDS:
        pad = (
            "כך הבחירה נשארת ברורה, יציבה ומעשית לאורך הדרך."
            if lang == "he"
            else "This keeps the choice clear, steady, and practical all the way through."
        )
        s = f"{s} {pad}".strip()
    if s and s[-1] not in ".!?":
        s = f"{s}."
    return s


def is_insufficient_delivery_marketing_text(
    text: str,
    *,
    source: str = "",
) -> bool:
    token = (text or "").strip()
    if not token:
        return True
    if (source or "").strip() == "deterministic_fallback":
        return True
    for marker in _DETERMINISTIC_FALLBACK_MARKERS:
        if token.endswith(marker) and count_packaging_marketing_words(token) < _PACKAGING_MIN_WORDS:
            return True
    if has_builder2_packaging_placeholder_residue(token):
        return True
    word_count = count_packaging_marketing_words(token)
    return word_count < _PACKAGING_MIN_WORDS or word_count > _PACKAGING_MAX_WORDS


def _prepare_existing_packaging_text(
    existing_text: str,
    existing_source: str,
    *,
    lang: str,
) -> Tuple[str, str, bool]:
    token = (existing_text or "").strip()
    source = existing_source or "delivery_existing"
    if token and not has_builder2_packaging_placeholder_residue(token):
        if not is_insufficient_delivery_marketing_text(token, source=source):
            return token, source, True

    sanitized = sanitize_builder2_packaging_marketing_text(existing_text)
    if sanitized != token and sanitized:
        source = "delivery_sanitized"
    if sanitized and not is_insufficient_delivery_marketing_text(sanitized, source=source):
        return sanitized, source, True
    finalized = finalize_builder2_packaging_paragraph(sanitized, lang=lang)
    if finalized and not is_insufficient_delivery_marketing_text(finalized, source="delivery_sanitized"):
        return finalized, "delivery_sanitized", True
    return sanitized or token, source, False


def _advertising_promise_from_plan(plan: Dict[str, Any]) -> str:
    for key in ("advertisingPromise", "advertising_promise", "promiseText"):
        value = str(plan.get(key) or "").strip()
        if value:
            return value
    closure = plan.get("advertisingClosure")
    if isinstance(closure, dict):
        slogan = str(closure.get("sloganText") or "").strip()
        if slogan:
            return slogan
    return ""


def ensure_builder2_packaging_marketing_text(
    *,
    existing_text: str = "",
    existing_source: str = "",
    product_name: str,
    product_description: str,
    plan: Optional[Dict[str, Any]] = None,
    content_language: str = "",
    headline_text: str = "",
) -> tuple[str, str]:
    """
    Return (marketingText, source). Reuses sufficient existing text; otherwise invokes
    the Builder2 packaging copy generator (GPT with deterministic fallback).
    Must run only when media-resume reasoning isolation is inactive.
    """
    plan = plan if isinstance(plan, dict) else {}
    from engine.video_language import normalize_video_content_language

    lang = normalize_video_content_language(content_language or plan.get("language") or plan.get("marketingLanguage") or "en")

    prepared, prepared_source, ready = _prepare_existing_packaging_text(
        existing_text,
        existing_source,
        lang=lang,
    )
    if ready:
        return prepared, prepared_source

    ad_goal = _advertising_promise_from_plan(plan)
    from engine.runway_video import generate_builder2_packaging_marketing_copy

    generated = generate_builder2_packaging_marketing_copy(
        (product_name or plan.get("productNameResolved") or "").strip() or "Product",
        (product_description or "").strip(),
        ad_goal,
        output_language=lang,
        headline_text=(headline_text or plan.get("headlineText") or "").strip(),
    )
    cleaned = sanitize_builder2_packaging_marketing_text(generated or "")
    cleaned = finalize_builder2_packaging_paragraph(cleaned, lang=lang)
    return (cleaned or "").strip(), "packaging_copy"
