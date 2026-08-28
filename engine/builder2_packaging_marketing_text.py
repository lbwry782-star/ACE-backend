"""
Builder2 packaging marketing text — ~50-word delivery copy outside media isolation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

_PACKAGING_MIN_WORDS = 45
_PACKAGING_MAX_WORDS = 55
_DETERMINISTIC_FALLBACK_MARKERS = (
    " Video delivery.",
    " וידאו.",
)


def count_packaging_marketing_words(text: str) -> int:
    return len(re.findall(r"\S+", (text or "").strip()))


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
    word_count = count_packaging_marketing_words(token)
    return word_count < _PACKAGING_MIN_WORDS or word_count > _PACKAGING_MAX_WORDS


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
    the established Builder2 packaging copy generator (GPT with deterministic fallback).
    Must run only when media-resume reasoning isolation is inactive.
    """
    if not is_insufficient_delivery_marketing_text(existing_text, source=existing_source):
        return (existing_text or "").strip(), existing_source or "delivery_existing"

    plan = plan if isinstance(plan, dict) else {}
    from engine.video_language import normalize_video_content_language

    lang = normalize_video_content_language(content_language or plan.get("language") or plan.get("marketingLanguage") or "en")
    ad_goal = _advertising_promise_from_plan(plan)
    from engine.runway_video import _fallback_packaging_marketing_copy

    generated = _fallback_packaging_marketing_copy(
        (product_name or plan.get("productNameResolved") or "").strip() or "Product",
        (product_description or "").strip(),
        ad_goal,
        output_language=lang,
        headline_text=(headline_text or plan.get("headlineText") or "").strip(),
    )
    return (generated or "").strip(), "packaging_copy"
