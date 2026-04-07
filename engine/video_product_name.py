"""
Video pipeline: single canonical product name (user-provided or auto-generated once per job).
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Dict, Tuple

import httpx
from openai import OpenAI

from engine.video_language import normalize_video_content_language, video_language_display_name

logger = logging.getLogger(__name__)


class VideoProductNameError(Exception):
    """Auto product-name generation failed; job must not continue with inconsistent naming."""


def _word_limit(s: str, max_words: int) -> str:
    words = (s or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _product_name_model() -> str:
    return (os.environ.get("VIDEO_PRODUCT_NAME_MODEL") or "gpt-4o-mini").strip()


def _timeout_s() -> float:
    return float((os.environ.get("VIDEO_PRODUCT_NAME_TIMEOUT_SECONDS") or "45").strip() or "45")


def generate_auto_product_name(product_description: str, content_language: str) -> str:
    """
    One short natural product name in the classified content language. Raises VideoProductNameError on failure.
    """
    desc = (product_description or "").strip()
    if not desc:
        raise VideoProductNameError("empty_description")

    lang = normalize_video_content_language(content_language)
    lang_name = video_language_display_name(lang)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise VideoProductNameError("no_openai_key")

    model = _product_name_model()
    user_prompt = f"""Product description:
{desc}

Task: Output exactly ONE short product name (2–5 words) in {lang_name} that fits this description.
Rules:
- Output only the name on a single line: no quotes, no labels (no "Name:"), no explanation, no second line.
- The name must be natural in {lang_name}. Short international loanwords (e.g. AI, SaaS) are allowed only when natural in that language."""

    t = min(15.0, _timeout_s())
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(connect=t, read=_timeout_s(), write=t, pool=t),
        max_retries=0,
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"You reply with only a product name in {lang_name}, 2–5 words, one line, no quotes.",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=80,
        )
        raw = (r.choices[0].message.content or "") if r.choices else ""
    except Exception as e:
        logger.error("VIDEO_PRODUCT_NAME_AUTO_FAIL err_type=%s err=%s", type(e).__name__, e)
        raise VideoProductNameError("openai_call_failed") from e

    name = _clean_single_line_name(raw)
    if not name or len(name) > 200:
        raise VideoProductNameError("empty_or_invalid_model_output")
    return name


def _clean_single_line_name(raw: str) -> str:
    s = (raw or "").strip()
    s = s.strip('"\'')
    s = re.sub(r"^name\s*:\s*", "", s, flags=re.I)
    s = s.split("\n")[0].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def resolve_video_product_name(
    user_product_name: str,
    product_description: str,
    content_language: str,
) -> Tuple[str, str]:
    """
    Returns (source 'user'|'auto', canonical_name).
    """
    u = (user_product_name or "").strip()
    if u:
        return "user", u
    auto = generate_auto_product_name(product_description, content_language)
    return "auto", auto


def apply_canonical_product_name_to_video_plan(plan: Dict[str, Any], canonical: str) -> None:
    """
    Single source of truth for productNameResolved and headlineText vs headlineDecision.
    Mutates plan in place.
    """
    c = (canonical or "").strip()
    if not c:
        return
    plan["productNameResolved"] = c
    hd = (plan.get("headlineDecision") or "").strip()
    ht = (plan.get("headlineText") or "").strip()
    if hd == "no_headline":
        return
    if hd == "product_name_only":
        plan["headlineText"] = _word_limit(c, 7)
        return
    if hd == "include_product_name":
        if c in ht:
            limited = _word_limit(ht, 7)
            if c in limited:
                plan["headlineText"] = limited
            else:
                plan["headlineText"] = _word_limit(f"{c} {ht}".strip(), 7)
        else:
            plan["headlineText"] = _word_limit(f"{c} {ht}".strip(), 7)
        return


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def product_name_reused_in_headline(canonical: str, headline: str, headline_decision: str) -> bool:
    hd = (headline_decision or "").strip()
    if hd == "no_headline":
        return True
    c = _nfc((canonical or "").strip())
    h = _nfc((headline or "").strip())
    if not c:
        return True
    if hd == "product_name_only":
        # Headline is capped at 7 words; canonical may be longer — compare to the same cap.
        return h == _word_limit(c, 7)
    return c in h


def product_name_reused_in_copy(canonical: str, copy: str) -> bool:
    c = _nfc((canonical or "").strip())
    if not c:
        return True
    return c in _nfc(copy or "")
