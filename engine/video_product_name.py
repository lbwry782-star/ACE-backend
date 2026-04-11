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

from engine.video_language import (
    is_english_only_product_name_script,
    is_hebrew_or_english_product_name_script,
    normalize_video_content_language,
)

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


def generate_auto_product_name(product_description: str, marketing_language: str) -> str:
    """
    Auto product name when user left name empty.
    marketing_language he: name may be Hebrew or English (Latin) only.
    marketing_language en: name must be English (Latin) only.
    Raises VideoProductNameError on failure.
    """
    desc = (product_description or "").strip()
    if not desc:
        raise VideoProductNameError("empty_description")

    lang = normalize_video_content_language(marketing_language)
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise VideoProductNameError("no_openai_key")

    model = _product_name_model()
    if lang == "en":
        user_prompt = f"""Product description:
{desc}

Task: Invent a product name in English only.
Output exactly ONE short brand-like name (2–5 words) using Latin letters only.
Rules:
- Output only the name on a single line: no quotes, no labels (no "Name:"), no explanation, no second line.
- English only: Latin script (A–Z, a–z). No Hebrew, Arabic, Cyrillic, or other scripts. Digits allowed if natural.
- Do not generate names in any other language. Short loanwords (e.g. AI, SaaS) are allowed when natural."""

        system_msg = (
            "Invent a product name in English only. Reply with only that name: 2–5 words, Latin script, one line, no quotes."
        )
    else:
        user_prompt = f"""Product description:
{desc}

Task: Invent a product name in Hebrew or English only.
Output exactly ONE short brand-like name (2–5 words) that fits this description.
Rules:
- Output only the name on a single line: no quotes, no labels (no "Name:"), no explanation, no second line.
- Use Hebrew letters and/or Latin (English) letters only — no Arabic, Cyrillic, or other alphabets. Prefer a cohesive name in Hebrew or in English; short Latin acronyms inside a Hebrew name are fine when natural.
- Do not generate names in any other language."""

        system_msg = (
            "Invent a product name in Hebrew or English only. Reply with only that name: 2–5 words, one line, no quotes."
        )

    t = min(15.0, _timeout_s())
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(connect=t, read=_timeout_s(), write=t, pool=t),
        max_retries=0,
    )
    extra_user = ""
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": user_prompt + extra_user,
                    },
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
        if lang == "en" and not is_english_only_product_name_script(name):
            if attempt == 0:
                extra_user = "\n\nYour previous answer contained non-Latin letters. Reply again with Latin letters only (English product name)."
                continue
            raise VideoProductNameError("auto_product_name_not_english")
        if lang == "he" and not is_hebrew_or_english_product_name_script(name):
            if attempt == 0:
                extra_user = (
                    "\n\nYour previous answer used letters outside Hebrew or English. "
                    "Reply again with a name in Hebrew script OR English (Latin) only — no other alphabets."
                )
                continue
            raise VideoProductNameError("auto_product_name_invalid_script")
        return name
    raise VideoProductNameError("empty_or_invalid_model_output")


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
    *,
    marketing_language: str | None = None,
) -> Tuple[str, str]:
    """
    Returns (source 'user'|'auto', canonical_name).

    User-provided name: validated with content_language (video/headline context).
    Auto-generated name: uses marketing_language (from product description letter counts);
    if marketing_language is omitted, falls back to content_language.
    """
    lang = normalize_video_content_language(content_language)
    mk = normalize_video_content_language(marketing_language) if marketing_language is not None else lang
    u = (user_product_name or "").strip()
    if u:
        if lang == "en" and not is_english_only_product_name_script(u):
            raise VideoProductNameError("product_name_not_english")
        return "user", u
    logger.info("PRODUCT_NAME_GENERATION_REQUIRED=true")
    if mk == "he":
        logger.info("PRODUCT_NAME_LANGUAGE_POLICY marketing_lang=he allowed=he|en")
    else:
        logger.info("PRODUCT_NAME_LANGUAGE_POLICY marketing_lang=en allowed=en")
    auto = generate_auto_product_name(product_description, mk)
    if mk == "en" and not is_english_only_product_name_script(auto):
        raise VideoProductNameError("auto_product_name_not_english")
    if mk == "he" and not is_hebrew_or_english_product_name_script(auto):
        raise VideoProductNameError("auto_product_name_invalid_script")
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
