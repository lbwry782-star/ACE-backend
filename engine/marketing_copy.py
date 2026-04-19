"""Shared marketing copy generation (45-55 words). Extracted from Builder1 engine for reuse."""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple

from openai import OpenAI

from engine import openai_retry
from engine.image_language import content_language_display_name, normalize_content_language
from engine.image_product_name import product_name_reused_in_copy

logger = logging.getLogger("engine.side_by_side_v1")


def _get_text_model() -> str:
    """Resolved text model; o4-mini is deprecated and mapped to o3-pro."""
    m = os.environ.get("OPENAI_TEXT_MODEL", "o3-pro")
    return "o3-pro" if m == "o4-mini" else m


def _repair_verbatim_marketing_product_name(copy_text: str, product_name: str, lang: str) -> str:
    """
    Ensure the canonical product name appears at least once when the model omitted it.

    For Hebrew copy with a Latin product name, avoid bare English-first paragraphs (e.g. \"NAME. …\"),
    which read unnaturally in RTL and confuse bidi. Prefer a short Hebrew function word before the name.
    English copy keeps a simple prefix.
    """
    n = (product_name or "").strip()
    t = (copy_text or "").strip()
    if not n or not t:
        return t or n
    if product_name_reused_in_copy(n, t):
        return t
    lc = normalize_content_language(lang)
    if lc == "he":
        return f"עם {n}, {t}".strip()
    return f"{n}. {t}".strip()


def _normalize_stray_quote_wrapping_around_name(text: str, name: str) -> Tuple[str, bool]:
    """Remove one redundant decorative quote pair around the exact product name (Hebrew marketing copy)."""
    n = (name or "").strip()
    t = text or ""
    if not n or not t:
        return text, False
    for ql, qr in (('"', '"'), ("\u201c", "\u201d"), ("«", "»"), ("״", "״")):
        needle = f"{ql}{n}{qr}"
        if needle in t:
            return t.replace(needle, n, 1), True
    return text, False


def _fix_leading_quoted_product_name_hebrew(text: str, name: str) -> Tuple[str, bool]:
    """Deterministic fix when copy opens with quote-wrapped product name (broken RTL / punctuation)."""
    s = (text or "").strip()
    n = (name or "").strip()
    if not s or not n:
        return text, False
    quote_like = frozenset('"\u201c\u201d״«\u05f4')
    idx = 0
    while idx < len(s) and s[idx].isspace():
        idx += 1
    if idx >= len(s) or s[idx] not in quote_like:
        return text, False
    j = idx
    while j < len(s) and (s[j] in quote_like or s[j].isspace()):
        j += 1
    if not s[j:].startswith(n):
        return text, False
    end = j + len(n)
    end2 = end
    while end2 < len(s) and (s[end2] in quote_like or s[end2].isspace() or s[end2] in "»'"):
        end2 += 1
    rest = s[end2:].strip()
    if not rest:
        return text, False
    return f"עם {n}, {rest}", True


def _hebrew_marketing_product_name_postprocess(text: str, product_name: str) -> Tuple[str, bool, str]:
    """
    Deterministic Hebrew marketing cleanup (no model calls).
    Returns (new_text, applied, reason) where reason is bad_prefix | stray_quotes | "".
    """
    n = (product_name or "").strip()
    t = (text or "").strip()
    if not n or not t:
        return text, False, ""
    t2, f1 = _fix_leading_quoted_product_name_hebrew(t, n)
    base = t2 if f1 else t
    t3, f2 = _normalize_stray_quote_wrapping_around_name(base, n)
    if f1:
        return t3, True, "bad_prefix"
    if f2:
        return t3, True, "stray_quotes"
    return text, False, ""


def _finalize_hebrew_marketing_after_name_fix(
    copy_text: str,
    canon: str,
    ad_goal: str,
    lang: str,
    *,
    require_verbatim_product_name: bool,
) -> str:
    """Re-trim/pad word count and restore verbatim name if required."""
    out = " ".join(copy_text.split()[:55])
    while len(out.split()) < 45:
        out = f"{out} {_marketing_copy_fallback(canon, ad_goal, lang)}".strip()
        out = " ".join(out.split()[:55])
    if require_verbatim_product_name and not product_name_reused_in_copy(canon, out):
        out = _repair_verbatim_marketing_product_name(out, canon, lang)
        out = " ".join(out.split()[:55])
    return out


def _apply_hebrew_marketing_name_postprocess_with_log(
    text: str,
    canon: str,
    ad_goal: str,
    lang: str,
    *,
    require_verbatim_product_name: bool,
) -> str:
    """Hebrew-only: deterministic name integration fix + logs (no model calls)."""
    if lang != "he" or not (canon or "").strip():
        return text
    out, pfn_applied, pfn_reason = _hebrew_marketing_product_name_postprocess(text, canon)
    if pfn_applied:
        out = _finalize_hebrew_marketing_after_name_fix(
            out,
            canon,
            ad_goal,
            lang,
            require_verbatim_product_name=require_verbatim_product_name,
        )
        logger.info(
            "MARKETING_TEXT_PRODUCT_NAME_FIXED applied=true reason=%s",
            pfn_reason,
        )
    else:
        logger.info("MARKETING_TEXT_PRODUCT_NAME_OK=true")
    return out


def _build_marketing_copy_user_prompt(
    product_name: str,
    product_description: str,
    ad_goal: str,
    lang_name: str,
    *,
    lang_code: str,
    strict_verbatim_name: bool,
    retry_tail: str = "",
) -> str:
    canon = (product_name or "").strip()
    desc = (product_description or "").strip()
    goal = (ad_goal or "").strip()
    verbatim = ""
    if strict_verbatim_name:
        verbatim = f"""
CANONICAL PRODUCT NAME — mandatory verbatim substring (at least once), exact characters:
{json.dumps(canon, ensure_ascii=False)}
- Paste this exact string into the marketing copy at least once (do not translate, transliterate, or rephrase it).
- Do NOT use the product description text (e.g. opening words like category phrases) as the product name. Description is context only.
- Do NOT invent an alternate brand or substitute name.{retry_tail}
"""
    else:
        verbatim = retry_tail
    hebrew_embedded = ""
    if lang_code == "he":
        hebrew_embedded = """
- Write natural, fluent Hebrew sentence structure.
- If you include an English product name, brand name, or English multi-word phrase inside a Hebrew sentence, keep normal left-to-right word order inside that English segment (e.g. "Fast Delivery" reads left-to-right as a phrase). Do not reverse English word order.
- If you include the product name, integrate it naturally into the Hebrew sentence (as subject, object, or after a preposition — not stuck at the very start unless grammar truly requires it).
- Do NOT open the paragraph with the bare English product name as the first token (e.g. "SHOESHOE. …"); lead with Hebrew and tuck the Latin name inside the sentence.
- Do NOT start the copy with a dangling quoted product name or a quote mark immediately wrapping the name before the rest of the sentence.
- Avoid patterns like: '"ProductName" ...' or a line that is only the name in quotes.
- Prefer natural phrasing such as: "<ProductName> היא ...", "עם <ProductName> ...", "בעזרת <ProductName> ..." (match gender/number to context).
"""
    return f"""Generate marketing copy for an advertisement.
{verbatim}
Product name: {canon}
Product description (context only): {desc}
Advertising goal: {goal}

Requirements:
- You MUST write the entire marketing copy in {lang_name}. Do not use any other language for sentences or clauses.
- Exception: short unavoidable loanwords (AI, SaaS, CRM, common abbreviations, or the canonical product name if Latin) may stay as-is; everything else must be {lang_name}.
- No mixed-language paragraphs; no alternating Hebrew/English sentences unless the job language is the only prose language.{hebrew_embedded}
- Exactly 45-55 words (count carefully)
- Must include product name
- Must be product-specific (not generic marketing language)
- Must include one short CTA (call-to-action) at the end
- Professional, compelling, clear
- No fluff, no generic phrases

Marketing copy:"""


def _marketing_copy_fallback(product_name: str, ad_goal: str, lang: str) -> str:
    """Short deterministic fallback in the requested language (image / marketing copy path)."""
    lang = normalize_content_language(lang)
    ag = (ad_goal or "").strip()
    pn = (product_name or "").strip() or "Product"
    if lang == "he":
        return (
            f"עם {pn} אפשר לענות בדיוק על הצורך הזה: {ag}. "
            f"הציעו חוויה ברורה, אמינה וממוקדת תוצאה למשתמשים שמחפשים פתרון אמיתי. "
            f"אל תפספסו את ההזדמנות לבחור במוצר שמקדם אתכם קדימה בביטחון. "
            f"גלו עוד, השוו והחליטו — והתחילו עוד היום."
        )
    return (
        f"{pn} helps you achieve {ag.lower()}. Discover how {pn} can transform your workflow. Get started today."
    )


class MarketingCopyVerbatimProductNameFailed(Exception):
    """Strict marketing copy could not include the canonical product name substring after repair."""


def generate_marketing_copy(
    product_name: str,
    product_description: str,
    ad_goal: str,
    max_retries: int = 2,
    output_language: str = "en",
    *,
    require_verbatim_product_name: bool = False,
) -> str:
    """
    Generate marketing copy: 45-55 words, product-specific, with CTA, in output_language.

    output_language: he | en (unknown codes normalize to he).
    When require_verbatim_product_name is True, the copy must contain the exact product_name substring
    (repair via prefix or fail); the model must not substitute the description as the name.
    """
    lang = normalize_content_language(output_language)
    lang_name = content_language_display_name(lang)
    canon = (product_name or "").strip() or "Product"
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_name = _get_text_model()

    max_attempts = max_retries + 1
    retry_tail = ""
    if lang == "he":
        system_msg = (
            "You are a marketing copywriter. You MUST write the marketing text entirely in Hebrew. "
            "Use natural, fluent Hebrew sentence structure. "
            "Do not produce English sentences or English marketing copy as the main prose. "
            "Short Latin loanwords (e.g. AI, SaaS) or the exact canonical product name if given in Latin may appear as-is; "
            "multi-word English phrases inside Hebrew must keep normal English left-to-right word order within the phrase. "
            "Integrate the product name naturally; do not open with a dangling quoted name. "
            "Return only the marketing copy text, no JSON, no quotes."
        )
    else:
        system_msg = (
            "You are a marketing copywriter. You MUST write the marketing text entirely in English. "
            "Do not produce Hebrew or other languages in the body of the copy. "
            "Return only the marketing copy text, no JSON, no quotes."
        )
    if require_verbatim_product_name:
        system_msg += (
            " When the user message includes a canonical product name as a JSON-quoted literal, "
            "include that exact character sequence in the copy at least once. "
            "Do not use paraphrases from the product description as the product name."
        )

    for attempt in range(max_attempts):
        prompt = _build_marketing_copy_user_prompt(
            canon,
            product_description,
            ad_goal,
            lang_name,
            lang_code=lang,
            strict_verbatim_name=require_verbatim_product_name,
            retry_tail=retry_tail,
        )
        try:
            logger.info(
                "MARKETING_COPY attempt=%s model=%s product=%s lang=%s verbatim=%s",
                attempt + 1,
                model_name,
                canon[:50],
                lang,
                require_verbatim_product_name,
            )

            def _copy_call():
                is_o_model = len(model_name) > 1 and model_name.startswith("o") and model_name[1].isdigit()
                if is_o_model:
                    full_input = f"{system_msg.strip()}\n\n{prompt}"
                    r = client.responses.create(model=model_name, input=full_input)
                    return r.output_text.strip()
                r = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                return r.choices[0].message.content.strip()

            copy_text = openai_retry.openai_call_with_retry(_copy_call, endpoint="responses")
            copy_text = copy_text.strip("\"'")

            word_count = len(copy_text.split())

            if word_count < 45:
                if attempt < max_attempts - 1:
                    logger.warning("MARKETING_COPY: word_count=%s < 45, retrying...", word_count)
                    retry_tail = (
                        (retry_tail + "\n\n" if retry_tail else "")
                        + f"MUST be 45-55 words (previous attempt was {word_count} words — too short)."
                    )
                    continue
                logger.warning("MARKETING_COPY: word_count=%s < 45, padding...", word_count)
                copy_text = f"{copy_text} {_marketing_copy_fallback(canon, ad_goal, lang)}".strip()
                copy_text = " ".join(copy_text.split()[:55])
            elif word_count > 55:
                if attempt < max_attempts - 1:
                    logger.warning("MARKETING_COPY: word_count=%s > 55, retrying...", word_count)
                    retry_tail = (
                        (retry_tail + "\n\n" if retry_tail else "")
                        + f"MUST be 45-55 words (previous attempt was {word_count} words — too long)."
                    )
                    continue
                logger.warning("MARKETING_COPY: word_count=%s > 55, truncating...", word_count)
                if require_verbatim_product_name:
                    copy_text = _repair_verbatim_marketing_product_name(copy_text, canon, lang)
                copy_text = " ".join(copy_text.split()[:55])

            while len(copy_text.split()) < 45:
                copy_text = f"{copy_text} {_marketing_copy_fallback(canon, ad_goal, lang)}".strip()
                copy_text = " ".join(copy_text.split()[:55])

            if len(copy_text.split()) > 55:
                if require_verbatim_product_name:
                    copy_text = _repair_verbatim_marketing_product_name(copy_text, canon, lang)
                copy_text = " ".join(copy_text.split()[:55])

            if require_verbatim_product_name and not product_name_reused_in_copy(canon, copy_text):
                if attempt < max_attempts - 1:
                    logger.warning(
                        "MARKETING_COPY: verbatim product name missing, retrying (attempt %s)",
                        attempt + 1,
                    )
                    retry_tail = (
                        (retry_tail + "\n\n" if retry_tail else "")
                        + "The previous answer did not contain this exact substring: "
                        + json.dumps(canon, ensure_ascii=False)
                        + ". Include it verbatim at least once. Do not use the description as the name."
                    )
                    continue
                copy_text = _repair_verbatim_marketing_product_name(copy_text, canon, lang)
                copy_text = " ".join(copy_text.split()[:55])
                while len(copy_text.split()) < 45:
                    copy_text = f"{copy_text} {_marketing_copy_fallback(canon, ad_goal, lang)}".strip()
                    copy_text = " ".join(copy_text.split()[:55])
                if not product_name_reused_in_copy(canon, copy_text):
                    raise MarketingCopyVerbatimProductNameFailed(
                        "marketing_copy_verbatim_product_name_failed"
                    )

            copy_text = _apply_hebrew_marketing_name_postprocess_with_log(
                copy_text,
                canon,
                ad_goal,
                lang,
                require_verbatim_product_name=require_verbatim_product_name,
            )

            final_word_count = len(copy_text.split())
            logger.info(
                "MARKETING_COPY SUCCESS: word_count=%s copy='%s...'",
                final_word_count,
                copy_text[:100],
            )
            return copy_text

        except openai_retry.OpenAIRateLimitError:
            raise
        except MarketingCopyVerbatimProductNameFailed:
            raise
        except Exception as e:
            logger.error(
                "MARKETING_COPY failed (attempt %s/%s): %s",
                attempt + 1,
                max_attempts,
                e,
            )
            if attempt < max_attempts - 1:
                continue
            logger.warning("MARKETING_COPY: Using fallback copy")
            fb = _marketing_copy_fallback(canon, ad_goal, lang)
            if require_verbatim_product_name:
                fb = _repair_verbatim_marketing_product_name(fb, canon, lang)
                fb = " ".join(fb.split()[:55])
                while len(fb.split()) < 45:
                    fb = f"{fb} {_marketing_copy_fallback(canon, ad_goal, lang)}".strip()
                    fb = " ".join(fb.split()[:55])
                if not product_name_reused_in_copy(canon, fb):
                    raise MarketingCopyVerbatimProductNameFailed(
                        "marketing_copy_verbatim_product_name_failed"
                    ) from e
            fb = _apply_hebrew_marketing_name_postprocess_with_log(
                fb,
                canon,
                ad_goal,
                lang,
                require_verbatim_product_name=require_verbatim_product_name,
            )
            return fb

    fb = _marketing_copy_fallback(canon, ad_goal, lang)
    if require_verbatim_product_name:
        fb = _repair_verbatim_marketing_product_name(fb, canon, lang)
        fb = " ".join(fb.split()[:55])
        while len(fb.split()) < 45:
            fb = f"{fb} {_marketing_copy_fallback(canon, ad_goal, lang)}".strip()
            fb = " ".join(fb.split()[:55])
        if not product_name_reused_in_copy(canon, fb):
            raise MarketingCopyVerbatimProductNameFailed("marketing_copy_verbatim_product_name_failed")
    fb = _apply_hebrew_marketing_name_postprocess_with_log(
        fb,
        canon,
        ad_goal,
        lang,
        require_verbatim_product_name=require_verbatim_product_name,
    )
    return fb

