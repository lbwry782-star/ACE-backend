"""
Builder1 focused marketing-text repair after series_ads structural validation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, TypeAlias

from engine.builder1_marketing_copy import (
    MARKETING_TEXT_WORD_COUNT,
    count_marketing_words,
    marketing_text_language_error,
    marketing_text_word_count_error,
    normalize_campaign_language,
    normalize_marketing_text,
    trim_marketing_text_to_50_words_if_usable,
    validate_marketing_text_language,
)
from engine.builder1_staged_parsers import StageParseError, coerce_json_dict

logger = logging.getLogger(__name__)

MarketingTextModelCaller: TypeAlias = Callable[[str, str], object]

MARKETING_TEXT_REPAIR_SYSTEM = """
You are a Builder1 marketing copy repair assistant.
Return JSON only. Return exactly this object and no additional top-level keys:
{"repairs":[{"index":1,"marketingText":"..."}]}
Rules:
- Each repair entry must contain only index and marketingText.
- marketingText must be exactly 50 words in the authoritative target language.
- Write the paragraph primarily in the target language.
- Brand names, product names, URLs, numbers, and technical terms may remain in Latin letters.
- Do not switch into another language for the body of the paragraph.
- One coherent paragraph only.
- No headings, bullets, labels, hashtags, or wrapping quotation marks.
- Support the ad execution and relative advantage without exposing internal strategy terminology.
- Do not repeat the headline mechanically.
- Do not invent unsupported product capabilities.
""".strip()


def _parse_repair_response(raw_payload: object, *, detected_language: str) -> Dict[int, str]:
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("marketing_text_repair", ["marketing_text_repair_not_object"]) from exc

    repairs_raw = obj.get("repairs")
    if not isinstance(repairs_raw, list):
        raise StageParseError("marketing_text_repair", ["marketing_text_repair_missing_repairs"])

    parsed: Dict[int, str] = {}
    for item in repairs_raw:
        if not isinstance(item, dict):
            raise StageParseError("marketing_text_repair", ["marketing_text_repair_entry_not_object"])
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError) as exc:
            raise StageParseError("marketing_text_repair", ["marketing_text_repair_invalid_index"]) from exc
        text = normalize_marketing_text(item.get("marketingText"))
        if not text:
            raise StageParseError("marketing_text_repair", ["marketing_text_repair_empty_text"])
        err = marketing_text_word_count_error(text)
        if err:
            raise StageParseError("marketing_text_repair", [err])
        lang_err = marketing_text_language_error(text, detected_language)
        if lang_err:
            raise StageParseError("marketing_text_repair", [lang_err])
        parsed[idx] = text
    return parsed


def _language_display_name(detected_language: str) -> str:
    return "Hebrew" if normalize_campaign_language(detected_language) == "he" else "English"


def build_marketing_text_repair_user_prompt(
    *,
    detected_language: str,
    relative_advantage: str,
    product_name: str,
    ads_to_repair: List[Dict[str, Any]],
) -> str:
    target = normalize_campaign_language(detected_language)
    return (
        f"TARGET LANGUAGE FOR ALL MARKETING TEXT: {_language_display_name(target)} ({target})\n"
        f"Product name: {product_name}\n"
        f"Relative advantage: {relative_advantage}\n"
        f"Required word count: {MARKETING_TEXT_WORD_COUNT}\n"
        "Rewrite only marketingText for these ads. Do not change any other ad field.\n"
        f"{json.dumps(ads_to_repair, ensure_ascii=False, indent=2)}\n"
        "Return repairs with corrected 50-word marketingText in the target language only."
    )


def ensure_series_ads_marketing_text(
    ads: List[Dict[str, Any]],
    *,
    detected_language: str,
    relative_advantage: str,
    product_name: str,
    brand_slogan: str,
    model_caller: MarketingTextModelCaller,
) -> List[Dict[str, Any]]:
    """Validate and, if needed, repair marketingText without rerunning series_ads."""
    _ = brand_slogan
    working = [dict(ad) for ad in ads]
    invalid = _collect_invalid_marketing_indexes(working, detected_language=detected_language)
    if not invalid:
        return working

    for attempt in (1, 2):
        repairs = _run_marketing_text_repair(
            working,
            invalid_indexes=invalid,
            detected_language=detected_language,
            relative_advantage=relative_advantage,
            product_name=product_name,
            model_caller=model_caller,
            attempt=attempt,
        )
        for idx, text in repairs.items():
            for ad in working:
                if int(ad.get("index", -1)) == idx:
                    ad["marketingText"] = text
                    logger.info(
                        "BUILDER1_MARKETING_TEXT_OK adIndex=%s wordCount=%s",
                        idx,
                        MARKETING_TEXT_WORD_COUNT,
                    )
                    logger.info(
                        "BUILDER1_MARKETING_LANGUAGE_OK adIndex=%s targetLanguage=%s",
                        idx,
                        normalize_campaign_language(detected_language),
                    )
        invalid = _collect_invalid_marketing_indexes(working, detected_language=detected_language)
        if not invalid:
            return working

    for idx in list(invalid):
        ad = next(a for a in working if int(a.get("index", -1)) == idx)
        trimmed = trim_marketing_text_to_50_words_if_usable(ad.get("marketingText"))
        if trimmed and not marketing_text_word_count_error(trimmed):
            if not marketing_text_language_error(trimmed, detected_language):
                ad["marketingText"] = trimmed
                logger.info(
                    "BUILDER1_MARKETING_TEXT_OK adIndex=%s wordCount=%s source=trim",
                    idx,
                    MARKETING_TEXT_WORD_COUNT,
                )
    invalid = _collect_invalid_marketing_indexes(working, detected_language=detected_language)
    if invalid:
        raise StageParseError(
            "marketing_text",
            [f"marketing_text_repair_failed:{','.join(map(str, invalid))}"],
        )
    return working


def _log_marketing_language_check(*, ad_index: int, detected_language: str, text: object) -> None:
    result = validate_marketing_text_language(text, detected_language)
    logger.info(
        "BUILDER1_MARKETING_LANGUAGE_CHECK adIndex=%s targetLanguage=%s dominantLanguage=%s "
        "valid=%s hebrewChars=%s latinChars=%s",
        ad_index,
        result["targetLanguage"],
        result["dominantLanguage"],
        result["valid"],
        result["hebrewCharacterCount"],
        result["latinCharacterCount"],
    )


def _collect_invalid_marketing_indexes(
    ads: List[Dict[str, Any]],
    *,
    detected_language: str,
) -> List[int]:
    invalid: List[int] = []
    for ad in ads:
        idx = int(ad.get("index", -1))
        text = ad.get("marketingText")
        count = count_marketing_words(text)
        logger.info("BUILDER1_MARKETING_TEXT_CHECK adIndex=%s wordCount=%s", idx, count)
        _log_marketing_language_check(ad_index=idx, detected_language=detected_language, text=text)
        if count != MARKETING_TEXT_WORD_COUNT or marketing_text_language_error(text, detected_language):
            invalid.append(idx)
    return list(dict.fromkeys(invalid))


def _run_marketing_text_repair(
    ads: List[Dict[str, Any]],
    *,
    invalid_indexes: List[int],
    detected_language: str,
    relative_advantage: str,
    product_name: str,
    model_caller: MarketingTextModelCaller,
    attempt: int,
) -> Dict[int, str]:
    ads_to_repair = []
    for ad in ads:
        idx = int(ad.get("index", -1))
        if idx not in invalid_indexes:
            continue
        logger.info("BUILDER1_MARKETING_TEXT_REPAIR adIndex=%s attempt=%s", idx, attempt)
        logger.info(
            "BUILDER1_MARKETING_LANGUAGE_REPAIR adIndex=%s targetLanguage=%s",
            idx,
            normalize_campaign_language(detected_language),
        )
        ads_to_repair.append(
            {
                "index": idx,
                "headline": ad.get("headline"),
                "conceptualExecution": ad.get("conceptualExecution"),
                "physicalExecution": ad.get("physicalExecution"),
                "visualExecution": ad.get("visualExecution"),
                "newContribution": ad.get("newContribution"),
                "existingMarketingText": ad.get("marketingText"),
            }
        )
    user_prompt = build_marketing_text_repair_user_prompt(
        detected_language=detected_language,
        relative_advantage=relative_advantage,
        product_name=product_name,
        ads_to_repair=ads_to_repair,
    )
    raw = model_caller(MARKETING_TEXT_REPAIR_SYSTEM, user_prompt)
    repairs = _parse_repair_response(raw, detected_language=detected_language)
    for idx in invalid_indexes:
        if idx not in repairs:
            raise StageParseError("marketing_text_repair", [f"marketing_text_repair_missing_index:{idx}"])
    return repairs
