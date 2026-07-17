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
    marketing_text_word_count_error,
    normalize_marketing_text,
    trim_marketing_text_to_50_words_if_usable,
    validate_marketing_text_50_words,
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
- marketingText must be exactly 50 words in the requested language.
- One coherent paragraph only.
- No headings, bullets, labels, hashtags, or wrapping quotation marks.
- Support the ad execution and relative advantage without exposing internal strategy terminology.
- Do not repeat the headline mechanically.
- Do not invent unsupported product capabilities.
""".strip()


def _parse_repair_response(raw_payload: object) -> Dict[int, str]:
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
        parsed[idx] = text
    return parsed


def build_marketing_text_repair_user_prompt(
    *,
    detected_language: str,
    relative_advantage: str,
    product_name: str,
    brand_slogan: str,
    ads_to_repair: List[Dict[str, Any]],
) -> str:
    return (
        f"Language: {detected_language}\n"
        f"Product name: {product_name}\n"
        f"Brand slogan: {brand_slogan}\n"
        f"Relative advantage: {relative_advantage}\n"
        f"Required word count: {MARKETING_TEXT_WORD_COUNT}\n"
        "Repair only the marketingText for these ads:\n"
        f"{json.dumps(ads_to_repair, ensure_ascii=False, indent=2)}\n"
        "Return repairs with corrected 50-word marketingText only."
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
    working = [dict(ad) for ad in ads]
    invalid = _collect_invalid_marketing_indexes(working)
    if not invalid:
        return working

    for attempt in (1, 2):
        repairs = _run_marketing_text_repair(
            working,
            invalid_indexes=invalid,
            detected_language=detected_language,
            relative_advantage=relative_advantage,
            product_name=product_name,
            brand_slogan=brand_slogan,
            model_caller=model_caller,
            attempt=attempt,
        )
        for idx, text in repairs.items():
            for ad in working:
                if int(ad.get("index", -1)) == idx:
                    ad["marketingText"] = text
                    logger.info("BUILDER1_MARKETING_TEXT_OK adIndex=%s wordCount=%s", idx, MARKETING_TEXT_WORD_COUNT)
        invalid = _collect_invalid_marketing_indexes(working)
        if not invalid:
            return working

    for idx in list(invalid):
        ad = next(a for a in working if int(a.get("index", -1)) == idx)
        trimmed = trim_marketing_text_to_50_words_if_usable(ad.get("marketingText"))
        if trimmed:
            ad["marketingText"] = trimmed
            logger.info("BUILDER1_MARKETING_TEXT_OK adIndex=%s wordCount=%s source=trim", idx, MARKETING_TEXT_WORD_COUNT)
    invalid = _collect_invalid_marketing_indexes(working)
    if invalid:
        raise StageParseError("marketing_text", [f"marketing_text_word_count_failed:{','.join(map(str, invalid))}"])
    return working


def _collect_invalid_marketing_indexes(ads: List[Dict[str, Any]]) -> List[int]:
    invalid: List[int] = []
    for ad in ads:
        idx = int(ad.get("index", -1))
        text = ad.get("marketingText")
        count = count_marketing_words(text)
        logger.info("BUILDER1_MARKETING_TEXT_CHECK adIndex=%s wordCount=%s", idx, count)
        if count != MARKETING_TEXT_WORD_COUNT:
            invalid.append(idx)
    return invalid


def _run_marketing_text_repair(
    ads: List[Dict[str, Any]],
    *,
    invalid_indexes: List[int],
    detected_language: str,
    relative_advantage: str,
    product_name: str,
    brand_slogan: str,
    model_caller: MarketingTextModelCaller,
    attempt: int,
) -> Dict[int, str]:
    ads_to_repair = []
    for ad in ads:
        idx = int(ad.get("index", -1))
        if idx not in invalid_indexes:
            continue
        logger.info("BUILDER1_MARKETING_TEXT_REPAIR adIndex=%s attempt=%s", idx, attempt)
        ads_to_repair.append(
            {
                "index": idx,
                "headline": ad.get("headline"),
                "sceneDescription": ad.get("sceneDescription"),
                "conceptualExecution": ad.get("conceptualExecution"),
                "physicalExecution": ad.get("physicalExecution"),
                "visualExecution": ad.get("visualExecution"),
                "newContribution": ad.get("newContribution"),
                "currentMarketingText": ad.get("marketingText"),
                "currentWordCount": count_marketing_words(ad.get("marketingText")),
            }
        )
    user_prompt = build_marketing_text_repair_user_prompt(
        detected_language=detected_language,
        relative_advantage=relative_advantage,
        product_name=product_name,
        brand_slogan=brand_slogan,
        ads_to_repair=ads_to_repair,
    )
    raw = model_caller(MARKETING_TEXT_REPAIR_SYSTEM, user_prompt)
    repairs = _parse_repair_response(raw)
    for idx in invalid_indexes:
        if idx not in repairs:
            raise StageParseError("marketing_text_repair", [f"marketing_text_repair_missing_index:{idx}"])
    return repairs
