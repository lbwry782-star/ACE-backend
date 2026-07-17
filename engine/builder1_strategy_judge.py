"""
Builder1 strategy judge — validates campaign plan before image generation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeAlias

from engine.builder1_marketing_copy import (
    MARKETING_TEXT_WORD_COUNT,
    count_marketing_words,
    normalize_campaign_language,
    validate_marketing_text_language,
)
from engine.builder1_plan_parser import _word_count
from engine.builder1_plan_spec import BRAND_SLOGAN_MAX_WORDS, HEADLINE_MAX_WORDS

logger = logging.getLogger(__name__)

JudgeModelCaller: TypeAlias = Callable[[str, str], object]

STALE_MARKETING_LENGTH_CODES = frozenset(
    {
        "marketing_copy_too_long",
        "marketing_copy_too_short",
        "marketing_text_too_long",
        "marketing_text_too_short",
        "marketing_copy_too_long_for_image",
        "marketing_copy_too_long_for_in_image_rendering",
    }
)

MARKETING_WORD_COUNT_CODES = frozenset(
    STALE_MARKETING_LENGTH_CODES
    | {
        "marketing_copy_wrong_word_count",
    }
)

MARKETING_LANGUAGE_CODES = frozenset({"marketing_copy_wrong_language"})

BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT = """
You are a strict advertising strategy auditor for Builder1 campaigns.

The plan you receive is already normalized by the server.
Do NOT reject format, adCount, detectedLanguage, ad indexes, or missing candidate scans.

Return JSON only:
{
  "pass": true,
  "rejectionReasonCodes": [],
  "unsupportedClaimDetected": false,
  "strategicProblemReal": true,
  "relativeAdvantageSupported": true,
  "relativeAdvantageDistinctive": true,
  "conceptualGeneratorIsAction": true,
  "conceptualGeneratorIsNotObject": true,
  "conceptualGeneratorDirectlyExpressesAdvantage": true,
  "conceptualGeneratorSupportsSeries": true,
  "physicalGeneratorEmbodiesConcept": true,
  "physicalGeneratorWasDerivedFromConcept": true,
  "physicalGeneratorDidNotReplaceConcept": true,
  "graphicGeneratorConcrete": true,
  "seriesCoherent": true,
  "sloganDerivesFromAdvantage": true,
  "noUnsupportedEvidence": true
}

IMAGE COPY (rendered inside the generated advertisement):
- brand name, brand slogan, optional headline
- must remain short for layout
- apply brevity rules only to headline and brandSlogan

SUPPORTING MARKETING TEXT BELOW THE IMAGE:
- field name: marketingText on each ad
- must contain exactly 50 words
- exactly 50 words is valid and required
- fewer than 50 words is invalid
- more than 50 words is invalid
- do NOT reject marketingText merely because 50 words feels visually long
- marketingText is displayed below the ad, not inside the image

Evaluate marketingText for relevance, advantage consistency, factual support, grammar,
language match, and absence of internal methodology terms.

The server provides authoritative detectedLanguage (he or en) in the user prompt.
marketingText must be written primarily in detectedLanguage.
Do not reject a predominantly Hebrew paragraph merely because it contains a few Latin brand,
product, URL, number, or technical tokens.
Do not reject a predominantly English paragraph merely because it contains a Hebrew brand name.
Use marketing_copy_wrong_language only when the paragraph is primarily in the wrong language.

Use rejectionReasonCodes such as:
- marketing_copy_wrong_word_count
- marketing_copy_unsupported_claim
- marketing_copy_irrelevant
- marketing_copy_incoherent
- marketing_copy_wrong_language
- unsupported_evidence_claim

Do NOT emit marketing_copy_too_long when marketingText contains exactly 50 words.

Fail if:
- unsupported product capabilities are presented as facts without brief support
- strategicProblemEvidence or relativeAdvantageBriefSupport invent surveys, percentages, study names, or statistics
- relative advantage is generic transparency/quality/trust/results
- conceptual generator is a theme, emotion, object, or equals physicalGenerator
- physical generator appears chosen before the conceptual action
- ads merely swap objects without performing the same conceptual action
- graphic generator lacks concrete renderable fields and recurring visible device
- headline or brandSlogan exceed image-copy brevity limits

Return structured JSON only.
""".strip()


@dataclass
class StrategyJudgeResult:
    passed: bool
    rejection_reason_codes: List[str]
    unsupported_claim_detected: bool = False
    raw: Dict[str, Any] | None = None


def _coerce_judge_dict(raw_payload: object) -> Dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no_json_object")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("judge_output_not_object")
        return obj
    raise ValueError("judge_output_not_object")


def _ads_from_plan(plan_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return []
    return [ad for ad in ads if isinstance(ad, dict)]


def all_marketing_text_exactly_required_count(plan_dict: Dict[str, Any]) -> bool:
    ads = _ads_from_plan(plan_dict)
    if not ads:
        return False
    return all(count_marketing_words(ad.get("marketingText")) == MARKETING_TEXT_WORD_COUNT for ad in ads)


def all_marketing_text_matches_language(plan_dict: Dict[str, Any]) -> bool:
    detected = normalize_campaign_language(plan_dict.get("detectedLanguage"))
    ads = _ads_from_plan(plan_dict)
    if not ads:
        return False
    return all(
        validate_marketing_text_language(ad.get("marketingText"), detected)["valid"] for ad in ads
    )


def deterministic_judge_checks(plan_dict: Dict[str, Any]) -> List[str]:
    """Server-side checks using the same marketing word counter as assembly."""
    reasons: List[str] = []
    detected = normalize_campaign_language(plan_dict.get("detectedLanguage"))
    ads = _ads_from_plan(plan_dict)
    for ad in ads:
        count = count_marketing_words(ad.get("marketingText"))
        if count != MARKETING_TEXT_WORD_COUNT:
            reasons.append("marketing_copy_wrong_word_count")
        lang_result = validate_marketing_text_language(ad.get("marketingText"), detected)
        if not lang_result["valid"]:
            reasons.append("marketing_copy_wrong_language")
        headline = ad.get("headline")
        if headline and _word_count(str(headline)) > HEADLINE_MAX_WORDS:
            reasons.append("headline_too_long")
    brand_slogan = str(plan_dict.get("brandSlogan") or "")
    if brand_slogan and _word_count(brand_slogan) > BRAND_SLOGAN_MAX_WORDS:
        reasons.append("brand_slogan_too_long")
    return list(dict.fromkeys(reasons))


def normalize_judge_rejection_codes(codes: List[str], plan_dict: Dict[str, Any]) -> List[str]:
    all_valid_length = all_marketing_text_exactly_required_count(plan_dict)
    all_valid_language = all_marketing_text_matches_language(plan_dict)
    normalized: List[str] = []
    for raw_code in codes:
        code = str(raw_code or "").strip()
        if not code:
            continue
        lowered = code.lower()
        if code in STALE_MARKETING_LENGTH_CODES or "too_long" in lowered and "marketing" in lowered:
            if all_valid_length:
                continue
            normalized.append("marketing_copy_wrong_word_count")
            continue
        if code == "marketing_copy_wrong_word_count" and all_valid_length:
            continue
        if code == "marketing_copy_wrong_language" and all_valid_language:
            continue
        normalized.append(code)
    return list(dict.fromkeys(normalized))


def finalize_judge_result(
    *,
    model_pass: bool,
    model_codes: List[str],
    plan_dict: Dict[str, Any],
    unsupported: bool,
    raw: Dict[str, Any],
) -> StrategyJudgeResult:
    precheck_codes = deterministic_judge_checks(plan_dict)
    codes = normalize_judge_rejection_codes(model_codes, plan_dict)
    if precheck_codes:
        codes = list(dict.fromkeys(precheck_codes + codes))
        passed = False
    elif not codes:
        passed = True
    else:
        passed = model_pass and not codes
    if not passed and not codes:
        codes = ["strategy_judge_failed"]
    if passed:
        logger.info("BUILDER1_STRATEGY_JUDGE_PASS")
    else:
        logger.error("BUILDER1_STRATEGY_JUDGE_FAIL codes=%s unsupported=%s", codes, unsupported)
    return StrategyJudgeResult(
        passed=passed,
        rejection_reason_codes=codes,
        unsupported_claim_detected=unsupported,
        raw=raw,
    )


def build_strategy_judge_user_prompt(
    *,
    product_description: str,
    plan_dict: Dict[str, Any],
) -> str:
    strip_keys = {
        "strategyCandidateScan",
        "conceptualGeneratorScan",
        "campaignSelfCheck",
        "strategyJudgeResult",
    }
    public_plan = {k: v for k, v in plan_dict.items() if k not in strip_keys}
    detected = normalize_campaign_language(plan_dict.get("detectedLanguage"))
    return (
        f"Brief:\n{product_description.strip()}\n\n"
        f"Authoritative detectedLanguage: {detected}\n\n"
        f"Proposed campaign plan:\n{json.dumps(public_plan, ensure_ascii=False, indent=2)}\n\n"
        "Audit this plan. marketingText must contain exactly 50 words below the image in detectedLanguage. "
        "Do not apply image-copy brevity limits to marketingText. "
        "Allow isolated brand names or technical terms in another script. Return JSON only."
    )


def judge_builder1_strategy(
    *,
    product_description: str,
    plan_dict: Dict[str, Any],
    model_caller: JudgeModelCaller,
) -> StrategyJudgeResult:
    user_prompt = build_strategy_judge_user_prompt(
        product_description=product_description,
        plan_dict=plan_dict,
    )
    try:
        raw_payload = model_caller(BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT, user_prompt)
        data = _coerce_judge_dict(raw_payload)
    except Exception as exc:
        logger.error("BUILDER1_STRATEGY_JUDGE_FAIL stage=call err=%s", exc)
        return StrategyJudgeResult(
            passed=False,
            rejection_reason_codes=["judge_call_failed"],
        )

    model_pass = bool(data.get("pass"))
    model_codes = [str(c) for c in (data.get("rejectionReasonCodes") or []) if str(c).strip()]
    unsupported = bool(data.get("unsupportedClaimDetected"))
    return finalize_judge_result(
        model_pass=model_pass,
        model_codes=model_codes,
        plan_dict=plan_dict,
        unsupported=unsupported,
        raw=data,
    )


def is_marketing_word_count_rejection(codes: List[str]) -> bool:
    return any(code in MARKETING_WORD_COUNT_CODES for code in codes)


def is_marketing_language_rejection(codes: List[str]) -> bool:
    return any(code in MARKETING_LANGUAGE_CODES for code in codes)
