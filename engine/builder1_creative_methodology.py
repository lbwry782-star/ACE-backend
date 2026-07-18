"""
Builder1 creative methodology — deterministic planning and judge checks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.builder1_no_logo import deterministic_no_logo_checks
from engine.builder1_slogan_stage import SLOGAN_REJECTION_CODES, is_slogan_rejection

METHODOLOGY_REJECTION_CODES = frozenset(
    SLOGAN_REJECTION_CODES
    | {
        "conceptual_generator_not_derived_from_slogan",
        "physical_generator_not_derived_from_concept",
        "physical_generator_is_product",
        "physical_generator_is_packaging",
        "unauthorized_product_visibility",
        "visual_requires_explanatory_headline",
        "campaign_transferable_to_competitor",
        "category_relevance_patched",
        "series_lacks_shared_visual_law",
        "graphic_generator_inconsistent",
        "hebrew_composition_rule_broken",
        "no_mechanism_reuse_inside_campaign",
        "same_image_different_headlines",
    }
)

_RESCUE_HEADLINE_PATTERNS: Tuple[str, ...] = (
    r"\b(this is|that is|because|means that|shows that|explains|why we|what happened)\b",
    r"\b(the joke|the visual|the image|the object)\b",
    r"\b(look at|see how|notice how)\b",
)

_TOY_MINIATURIZATION_PATTERNS: Tuple[str, ...] = (
    r"\b(miniature|mini version|tiny copy|toy[- ]sized|child[- ]sized|small duplicate)\b",
)

_PRIOR_CAMPAIGN_OBJECTS: Tuple[str, ...] = ()

_EXPLICIT_CROSS_CAMPAIGN_REUSE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\breuse the previous campaign\b", "no_mechanism_reuse_inside_campaign"),
    (r"\bsame mechanism as the last brand\b", "no_mechanism_reuse_inside_campaign"),
    (r"\bcopy the earlier visual\b", "no_mechanism_reuse_inside_campaign"),
    (r"\breuse the last campaign\b", "no_mechanism_reuse_inside_campaign"),
    (r"\bfrom the previous brand campaign\b", "no_mechanism_reuse_inside_campaign"),
)

FOUNDATIONAL_STRATEGIC_REJECTION_CODES = frozenset(
    {
        "campaign_transferable_to_competitor",
        "category_relevance_patched",
        "advantage_not_currently_true",
        "relative_advantage_not_currently_true",
        "strategy_not_brand_ownable",
        "business_transformation_required",
        "client_consultation_required",
        "material_client_investment_required",
        "unsupported_future_capability",
    }
)

INTERNAL_AD_FIELDS = (
    "familiarExpectation",
    "singleChangedPropertyOrAction",
    "immediateClarityReason",
    "sloganConnection",
    "relativeAdvantageConnection",
    "brandOwnershipReason",
    "categoryRelevanceReason",
    "headlineRequired",
    "headlineReason",
    "productVisible",
    "packagingVisible",
    "productIsMainVisual",
    "productIsPhysicalGenerator",
    "sameVisualLawProof",
    "distinctFromOtherAdsReason",
    "noReuseCheck",
    "competitorTransferTest",
    "transferRisk",
)


def is_foundational_strategic_rejection(codes: List[str]) -> bool:
    return any(code in FOUNDATIONAL_STRATEGIC_REJECTION_CODES for code in codes)


def is_methodology_rejection(codes: List[str]) -> bool:
    return any(code in METHODOLOGY_REJECTION_CODES for code in codes)


def methodology_repair_stage(codes: List[str]) -> Optional[str]:
    """Map methodology rejection codes to restart/repair stage."""
    unique = list(dict.fromkeys(codes))
    if is_slogan_rejection(unique) and not is_foundational_strategic_rejection(unique):
        return "slogan_scan"
    if any(
        code in unique
        for code in (
            "conceptual_generator_not_derived_from_slogan",
            "campaign_transferable_to_competitor",
            "category_relevance_patched",
        )
    ):
        if "campaign_transferable_to_competitor" in unique or "category_relevance_patched" in unique:
            return "strategy_scan"
        return "conceptual_scan"
    if any(
        code in codes
        for code in (
            "physical_generator_not_derived_from_concept",
            "physical_generator_is_product",
            "physical_generator_is_packaging",
            "unauthorized_product_visibility",
        )
    ):
        return "brand_physical"
    if any(
        code in codes
        for code in (
            "visual_requires_explanatory_headline",
            "series_lacks_shared_visual_law",
            "same_image_different_headlines",
            "no_mechanism_reuse_inside_campaign",
        )
    ):
        return "series_ads"
    if "graphic_generator_inconsistent" in codes or "hebrew_composition_rule_broken" in codes:
        return "graphic_system"
    return None


def earliest_methodology_repair_stage(codes: List[str]) -> Optional[str]:
    unique = list(dict.fromkeys(codes))
    stage_checks = (
        ("strategy_scan", FOUNDATIONAL_STRATEGIC_REJECTION_CODES),
        ("slogan_scan", SLOGAN_REJECTION_CODES),
        (
            "conceptual_scan",
            frozenset(
                {
                    "conceptual_generator_not_derived_from_slogan",
                    "campaign_transferable_to_competitor",
                    "category_relevance_patched",
                }
            ),
        ),
        (
            "brand_physical",
            frozenset(
                {
                    "physical_generator_not_derived_from_concept",
                    "physical_generator_is_product",
                    "physical_generator_is_packaging",
                    "unauthorized_product_visibility",
                }
            ),
        ),
        (
            "graphic_system",
            frozenset(
                {
                    "graphic_generator_inconsistent",
                    "hebrew_composition_rule_broken",
                }
            ),
        ),
        (
            "series_ads",
            frozenset(
                {
                    "visual_requires_explanatory_headline",
                    "series_lacks_shared_visual_law",
                    "same_image_different_headlines",
                    "no_mechanism_reuse_inside_campaign",
                }
            ),
        ),
    )
    for stage, stage_codes in stage_checks:
        if any(code in unique for code in stage_codes):
            if stage == "strategy_scan" and is_slogan_rejection(unique) and not is_foundational_strategic_rejection(unique):
                continue
            return stage
    return methodology_repair_stage(unique)


def scan_prompt_for_reused_mechanisms(text: str) -> Optional[str]:
    lowered = text.lower()
    for pattern, code in _EXPLICIT_CROSS_CAMPAIGN_REUSE_PATTERNS:
        if re.search(pattern, lowered, re.I):
            return code
    return None


def _ads_from_plan(plan_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    ads = plan_dict.get("ads")
    if not isinstance(ads, list):
        return []
    return [ad for ad in ads if isinstance(ad, dict)]


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _headline_is_rescue(headline: str, scene: str) -> bool:
    if not headline:
        return False
    lowered = headline.lower()
    for pattern in _RESCUE_HEADLINE_PATTERNS:
        if re.search(pattern, lowered, re.I):
            return True
    scene_words = set(re.findall(r"[a-zA-Z\u0590-\u05FF]{4,}", scene.lower()))
    headline_words = set(re.findall(r"[a-zA-Z\u0590-\u05FF]{4,}", lowered))
    if scene_words and headline_words and headline_words.issubset(scene_words):
        return True
    return False


def deterministic_methodology_checks(plan_dict: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    reasons.extend(deterministic_no_logo_checks(plan_dict))

    slogan = _norm(plan_dict.get("brandSlogan"))
    slogan_action = _norm(plan_dict.get("sloganAction"))
    relative_advantage = _norm(plan_dict.get("relativeAdvantage"))
    conceptual_action = _norm(plan_dict.get("conceptualGeneratorAction"))
    physical = _norm(plan_dict.get("physicalGenerator"))
    campaign_rationale = _norm(plan_dict.get("campaignRationale"))
    detected = _norm(plan_dict.get("detectedLanguage")).lower()
    embodiment = _norm(plan_dict.get("embodimentChoice")).lower()
    visibility_reason = _norm(plan_dict.get("productVisibilityJustification"))
    visibility_policy = _norm(plan_dict.get("productVisibilityPolicy")).upper() or "FORBIDDEN"

    if slogan_action and conceptual_action:
        why_slogan = _norm(plan_dict.get("conceptualGeneratorWhyItExpressesSlogan"))
        if not why_slogan:
            action_tokens = set(re.findall(r"[a-zA-Z\u0590-\u05FF]{4,}", slogan_action.lower()))
            concept_tokens = set(re.findall(r"[a-zA-Z\u0590-\u05FF]{4,}", conceptual_action.lower()))
            if action_tokens and concept_tokens and not (action_tokens & concept_tokens):
                reasons.append("conceptual_generator_not_derived_from_slogan")

    reasons.extend(
        _deterministic_methodology_checks_without_semantic_concept_derivation(
            plan_dict,
            slogan=slogan,
            slogan_action=slogan_action,
            relative_advantage=relative_advantage,
            conceptual_action=conceptual_action,
            physical=physical,
            campaign_rationale=campaign_rationale,
            detected=detected,
            embodiment=embodiment,
            visibility_reason=visibility_reason,
            visibility_policy=visibility_policy,
        )
    )
    return list(dict.fromkeys(reasons))


def _deterministic_methodology_checks_without_semantic_concept_derivation(
    plan_dict: Dict[str, Any],
    *,
    slogan: str = "",
    slogan_action: str = "",
    relative_advantage: str = "",
    conceptual_action: str = "",
    physical: str = "",
    campaign_rationale: str = "",
    detected: str = "",
    embodiment: str = "",
    visibility_reason: str = "",
    visibility_policy: str = "FORBIDDEN",
) -> List[str]:
    reasons: List[str] = []
    if not any([slogan, slogan_action, relative_advantage, conceptual_action, physical]):
        slogan = _norm(plan_dict.get("brandSlogan"))
        slogan_action = _norm(plan_dict.get("sloganAction"))
        relative_advantage = _norm(plan_dict.get("relativeAdvantage"))
        conceptual_action = _norm(plan_dict.get("conceptualGeneratorAction"))
        physical = _norm(plan_dict.get("physicalGenerator"))
        campaign_rationale = _norm(plan_dict.get("campaignRationale"))
        detected = _norm(plan_dict.get("detectedLanguage")).lower()
        embodiment = _norm(plan_dict.get("embodimentChoice")).lower()
        visibility_reason = _norm(plan_dict.get("productVisibilityJustification"))
        visibility_policy = _norm(plan_dict.get("productVisibilityPolicy")).upper() or "FORBIDDEN"

    if physical and conceptual_action:
        physical_role = _norm(plan_dict.get("physicalGeneratorCampaignRole"))
        physical_purpose = _norm(plan_dict.get("physicalGeneratorNaturalPurpose"))
        combined = f"{physical} {physical_role} {physical_purpose} {campaign_rationale}".lower()
        disconnected_markers = (
            "unrelated object",
            "generic product shot",
            "default product",
            "random object",
            "semantically unrelated",
        )
        if any(marker in combined for marker in disconnected_markers):
            reasons.append("physical_generator_not_derived_from_concept")

    if visibility_policy == "FORBIDDEN":
        for ad in _ads_from_plan(plan_dict):
            if ad.get("productVisible") is True or ad.get("productIsMainVisual") is True:
                reasons.append("unauthorized_product_visibility")
            if ad.get("packagingVisible") is True:
                reasons.append("unauthorized_product_visibility")
            if ad.get("productIsPhysicalGenerator") is True:
                reasons.append("physical_generator_is_product")
        if embodiment == "literal":
            reasons.append("unauthorized_product_visibility")
        if visibility_reason and "show" in visibility_reason.lower():
            reasons.append("unauthorized_product_visibility")

    transfer = _norm(plan_dict.get("competitorTransferTest")).lower()
    if transfer in {"yes", "true", "transferable", "high"}:
        reasons.append("campaign_transferable_to_competitor")
    transfer_risk = _norm(plan_dict.get("transferRisk")).lower()
    if transfer_risk == "high":
        reasons.append("campaign_transferable_to_competitor")

    category_patch = _norm(plan_dict.get("categoryRelevancePatched")).lower()
    if category_patch in {"yes", "true"}:
        reasons.append("category_relevance_patched")

    graphic = plan_dict.get("graphicGenerator")
    if isinstance(graphic, dict):
        placements = {str(ad.get("index")): _norm(ad.get("sceneDescription")).lower() for ad in _ads_from_plan(plan_dict)}
        if detected == "he":
            placement = _norm(graphic.get("sloganPlacement")).lower()
            reason = _norm(graphic.get("sloganPlacementReason"))
            if placement and placement != "bottom_left" and not reason:
                reasons.append("hebrew_composition_rule_broken")

    ads = _ads_from_plan(plan_dict)
    scene_signatures: Set[str] = set()
    for ad in ads:
        scene = _norm(ad.get("sceneDescription"))
        headline = _norm(ad.get("headline"))
        if headline and _headline_is_rescue(headline, scene):
            reasons.append("visual_requires_explanatory_headline")
        signature = re.sub(r"\s+", " ", scene.lower())
        if signature in scene_signatures and headline:
            reasons.append("same_image_different_headlines")
        scene_signatures.add(signature)

        physical_exec = _norm(ad.get("physicalExecution")).lower()
        for pattern in _TOY_MINIATURIZATION_PATTERNS:
            if re.search(pattern, physical_exec):
                changed = _norm(ad.get("singleChangedPropertyOrAction")).lower()
                if changed and "miniatur" in changed:
                    reasons.append("series_lacks_shared_visual_law")
                break

        for field in ("sceneDescription", "physicalExecution", "visualExecution", "campaignRationale"):
            code = scan_prompt_for_reused_mechanisms(_norm(ad.get(field) if field != "campaignRationale" else campaign_rationale))
            if code:
                reasons.append(code)

        no_reuse = _norm(ad.get("noReuseCheck")).lower()
        if no_reuse in {"duplicate", "same", "reused"}:
            reasons.append("no_mechanism_reuse_inside_campaign")

    return list(dict.fromkeys(reasons))


def deterministic_builder1_integrity_checks(plan_dict: Dict[str, Any]) -> List[str]:
    """Objective campaign invariants for post-series validation — no creative semantic judging."""
    reasons: List[str] = []
    reasons.extend(deterministic_no_logo_checks(plan_dict))
    reasons.extend(_deterministic_methodology_checks_without_semantic_concept_derivation(plan_dict))
    return list(dict.fromkeys(reasons))


def strip_internal_ad_fields(ad: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in ad.items() if key not in INTERNAL_AD_FIELDS and key not in {"competitorTransferTest", "transferRisk"}}


def strip_internal_plan_fields(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(plan_dict)
    for key in (
        "embodimentChoice",
        "productVisibilityJustification",
        "productVisibilityPolicy",
        "productVisibilitySource",
        "transferredObject",
        "transferredObjectAction",
        "whyClearerThanShowingProduct",
        "competitorTransferTest",
        "transferRisk",
        "categoryRelevancePatched",
        "conceptualGeneratorWhyItExpressesSlogan",
        "brandOwnershipReason",
        "sloganPlacementReason",
    ):
        cleaned.pop(key, None)
    ads = cleaned.get("ads")
    if isinstance(ads, list):
        cleaned["ads"] = [strip_internal_ad_fields(ad) if isinstance(ad, dict) else ad for ad in ads]
    return cleaned
