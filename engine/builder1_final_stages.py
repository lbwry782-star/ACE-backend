"""
Builder1 final campaign substages (5A brand/physical, 5B graphic, 5C series/ads).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from engine.builder1_consolidated_stages import build_conceptual_lineage
from engine.builder1_plan_parser import (
    _norm_text,
    _parse_graphic_generator,
    _parse_series_generator,
    _reject_legacy_fields,
    _word_count,
    check_unsupported_evidence,
    validate_series_plan_structure,
)
from engine.builder1_plan_spec import Builder1GraphicGenerator, Builder1SeriesPlan
from engine.builder1_client_boundary import (
    validate_brand_physical_boundary_text,
    validate_series_ads_boundary_text,
)
from engine.builder1_staged_parsers import (
    ConceptualCandidate,
    StageParseError,
    StrategyCandidate,
    StrategySelection,
    coerce_json_dict,
)
from engine.builder1_slogan_stage import SloganCandidate

logger = logging.getLogger(__name__)

BRAND_PHYSICAL_FORBIDDEN = {
    "strategyCandidateScan",
    "conceptualGeneratorScan",
    "graphicGenerator",
    "seriesGenerator",
    "ads",
    "format",
    "adCount",
    "detectedLanguage",
    "strategicProblem",
    "relativeAdvantage",
    "conceptualGenerator",
    "palette",
    "layoutTemplate",
}

GRAPHIC_FORBIDDEN = {
    "strategyCandidateScan",
    "conceptualGeneratorScan",
    "ads",
    "seriesGenerator",
    "brandSlogan",
    "physicalGenerator",
    "format",
    "adCount",
}

SERIES_ADS_FORBIDDEN = {
    "strategyCandidateScan",
    "conceptualGeneratorScan",
    "graphicGenerator",
    "brandSlogan",
    "physicalGenerator",
    "format",
    "adCount",
    "detectedLanguage",
}

_SNAKE_TO_CAMEL = {
    "layout_template": "layoutTemplate",
    "headline_placement": "headlinePlacement",
    "headline_alignment": "headlineAlignment",
    "headline_max_width_percent": "headlineMaxWidthPercent",
    "brand_block_placement": "brandBlockPlacement",
    "slogan_placement": "sloganPlacement",
    "copy_safe_area": "copySafeArea",
    "typography_style": "typographyStyle",
    "headline_scale": "headlineScale",
    "brand_scale": "brandScale",
    "slogan_scale": "sloganScale",
    "image_style": "imageStyle",
    "background_treatment": "backgroundTreatment",
    "border_treatment": "borderTreatment",
    "recurring_graphic_device": "recurringGraphicDevice",
    "recurring_graphic_device_rule": "recurringGraphicDeviceRule",
    "shape_language": "shapeLanguage",
    "framing_rule": "framingRule",
    "spacing_rule": "spacingRule",
}


@dataclass
class BrandPhysicalOutput:
    product_name_resolved: str
    physical_generator: str
    physical_generator_natural_purpose: str
    physical_generator_campaign_role: str
    physical_generator_is_product: bool
    physical_generator_is_packaging: bool
    works_without_product_visible: bool
    transferred_object: str
    transferred_object_action: str
    why_clearer_than_showing_product: str
    medium_participates: bool
    medium_role: str
    campaign_rationale: str


@dataclass
class SeriesAdsOutput:
    series_generator: Dict[str, str]
    ads: List[Dict[str, Any]]


_PER_AD_MODEL_SLOGAN_KEYS = ("brandSlogan", "slogan", "campaignSlogan")

_AD_INTERNAL_FIELD_KEYS = frozenset(
    {
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
        "brandSlogan",
    }
)


def strip_model_slogan_fields_from_series_ads(
    ads: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove any model-generated per-ad slogan fields before server assembly."""
    cleaned: List[Dict[str, Any]] = []
    for ad in ads:
        if not isinstance(ad, dict):
            cleaned.append(ad)
            continue
        ad_copy = dict(ad)
        for key in _PER_AD_MODEL_SLOGAN_KEYS:
            ad_copy.pop(key, None)
        cleaned.append(ad_copy)
    return cleaned


def inject_fixed_campaign_slogan_into_series_ads(
    ads: List[Dict[str, Any]],
    *,
    fixed_slogan: str,
) -> List[Dict[str, Any]]:
    """Return ad dicts with deprecated model slogan fields removed for assembly."""
    return strip_model_slogan_fields_from_series_ads(ads)


def build_series_ad_internals(
    ads: List[Dict[str, Any]],
    *,
    fixed_slogan: str,
) -> Dict[int, Dict[str, Any]]:
    """Build per-ad internals with the server-owned fixed campaign slogan."""
    ad_internals: Dict[int, Dict[str, Any]] = {}
    for ad in ads:
        if not isinstance(ad, dict) or ad.get("index") is None:
            continue
        idx = int(ad["index"])
        ad_internals[idx] = {
            key: ad.get(key)
            for key in ad
            if key in _AD_INTERNAL_FIELD_KEYS
        }
        ad_internals[idx]["brandSlogan"] = fixed_slogan
    return ad_internals


def log_stage_parse_failure(stage: str, raw_payload: object, reasons: List[str]) -> None:
    keys: List[str] = []
    types: Dict[str, str] = {}
    try:
        obj = coerce_json_dict(raw_payload)
        keys = sorted(str(k) for k in obj.keys())
        for k, v in obj.items():
            types[str(k)] = type(v).__name__
    except Exception:
        keys = []
        types = {"_root": type(raw_payload).__name__}
    logger.error(
        "BUILDER1_STAGE_FAILED stage=%s reasons=%s top_level_keys=%s field_types=%s",
        stage,
        reasons,
        keys,
        types,
    )


def _normalize_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if value is None:
        return default
    raise ValueError("not_boolean")


def _reject_forbidden_keys(obj: Dict[str, Any], forbidden: set[str], reasons: List[str], prefix: str) -> None:
    for key in forbidden:
        if key in obj:
            reasons.append(f"{prefix}_forbidden_field:{key}")


def parse_brand_physical_output(
    raw_payload: object,
    *,
    product_description: str = "",
) -> BrandPhysicalOutput:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("brand_physical", ["brand_physical_not_object"]) from exc

    _reject_forbidden_keys(obj, BRAND_PHYSICAL_FORBIDDEN, reasons, "brand_physical")
    _reject_legacy_fields(obj, reasons)

    forbidden_legacy = (
        "embodimentChoice",
        "productVisibilityJustification",
        "productVisibilityRequired",
        "productVisibilityReason",
    )
    for key in forbidden_legacy:
        if key in obj:
            reasons.append(f"brand_physical_forbidden_field:{key}")

    required = [
        ("productNameResolved", "missing_productNameResolved"),
        ("physicalGenerator", "missing_physicalGenerator"),
        ("physicalGeneratorNaturalPurpose", "missing_physicalGeneratorNaturalPurpose"),
        ("physicalGeneratorCampaignRole", "missing_physicalGeneratorCampaignRole"),
        ("transferredObject", "missing_transferredObject"),
        ("transferredObjectAction", "missing_transferredObjectAction"),
        ("whyClearerThanShowingProduct", "missing_whyClearerThanShowingProduct"),
        ("campaignRationale", "missing_campaignRationale"),
    ]
    for field, code in required:
        if not _norm_text(obj.get(field)):
            reasons.append(code)

    if obj.get("brandSlogan") or obj.get("sloganDerivation") or obj.get("sloganAction"):
        reasons.append("brand_physical_must_not_create_slogan")

    try:
        physical_generator_is_product = bool(obj.get("physicalGeneratorIsProduct"))
        physical_generator_is_packaging = bool(obj.get("physicalGeneratorIsPackaging"))
        works_without_product_visible = bool(obj.get("worksWithoutProductVisible"))
    except (TypeError, ValueError):
        reasons.append("brand_physical_invariant_not_boolean")
        physical_generator_is_product = True
        physical_generator_is_packaging = True
        works_without_product_visible = False

    if physical_generator_is_product:
        reasons.append("physical_generator_is_product")
    if physical_generator_is_packaging:
        reasons.append("physical_generator_is_packaging")
    if not works_without_product_visible:
        reasons.append("physical_generator_requires_product_visible")

    try:
        medium_participates = _normalize_bool(obj.get("mediumParticipates"), default=False)
    except ValueError:
        reasons.append("medium_participates_not_boolean")
        medium_participates = False

    medium_role = _norm_text(obj.get("mediumRole"))
    if not medium_participates:
        medium_role = ""
    elif not medium_role:
        reasons.append("medium_role_required_when_participates")
    if medium_participates is False and _norm_text(obj.get("mediumRole")):
        medium_role = ""

    campaign_rationale = _norm_text(obj.get("campaignRationale"))
    physical_campaign_role = _norm_text(obj.get("physicalGeneratorCampaignRole"))
    reasons.extend(
        validate_brand_physical_boundary_text(
            brand_slogan="",
            slogan_action="",
            campaign_rationale=campaign_rationale,
            physical_generator_campaign_role=physical_campaign_role,
            product_description=product_description,
        )
    )

    if reasons:
        log_stage_parse_failure("brand_physical", obj, reasons)
        raise StageParseError("brand_physical", reasons)

    return BrandPhysicalOutput(
        product_name_resolved=_norm_text(obj.get("productNameResolved")),
        physical_generator=_norm_text(obj.get("physicalGenerator")),
        physical_generator_natural_purpose=_norm_text(obj.get("physicalGeneratorNaturalPurpose")),
        physical_generator_campaign_role=physical_campaign_role,
        physical_generator_is_product=physical_generator_is_product,
        physical_generator_is_packaging=physical_generator_is_packaging,
        works_without_product_visible=works_without_product_visible,
        transferred_object=_norm_text(obj.get("transferredObject")),
        transferred_object_action=_norm_text(obj.get("transferredObjectAction")),
        why_clearer_than_showing_product=_norm_text(obj.get("whyClearerThanShowingProduct")),
        medium_participates=medium_participates,
        medium_role=medium_role,
        campaign_rationale=campaign_rationale,
    )


def _normalize_graphic_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj)
    for snake, camel in _SNAKE_TO_CAMEL.items():
        if snake in out and camel not in out:
            out[camel] = out.pop(snake)
    if "palette" in out and isinstance(out["palette"], dict):
        palette = dict(out["palette"])
        for snake, camel in (
            ("dominant", "dominant"),
            ("secondary", "secondary"),
            ("accent", "accent"),
            ("background", "background"),
            ("text", "text"),
        ):
            if snake in palette:
                palette[camel] = palette[snake]
        out["palette"] = palette
    if "copySafeArea" not in out and "copy_safe_area" in out:
        out["copySafeArea"] = out.pop("copy_safe_area")
    return out


def _unwrap_graphic_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "graphicGenerator" in obj and isinstance(obj.get("graphicGenerator"), dict):
        return _normalize_graphic_keys(obj["graphicGenerator"])
    return _normalize_graphic_keys(obj)


def parse_graphic_system_output(raw_payload: object) -> Builder1GraphicGenerator:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("graphic_system", ["graphic_system_not_object"]) from exc

    _reject_forbidden_keys(obj, GRAPHIC_FORBIDDEN, reasons, "graphic_system")
    graphic_raw = _unwrap_graphic_payload(obj)
    graphic = _parse_graphic_generator(graphic_raw, reasons)

    if reasons or graphic is None:
        log_stage_parse_failure("graphic_system", obj, reasons or ["graphic_generator_invalid"])
        raise StageParseError("graphic_system", reasons or ["graphic_generator_invalid"])
    return graphic


def _normalize_ad_indexes(ads_raw: List[Any], expected_ad_count: int, reasons: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(ads_raw, list):
        reasons.append("ads_not_list")
        return []

    if len(ads_raw) != expected_ad_count:
        reasons.append("ads_length_mismatch")
        return []

    normalized: List[Dict[str, Any]] = []
    for pos, item in enumerate(ads_raw, start=1):
        if not isinstance(item, dict):
            reasons.append("ad_not_object")
            continue
        ad = dict(item)
        raw_index = ad.get("index")
        if raw_index is None:
            logger.info("BUILDER1_STAGE_NORMALIZE stage=series_ads field=index action=inject position=%s", pos)
            ad["index"] = pos
        else:
            try:
                parsed_index = int(raw_index)
            except (TypeError, ValueError):
                logger.info(
                    "BUILDER1_STAGE_NORMALIZE stage=series_ads field=index action=coerce_string position=%s",
                    pos,
                )
                parsed_index = pos
            if parsed_index != pos:
                logger.info(
                    "BUILDER1_STAGE_NORMALIZE stage=series_ads field=index action=position_override "
                    "model_index=%s authoritative_index=%s",
                    parsed_index,
                    pos,
                )
            ad["index"] = pos
        normalized.append(ad)
    return normalized


def parse_series_ads_output(
    raw_payload: object,
    *,
    expected_ad_count: int,
    product_description: str = "",
    visibility_policy: Optional[Any] = None,
) -> SeriesAdsOutput:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("series_ads", ["series_ads_not_object"]) from exc

    _reject_forbidden_keys(obj, SERIES_ADS_FORBIDDEN, reasons, "series_ads")
    _reject_legacy_fields(obj, reasons)

    series_dict: Dict[str, str] = {}
    series_raw = obj.get("seriesGenerator")
    if isinstance(series_raw, str):
        reasons.append("series_generator_not_object")
    elif series_raw is None:
        reasons.append("series_generator_missing")
    else:
        series_reasons: List[str] = []
        series = _parse_series_generator(series_raw, series_reasons)
        if series is None or series_reasons:
            reasons.extend(series_reasons or ["series_generator_incomplete"])
        else:
            series_dict = {
                "type": series.type,
                "principle": series.principle,
                "progression": series.progression,
            }

    ads_raw = obj.get("ads")
    normalized_ads = _normalize_ad_indexes(ads_raw if isinstance(ads_raw, list) else [], expected_ad_count, reasons)
    normalized_ads = strip_model_slogan_fields_from_series_ads(normalized_ads)
    if visibility_policy is not None:
        from engine.builder1_product_visibility import enforce_series_ad_visibility_fields

        normalized_ads = enforce_series_ad_visibility_fields(normalized_ads, policy=visibility_policy)
    reasons.extend(
        validate_series_ads_boundary_text(normalized_ads, product_description=product_description)
    )

    if reasons:
        log_stage_parse_failure("series_ads", obj, reasons)
        raise StageParseError("series_ads", reasons)

    return SeriesAdsOutput(series_generator=series_dict, ads=normalized_ads)


def assemble_builder1_campaign(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    detected_language: str,
    exploration_seed: str,
    product_name_resolved: str,
    strategy: StrategyCandidate,
    strategy_selection: StrategySelection,
    selected_slogan: SloganCandidate,
    conceptual: ConceptualCandidate,
    brand_physical: BrandPhysicalOutput,
    graphic: Builder1GraphicGenerator,
    series_ads: SeriesAdsOutput,
    visibility_policy: Any = None,
    visibility_source: Any = None,
) -> Builder1SeriesPlan:
    """Deterministic final plan assembly — server injects authoritative fields."""
    from engine.builder1_product_visibility import (
        ProductVisibilityPolicy,
        ProductVisibilitySource,
    )

    if visibility_policy is None:
        visibility_policy = ProductVisibilityPolicy.FORBIDDEN
    if visibility_source is None:
        visibility_source = ProductVisibilitySource.DEFAULT
    evidence = strategy.brief_support
    if check_unsupported_evidence(evidence, product_description):
        raise StageParseError("assemble", ["unsupported_evidence_claim"])

    fixed_slogan = selected_slogan.brand_slogan
    assembled_ads = inject_fixed_campaign_slogan_into_series_ads(series_ads.ads, fixed_slogan=fixed_slogan)

    assembled: Dict[str, Any] = {
        "productName": product_name,
        "productDescription": product_description,
        "format": format_value,
        "adCount": ad_count,
        "detectedLanguage": detected_language,
        "productNameResolved": product_name_resolved,
        "strategicProblem": strategy.strategic_problem,
        "strategicProblemEvidence": evidence,
        "relativeAdvantage": strategy.relative_advantage,
        "relativeAdvantageSource": strategy.advantage_source,
        "relativeAdvantageBriefSupport": strategy.brief_support,
        "relativeAdvantageClaimRisk": strategy.claim_risk,
        "problemAdvantageLink": f"{strategy.relative_advantage} addresses {strategy.strategic_problem}",
        "brandSlogan": fixed_slogan,
        "sloganDerivation": selected_slogan.derivation_from_advantage,
        "sloganAction": selected_slogan.implied_action,
        "conceptualGenerator": conceptual.generator,
        "conceptualGeneratorAction": conceptual.action,
        "conceptualGeneratorInput": conceptual.input,
        "conceptualGeneratorTransformation": conceptual.transformation,
        "conceptualGeneratorResult": conceptual.result,
        "conceptualGeneratorWhyItExpressesSlogan": conceptual.why_it_expresses_slogan,
        "conceptualGeneratorWhyItExpressesAdvantage": conceptual.why_it_expresses_advantage,
        "transferredObject": brand_physical.transferred_object,
        "transferredObjectAction": brand_physical.transferred_object_action,
        "whyClearerThanShowingProduct": brand_physical.why_clearer_than_showing_product,
        "brandOwnershipReason": selected_slogan.why_ownable,
        "competitorTransferTest": selected_slogan.competitor_transfer_risk,
        "transferRisk": selected_slogan.competitor_transfer_risk,
        "physicalGenerator": brand_physical.physical_generator,
        "physicalGeneratorNaturalPurpose": brand_physical.physical_generator_natural_purpose,
        "physicalGeneratorCampaignRole": brand_physical.physical_generator_campaign_role,
        "graphicGenerator": {
            "palette": {
                "dominant": graphic.palette.dominant,
                "secondary": graphic.palette.secondary,
                "accent": graphic.palette.accent,
                "background": graphic.palette.background,
                "text": graphic.palette.text,
            },
            "layoutTemplate": graphic.layout_template,
            "headlinePlacement": graphic.headline_placement,
            "headlineAlignment": graphic.headline_alignment,
            "headlineMaxWidthPercent": graphic.headline_max_width_percent,
            "brandBlockPlacement": graphic.brand_block_placement,
            "sloganPlacement": graphic.slogan_placement,
            "sloganPlacementReason": getattr(graphic, "slogan_placement_reason", ""),
            "copySafeArea": {
                "side": graphic.copy_safe_area.side,
                "widthPercent": graphic.copy_safe_area.width_percent,
            },
            "typographyStyle": graphic.typography_style,
            "headlineScale": graphic.headline_scale,
            "brandScale": graphic.brand_scale,
            "sloganScale": graphic.slogan_scale,
            "imageStyle": graphic.image_style,
            "backgroundTreatment": graphic.background_treatment,
            "borderTreatment": graphic.border_treatment,
            "recurringGraphicDevice": graphic.recurring_graphic_device,
            "recurringGraphicDeviceRule": graphic.recurring_graphic_device_rule,
            "shapeLanguage": graphic.shape_language,
            "framingRule": graphic.framing_rule,
            "spacingRule": graphic.spacing_rule,
        },
        "seriesGenerator": series_ads.series_generator,
        "mediumParticipates": brand_physical.medium_participates,
        "mediumRole": brand_physical.medium_role,
        "campaignRationale": brand_physical.campaign_rationale,
        "productVisibilityPolicy": getattr(visibility_policy, "value", str(visibility_policy)),
        "ads": assembled_ads,
    }

    plan, reasons = validate_series_plan_structure(
        assembled,
        expected_format=format_value,
        expected_ad_count=ad_count,
        product_name=product_name,
        product_description=product_description,
        require_internal_scans=False,
    )
    if plan is None:
        raise StageParseError("assemble", reasons)

    from dataclasses import replace

    ad_internals = build_series_ad_internals(assembled_ads, fixed_slogan=fixed_slogan)
    conceptual_lineage = build_conceptual_lineage(
        selected_slogan=selected_slogan,
        selected_conceptual=conceptual,
    )
    return replace(
        plan,
        planning_internals={
            "conceptualLineage": conceptual_lineage,
            "conceptualGeneratorWhyItExpressesSlogan": conceptual.why_it_expresses_slogan,
            "productVisibilityPolicy": getattr(visibility_policy, "value", str(visibility_policy)),
            "productVisibilitySource": getattr(visibility_source, "value", str(visibility_source)),
            "transferredObject": brand_physical.transferred_object,
            "transferredObjectAction": brand_physical.transferred_object_action,
            "whyClearerThanShowingProduct": brand_physical.why_clearer_than_showing_product,
            "brandOwnershipReason": selected_slogan.why_ownable,
            "competitorTransferTest": selected_slogan.competitor_transfer_risk,
            "transferRisk": selected_slogan.competitor_transfer_risk,
            "sloganPlacementReason": getattr(graphic, "slogan_placement_reason", ""),
            "adInternals": ad_internals,
        },
    )
