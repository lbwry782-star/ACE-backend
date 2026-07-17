"""
Builder1 campaign-series plan parser and deterministic validation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from engine.builder1_plan_spec import (
    AD_COUNT_MAX,
    AD_COUNT_MIN,
    BRAND_SLOGAN_MAX_WORDS,
    FORMAT_LANDSCAPE,
    FORMAT_PORTRAIT,
    FORMAT_SQUARE,
    HEADLINE_MAX_WORDS,
    Builder1AdPlan,
    Builder1CompositionGrid,
    Builder1GraphicGenerator,
    Builder1SeriesGenerator,
    Builder1SeriesPlan,
    Builder1Typography,
)

SUPPORTED_LANGUAGES = {"he", "en", "ar", "ru", "fr", "de", "es", "it", "pt", "nl"}

LEGACY_FIELDS = {
    "objectA",
    "objectASecondary",
    "objectB",
    "visualSimilarityScore",
    "modeDecision",
    "advertisingPromise",
    "visualDescription",
}

LEGACY_MODES = {"SIDE_BY_SIDE", "REPLACEMENT"}


class Builder1SeriesPlanParseError(ValueError):
    def __init__(self, reason: str, details: Optional[str] = None):
        self.reason = reason
        self.details = details
        super().__init__(reason if not details else f"{reason}: {details}")


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    return " ".join(s.strip().split())


def _word_count(text: str) -> int:
    t = _norm_text(text)
    if not t:
        return 0
    return len(t.split())


def _normalize_headline(value: object) -> Optional[str]:
    if value is None:
        return None
    s = _norm_text(value)
    return s if s else None


def _reject_legacy_fields(obj: Dict[str, Any], reasons: List[str]) -> None:
    for key in LEGACY_FIELDS:
        if key in obj and obj.get(key) not in (None, "", []):
            reasons.append(f"legacy_field_present:{key}")
    mode = _norm_text(obj.get("modeDecision"))
    if mode in LEGACY_MODES:
        reasons.append(f"legacy_mode:{mode}")


def _parse_graphic_generator(raw: object, reasons: List[str]) -> Optional[Builder1GraphicGenerator]:
    if not isinstance(raw, dict):
        reasons.append("graphic_generator_not_object")
        return None
    palette = raw.get("colorPalette")
    if not isinstance(palette, list) or not palette:
        reasons.append("graphic_generator_missing_palette")
        return None
    colors = [_norm_text(c) for c in palette if _norm_text(c)]
    if not colors:
        reasons.append("graphic_generator_empty_palette")
        return None

    typo_raw = raw.get("typography")
    if not isinstance(typo_raw, dict):
        reasons.append("graphic_generator_missing_typography")
        return None
    headline_style = _norm_text(typo_raw.get("headlineStyle"))
    slogan_style = _norm_text(typo_raw.get("sloganStyle"))
    brand_style = _norm_text(typo_raw.get("brandStyle"))
    if not headline_style or not slogan_style or not brand_style:
        reasons.append("graphic_generator_incomplete_typography")
        return None

    comp_raw = raw.get("composition")
    if not isinstance(comp_raw, dict):
        reasons.append("graphic_generator_missing_composition")
        return None
    grid = _norm_text(comp_raw.get("grid"))
    visual_area = _norm_text(comp_raw.get("visualArea"))
    copy_area = _norm_text(comp_raw.get("copyArea"))
    alignment = _norm_text(comp_raw.get("alignment"))
    slogan_placement = _norm_text(comp_raw.get("sloganPlacement"))
    brand_placement = _norm_text(comp_raw.get("brandPlacement"))
    if not all([grid, visual_area, copy_area, alignment, slogan_placement, brand_placement]):
        reasons.append("graphic_generator_incomplete_composition")
        return None

    image_style = _norm_text(raw.get("imageStyle"))
    spacing = _norm_text(raw.get("spacing"))
    visual_treatment = _norm_text(raw.get("visualTreatment"))
    background_treatment = _norm_text(raw.get("backgroundTreatment"))
    if not all([image_style, spacing, visual_treatment, background_treatment]):
        reasons.append("graphic_generator_missing_style_fields")
        return None

    return Builder1GraphicGenerator(
        color_palette=colors,
        typography=Builder1Typography(
            headline_style=headline_style,
            slogan_style=slogan_style,
            brand_style=brand_style,
        ),
        composition=Builder1CompositionGrid(
            grid=grid,
            visual_area=visual_area,
            copy_area=copy_area,
            alignment=alignment,
            slogan_placement=slogan_placement,
            brand_placement=brand_placement,
        ),
        image_style=image_style,
        spacing=spacing,
        visual_treatment=visual_treatment,
        background_treatment=background_treatment,
    )


def _parse_series_generator(raw: object, reasons: List[str]) -> Optional[Builder1SeriesGenerator]:
    if not isinstance(raw, dict):
        reasons.append("series_generator_not_object")
        return None
    t = _norm_text(raw.get("type"))
    principle = _norm_text(raw.get("principle"))
    progression = _norm_text(raw.get("progression"))
    if not t or not principle or not progression:
        reasons.append("series_generator_incomplete")
        return None
    return Builder1SeriesGenerator(type=t, principle=principle, progression=progression)


def validate_series_plan_structure(
    obj: Dict[str, Any],
    *,
    expected_format: str,
    expected_ad_count: int,
    product_name: str,
    product_description: str,
) -> Tuple[Optional[Builder1SeriesPlan], List[str]]:
    reasons: List[str] = []

    if not isinstance(obj, dict):
        return None, ["plan_not_object"]

    _reject_legacy_fields(obj, reasons)

    detected = _norm_text(obj.get("detectedLanguage")).lower()
    if not detected or detected not in SUPPORTED_LANGUAGES:
        reasons.append("invalid_detected_language")

    fmt = _norm_text(obj.get("format")).lower()
    if fmt != expected_format:
        reasons.append("format_mismatch")

    try:
        ad_count = int(obj.get("adCount"))
    except (TypeError, ValueError):
        reasons.append("invalid_ad_count_type")
        ad_count = -1
    if ad_count < AD_COUNT_MIN or ad_count > AD_COUNT_MAX:
        reasons.append("invalid_ad_count_range")
    if ad_count != expected_ad_count:
        reasons.append("ad_count_mismatch")

    required_strings = [
        ("strategicProblem", "missing_strategic_problem"),
        ("strategicProblemEvidence", "missing_strategic_problem_evidence"),
        ("relativeAdvantage", "missing_relative_advantage"),
        ("problemAdvantageLink", "missing_problem_advantage_link"),
        ("brandSlogan", "missing_brand_slogan"),
        ("sloganDerivation", "missing_slogan_derivation"),
        ("sloganAction", "missing_slogan_action"),
        ("conceptualGenerator", "missing_conceptual_generator"),
        ("conceptualGeneratorAction", "missing_conceptual_generator_action"),
        ("physicalGenerator", "missing_physical_generator"),
        ("physicalGeneratorNaturalPurpose", "missing_physical_generator_natural_purpose"),
        ("physicalGeneratorCampaignRole", "missing_physical_generator_campaign_role"),
        ("campaignRationale", "missing_campaign_rationale"),
    ]
    for field_name, code in required_strings:
        if not _norm_text(obj.get(field_name)):
            reasons.append(code)

    brand_slogan = _norm_text(obj.get("brandSlogan"))
    if brand_slogan and _word_count(brand_slogan) > BRAND_SLOGAN_MAX_WORDS:
        reasons.append("brand_slogan_too_long")

    if "brandSlogan" in obj and isinstance(obj.get("ads"), list):
        for ad_raw in obj["ads"]:
            if isinstance(ad_raw, dict):
                for bad in ("brandSlogan", "slogan", "campaignSlogan"):
                    if _norm_text(ad_raw.get(bad)):
                        reasons.append("per_ad_slogan_forbidden")

    medium_participates = obj.get("mediumParticipates")
    if not isinstance(medium_participates, bool):
        reasons.append("medium_participates_not_boolean")
        medium_participates = False
    medium_role = _norm_text(obj.get("mediumRole"))
    if medium_participates and not medium_role:
        reasons.append("medium_role_required_when_participates")
    if not medium_participates and medium_role:
        reasons.append("medium_role_forbidden_when_not_participates")

    graphic = _parse_graphic_generator(obj.get("graphicGenerator"), reasons)
    series_gen = _parse_series_generator(obj.get("seriesGenerator"), reasons)

    ads_raw = obj.get("ads")
    if not isinstance(ads_raw, list):
        reasons.append("ads_not_list")
        return None, reasons

    if len(ads_raw) != expected_ad_count:
        reasons.append("ads_length_mismatch")

    parsed_ads: List[Builder1AdPlan] = []
    seen_indexes: set[int] = set()
    phys_set: set[str] = set()
    vis_set: set[str] = set()
    scene_set: set[str] = set()
    execution_signatures: List[Tuple[str, str, str, Optional[str]]] = []

    for ad_raw in ads_raw:
        if not isinstance(ad_raw, dict):
            reasons.append("ad_not_object")
            continue
        try:
            idx = int(ad_raw.get("index"))
        except (TypeError, ValueError):
            reasons.append("ad_index_invalid")
            continue
        if idx in seen_indexes:
            reasons.append("duplicate_ad_index")
        seen_indexes.add(idx)

        headline = _normalize_headline(ad_raw.get("headline"))
        if headline and _word_count(headline) > HEADLINE_MAX_WORDS:
            reasons.append("headline_too_long")

        pe = _norm_text(ad_raw.get("physicalExecution"))
        ve = _norm_text(ad_raw.get("visualExecution"))
        sd = _norm_text(ad_raw.get("sceneDescription"))
        nc = _norm_text(ad_raw.get("newContribution"))
        if not nc:
            reasons.append("missing_new_contribution")
        if not pe:
            reasons.append("missing_physical_execution")
        if not ve:
            reasons.append("missing_visual_execution")
        if not sd:
            reasons.append("missing_scene_description")

        pe_key = pe.lower()
        ve_key = ve.lower()
        sd_key = sd.lower()
        if pe_key and pe_key in phys_set:
            reasons.append("duplicate_physical_execution")
        if ve_key and ve_key in vis_set:
            reasons.append("duplicate_visual_execution")
        if sd_key and sd_key in scene_set:
            reasons.append("duplicate_scene_description")
        phys_set.add(pe_key)
        vis_set.add(ve_key)
        scene_set.add(sd_key)
        execution_signatures.append((pe_key, ve_key, sd_key, headline))

        parsed_ads.append(
            Builder1AdPlan(
                index=idx,
                variation_label=_norm_text(ad_raw.get("variationLabel")) or f"ad-{idx}",
                new_contribution=nc,
                physical_execution=pe,
                visual_execution=ve,
                scene_description=sd,
                headline=headline,
                headline_needed_reason=_norm_text(ad_raw.get("headlineNeededReason")),
                marketing_text=_norm_text(ad_raw.get("marketingText")),
            )
        )

    expected_indexes = set(range(1, expected_ad_count + 1))
    if seen_indexes != expected_indexes:
        reasons.append("ad_indexes_not_sequential")

    if len(parsed_ads) == expected_ad_count and len(parsed_ads) >= 2:
        varying = False
        for i in range(len(execution_signatures)):
            for j in range(i + 1, len(execution_signatures)):
                a = execution_signatures[i]
                b = execution_signatures[j]
                if a[:3] != b[:3]:
                    varying = True
                    break
                if a[3] != b[3]:
                    pass
            if varying:
                break
        if not varying:
            only_headline_diff = False
            if len(set(execution_signatures)) == 1:
                headlines = [s[3] for s in execution_signatures]
                if len(set(headlines)) > 1:
                    only_headline_diff = True
            if only_headline_diff or (
                len({s[:3] for s in execution_signatures}) == 1
                and len({s[3] for s in execution_signatures}) > 1
            ):
                reasons.append("headline_only_variation")

    if reasons:
        return None, reasons

    parsed_ads.sort(key=lambda a: a.index)
    product_name_resolved = _norm_text(obj.get("productNameResolved")) or product_name or "Product"

    return (
        Builder1SeriesPlan(
            product_name=product_name,
            product_description=product_description,
            format=expected_format,
            ad_count=expected_ad_count,
            product_name_resolved=product_name_resolved,
            detected_language=detected,
            strategic_problem=_norm_text(obj.get("strategicProblem")),
            strategic_problem_evidence=_norm_text(obj.get("strategicProblemEvidence")),
            relative_advantage=_norm_text(obj.get("relativeAdvantage")),
            problem_advantage_link=_norm_text(obj.get("problemAdvantageLink")),
            brand_slogan=brand_slogan,
            slogan_derivation=_norm_text(obj.get("sloganDerivation")),
            slogan_action=_norm_text(obj.get("sloganAction")),
            conceptual_generator=_norm_text(obj.get("conceptualGenerator")),
            conceptual_generator_action=_norm_text(obj.get("conceptualGeneratorAction")),
            physical_generator=_norm_text(obj.get("physicalGenerator")),
            physical_generator_natural_purpose=_norm_text(obj.get("physicalGeneratorNaturalPurpose")),
            physical_generator_campaign_role=_norm_text(obj.get("physicalGeneratorCampaignRole")),
            graphic_generator=graphic,  # type: ignore[arg-type]
            series_generator=series_gen,  # type: ignore[arg-type]
            medium_participates=medium_participates,
            medium_role=medium_role if medium_participates else "",
            campaign_rationale=_norm_text(obj.get("campaignRationale")),
            ads=parsed_ads,
        ),
        [],
    )


def parse_builder1_series_plan(
    raw: Dict[str, Any],
    *,
    expected_format: str,
    expected_ad_count: int,
    product_name: str,
    product_description: str,
) -> Builder1SeriesPlan:
    plan, reasons = validate_series_plan_structure(
        raw,
        expected_format=expected_format,
        expected_ad_count=expected_ad_count,
        product_name=product_name,
        product_description=product_description,
    )
    if plan is None:
        raise Builder1SeriesPlanParseError(reasons[0] if reasons else "invalid_plan", ";".join(reasons))
    return plan
