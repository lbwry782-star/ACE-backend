"""
Builder1 campaign-series plan parser and deterministic validation.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.builder1_plan_spec import (
    AD_COUNT_MAX,
    AD_COUNT_MIN,
    BACKGROUND_TREATMENT_ENUMS,
    BORDER_TREATMENT_ENUMS,
    BRAND_SLOGAN_MAX_WORDS,
    COPY_SAFE_SIDES,
    HEADLINE_ALIGNMENTS,
    HEADLINE_MAX_WORDS,
    HEADLINE_PLACEMENTS,
    HEADLINE_TREATMENTS,
    IMAGE_STYLE_ENUMS,
    LAYOUT_TEMPLATES,
    RELATIVE_ADVANTAGE_SOURCES,
    WEAK_CONCEPTUAL_TERMS,
    Builder1AdPlan,
    Builder1CopySafeArea,
    Builder1GraphicGenerator,
    Builder1Palette,
    Builder1SeriesGenerator,
    Builder1SeriesPlan,
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

HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

EXPLORATION_LENSES = {
    "economic",
    "perceptual",
    "emotional",
    "operational",
    "time",
    "accessibility",
    "expertise",
    "challenger_positioning",
    "participation",
    "simplicity",
    "specialization",
    "category_convention",
    "weakness_converted",
}

UNSUPPORTED_CAPABILITY_TERMS = (
    "live dashboard",
    "dashboard",
    "guaranteed results",
    "guaranteed roi",
    "measurable sales",
    "sales increase",
    "live reporting",
    "real-time reporting",
    "money-back guarantee",
)


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
    return len(t.split()) if t else 0


def _normalize_headline(value: object) -> Optional[str]:
    if value is None:
        return None
    s = _norm_text(value)
    return s if s else None


def _norm_key(text: str) -> str:
    return _norm_text(text).lower()


def _is_valid_hex(value: str) -> bool:
    return bool(HEX_COLOR_RE.match(_norm_text(value)))


def _reject_legacy_fields(obj: Dict[str, Any], reasons: List[str]) -> None:
    for key in LEGACY_FIELDS:
        if key in obj and obj.get(key) not in (None, "", []):
            reasons.append(f"legacy_field_present:{key}")
    mode = _norm_text(obj.get("modeDecision"))
    if mode in LEGACY_MODES:
        reasons.append(f"legacy_mode:{mode}")


def _validate_strategy_candidate_scan(raw: object, reasons: List[str]) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        reasons.append("strategy_scan_not_object")
        return
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 10:
        reasons.append("strategy_scan_insufficient_candidates")
        return
    lenses: Set[str] = set()
    signatures: Set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            reasons.append("strategy_scan_candidate_not_object")
            continue
        lens = _norm_key(str(item.get("lens") or ""))
        problem = _norm_key(str(item.get("problem") or ""))
        advantage = _norm_key(str(item.get("advantage") or ""))
        if lens:
            lenses.add(lens)
        sig = f"{problem}|{advantage}"
        if sig.strip("|"):
            signatures.add(sig)
    if len(lenses) < 8:
        reasons.append("strategy_scan_insufficient_lens_diversity")
    if len(signatures) < 8:
        reasons.append("strategy_scan_duplicate_candidates")


def _check_unsupported_claims(
    *,
    product_description: str,
    relative_advantage: str,
    relative_advantage_source: str,
    reasons: List[str],
) -> None:
    brief = _norm_key(product_description)
    adv = _norm_key(relative_advantage)
    if relative_advantage_source == "explicit_brief":
        return
    for term in UNSUPPORTED_CAPABILITY_TERMS:
        if term in adv and term not in brief:
            reasons.append("unsupported_product_capability")
            break


def _parse_graphic_generator(raw: object, reasons: List[str]) -> Optional[Builder1GraphicGenerator]:
    if not isinstance(raw, dict):
        reasons.append("graphic_generator_not_object")
        return None

    palette_raw = raw.get("palette")
    if not isinstance(palette_raw, dict):
        reasons.append("graphic_generator_missing_palette")
        return None
    palette_fields = ("dominant", "secondary", "accent", "background", "text")
    palette_vals = {k: _norm_text(palette_raw.get(k)) for k in palette_fields}
    if not all(palette_vals.values()):
        reasons.append("graphic_generator_incomplete_palette")
        return None
    for val in palette_vals.values():
        if not _is_valid_hex(val):
            reasons.append("graphic_generator_invalid_hex")
            break

    layout = _norm_text(raw.get("layoutTemplate"))
    if layout not in LAYOUT_TEMPLATES:
        reasons.append("graphic_generator_invalid_layout")

    headline_placement = _norm_text(raw.get("headlinePlacement"))
    if headline_placement not in HEADLINE_PLACEMENTS:
        reasons.append("graphic_generator_invalid_headline_placement")

    headline_alignment = _norm_text(raw.get("headlineAlignment"))
    if headline_alignment not in HEADLINE_ALIGNMENTS:
        reasons.append("graphic_generator_invalid_headline_alignment")

    try:
        headline_max_width = int(raw.get("headlineMaxWidthPercent"))
    except (TypeError, ValueError):
        reasons.append("graphic_generator_invalid_headline_width")
        headline_max_width = -1
    if headline_max_width < 10 or headline_max_width > 50:
        reasons.append("graphic_generator_invalid_headline_width")

    headline_color = _norm_text(raw.get("headlineColor"))
    if not _is_valid_hex(headline_color):
        reasons.append("graphic_generator_invalid_headline_color")

    headline_treatment = _norm_text(raw.get("headlineTreatment"))
    if headline_treatment not in HEADLINE_TREATMENTS:
        reasons.append("graphic_generator_invalid_headline_treatment")

    brand_block = _norm_text(raw.get("brandBlockPlacement"))
    if brand_block not in HEADLINE_PLACEMENTS:
        reasons.append("graphic_generator_invalid_brand_placement")

    slogan_placement = _norm_text(raw.get("sloganPlacement"))
    if slogan_placement not in HEADLINE_PLACEMENTS:
        reasons.append("graphic_generator_invalid_slogan_placement")

    copy_raw = raw.get("copySafeArea")
    if not isinstance(copy_raw, dict):
        reasons.append("graphic_generator_missing_copy_safe_area")
        return None
    side = _norm_text(copy_raw.get("side"))
    if side not in COPY_SAFE_SIDES:
        reasons.append("graphic_generator_invalid_copy_safe_side")
    try:
        width_percent = int(copy_raw.get("widthPercent"))
    except (TypeError, ValueError):
        reasons.append("graphic_generator_invalid_copy_safe_width")
        width_percent = -1
    if width_percent < 15 or width_percent > 50:
        reasons.append("graphic_generator_invalid_copy_safe_width")

    image_style = _norm_text(raw.get("imageStyle"))
    if image_style not in IMAGE_STYLE_ENUMS:
        reasons.append("graphic_generator_invalid_image_style")

    background = _norm_text(raw.get("backgroundTreatment"))
    if background not in BACKGROUND_TREATMENT_ENUMS:
        reasons.append("graphic_generator_invalid_background")

    border = _norm_text(raw.get("borderTreatment"))
    if border not in BORDER_TREATMENT_ENUMS:
        reasons.append("graphic_generator_invalid_border")

    device = _norm_text(raw.get("recurringGraphicDevice"))
    device_rule = _norm_text(raw.get("recurringGraphicDeviceRule"))
    framing = _norm_text(raw.get("framingRule"))
    if not device:
        reasons.append("graphic_generator_missing_recurring_device")
    if not device_rule:
        reasons.append("graphic_generator_missing_device_rule")
    if not framing:
        reasons.append("graphic_generator_missing_framing_rule")

    if reasons:
        return None

    return Builder1GraphicGenerator(
        palette=Builder1Palette(**palette_vals),
        layout_template=layout,
        headline_placement=headline_placement,
        headline_alignment=headline_alignment,
        headline_max_width_percent=headline_max_width,
        headline_color=headline_color,
        headline_treatment=headline_treatment,
        brand_block_placement=brand_block,
        slogan_placement=slogan_placement,
        copy_safe_area=Builder1CopySafeArea(side=side, width_percent=width_percent),
        image_style=image_style,
        background_treatment=background,
        border_treatment=border,
        recurring_graphic_device=device,
        recurring_graphic_device_rule=device_rule,
        framing_rule=framing,
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


def _validate_conceptual_generator(
    *,
    conceptual: str,
    conceptual_action: str,
    conceptual_input: str,
    conceptual_transform: str,
    conceptual_result: str,
    physical: str,
    reasons: List[str],
) -> None:
    if not conceptual_action:
        reasons.append("missing_conceptual_generator_action")
    if not conceptual_input:
        reasons.append("missing_conceptual_generator_input")
    if not conceptual_transform:
        reasons.append("missing_conceptual_generator_transformation")
    if not conceptual_result:
        reasons.append("missing_conceptual_generator_result")
    c_key = _norm_key(conceptual)
    p_key = _norm_key(physical)
    if c_key and p_key and c_key == p_key:
        reasons.append("conceptual_equals_physical_generator")
    if c_key in WEAK_CONCEPTUAL_TERMS and not conceptual_transform:
        reasons.append("conceptual_generator_too_generic")


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
    _validate_strategy_candidate_scan(obj.get("strategyCandidateScan"), reasons)

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
        ("relativeAdvantageBriefSupport", "missing_relative_advantage_brief_support"),
        ("relativeAdvantageClaimRisk", "missing_relative_advantage_claim_risk"),
        ("problemAdvantageLink", "missing_problem_advantage_link"),
        ("brandSlogan", "missing_brand_slogan"),
        ("sloganDerivation", "missing_slogan_derivation"),
        ("sloganAction", "missing_slogan_action"),
        ("conceptualGenerator", "missing_conceptual_generator"),
        ("conceptualGeneratorAction", "missing_conceptual_generator_action"),
        ("conceptualGeneratorInput", "missing_conceptual_generator_input"),
        ("conceptualGeneratorTransformation", "missing_conceptual_generator_transformation"),
        ("conceptualGeneratorResult", "missing_conceptual_generator_result"),
        ("conceptualGeneratorWhyItExpressesAdvantage", "missing_conceptual_generator_why"),
        ("physicalGenerator", "missing_physical_generator"),
        ("physicalGeneratorNaturalPurpose", "missing_physical_generator_natural_purpose"),
        ("physicalGeneratorCampaignRole", "missing_physical_generator_campaign_role"),
        ("campaignRationale", "missing_campaign_rationale"),
    ]
    for field_name, code in required_strings:
        if not _norm_text(obj.get(field_name)):
            reasons.append(code)

    adv_source = _norm_text(obj.get("relativeAdvantageSource"))
    if adv_source not in RELATIVE_ADVANTAGE_SOURCES:
        reasons.append("invalid_relative_advantage_source")

    relative_advantage = _norm_text(obj.get("relativeAdvantage"))
    _check_unsupported_claims(
        product_description=product_description,
        relative_advantage=relative_advantage,
        relative_advantage_source=adv_source,
        reasons=reasons,
    )

    brand_slogan = _norm_text(obj.get("brandSlogan"))
    if brand_slogan and _word_count(brand_slogan) > BRAND_SLOGAN_MAX_WORDS:
        reasons.append("brand_slogan_too_long")

    if isinstance(obj.get("ads"), list):
        for ad_raw in obj["ads"]:
            if isinstance(ad_raw, dict):
                for bad in ("brandSlogan", "slogan", "campaignSlogan"):
                    if _norm_text(ad_raw.get(bad)):
                        reasons.append("per_ad_slogan_forbidden")

    _validate_conceptual_generator(
        conceptual=_norm_text(obj.get("conceptualGenerator")),
        conceptual_action=_norm_text(obj.get("conceptualGeneratorAction")),
        conceptual_input=_norm_text(obj.get("conceptualGeneratorInput")),
        conceptual_transform=_norm_text(obj.get("conceptualGeneratorTransformation")),
        conceptual_result=_norm_text(obj.get("conceptualGeneratorResult")),
        physical=_norm_text(obj.get("physicalGenerator")),
        reasons=reasons,
    )

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
    conceptual_exec_set: set[str] = set()
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
        ce = _norm_text(ad_raw.get("conceptualExecution"))
        cap = _norm_text(ad_raw.get("conceptualActionProof"))
        if not nc:
            reasons.append("missing_new_contribution")
        if not pe:
            reasons.append("missing_physical_execution")
        if not ve:
            reasons.append("missing_visual_execution")
        if not sd:
            reasons.append("missing_scene_description")
        if not ce:
            reasons.append("missing_conceptual_execution")
        if not cap:
            reasons.append("missing_conceptual_action_proof")

        pe_key = pe.lower()
        ve_key = ve.lower()
        sd_key = sd.lower()
        ce_key = ce.lower()
        if pe_key and pe_key in phys_set:
            reasons.append("duplicate_physical_execution")
        if ve_key and ve_key in vis_set:
            reasons.append("duplicate_visual_execution")
        if sd_key and sd_key in scene_set:
            reasons.append("duplicate_scene_description")
        if ce_key and ce_key in conceptual_exec_set:
            reasons.append("duplicate_conceptual_execution")
        phys_set.add(pe_key)
        vis_set.add(ve_key)
        scene_set.add(sd_key)
        conceptual_exec_set.add(ce_key)
        execution_signatures.append((pe_key, ve_key, sd_key, headline))

        parsed_ads.append(
            Builder1AdPlan(
                index=idx,
                variation_label=_norm_text(ad_raw.get("variationLabel")) or f"ad-{idx}",
                new_contribution=nc,
                physical_execution=pe,
                visual_execution=ve,
                scene_description=sd,
                conceptual_execution=ce,
                conceptual_action_proof=cap,
                headline=headline,
                headline_needed_reason=_norm_text(ad_raw.get("headlineNeededReason")),
                marketing_text=_norm_text(ad_raw.get("marketingText")),
            )
        )

    expected_indexes = set(range(1, expected_ad_count + 1))
    if seen_indexes != expected_indexes:
        reasons.append("ad_indexes_not_sequential")

    if len(parsed_ads) == expected_ad_count and len(parsed_ads) >= 2:
        if len({s[:3] for s in execution_signatures}) == 1 and len({s[3] for s in execution_signatures}) > 1:
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
            relative_advantage=relative_advantage,
            relative_advantage_source=adv_source,
            relative_advantage_brief_support=_norm_text(obj.get("relativeAdvantageBriefSupport")),
            relative_advantage_claim_risk=_norm_text(obj.get("relativeAdvantageClaimRisk")),
            problem_advantage_link=_norm_text(obj.get("problemAdvantageLink")),
            brand_slogan=brand_slogan,
            slogan_derivation=_norm_text(obj.get("sloganDerivation")),
            slogan_action=_norm_text(obj.get("sloganAction")),
            conceptual_generator=_norm_text(obj.get("conceptualGenerator")),
            conceptual_generator_action=_norm_text(obj.get("conceptualGeneratorAction")),
            conceptual_generator_input=_norm_text(obj.get("conceptualGeneratorInput")),
            conceptual_generator_transformation=_norm_text(obj.get("conceptualGeneratorTransformation")),
            conceptual_generator_result=_norm_text(obj.get("conceptualGeneratorResult")),
            conceptual_generator_why_it_expresses_advantage=_norm_text(
                obj.get("conceptualGeneratorWhyItExpressesAdvantage")
            ),
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
