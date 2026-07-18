"""
Builder1 campaign-series planning spec (active production schema).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

FORMAT_LANDSCAPE = "landscape"
FORMAT_PORTRAIT = "portrait"
FORMAT_SQUARE = "square"

AD_COUNT_MIN = 2
AD_COUNT_MAX = 4

BRAND_SLOGAN_MAX_WORDS = 6
HEADLINE_MAX_WORDS = 7
MARKETING_TEXT_WORD_COUNT = 50

RELATIVE_ADVANTAGE_SOURCES = {
    "explicit_brief",
    "category_inference",
    "brand_position",
    "observable_product_mechanism",
}

LAYOUT_TEMPLATES = {
    "visual_right_copy_left",
    "visual_left_copy_right",
    "visual_top_copy_bottom",
    "visual_bottom_copy_top",
    "full_visual_copy_overlay",
}

HEADLINE_PLACEMENTS = {
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center_left",
    "center_right",
}

HEADLINE_ALIGNMENTS = {"left", "right", "center"}

HEADLINE_TREATMENTS = {"plain", "bold", "outline", "shadow", "inverted_box"}

TYPOGRAPHY_STYLE_ENUMS = {
    "bold_geometric_sans",
    "clean_modern_sans",
    "editorial_serif",
    "condensed_sans",
    "rounded_sans",
    "high_contrast_display",
}

TEXT_SCALE_ENUMS = {"small", "medium", "large", "extra_large"}

IMAGE_STYLE_ENUMS = {
    "editorial_photography",
    "studio_product",
    "documentary",
    "illustration_flat",
    "illustration_textured",
    "cinematic",
    "minimal_photography",
}

BACKGROUND_TREATMENT_ENUMS = {
    "solid",
    "gradient",
    "textured",
    "photographic_blur",
    "split_color",
}

BORDER_TREATMENT_ENUMS = {"none", "thin_frame", "heavy_frame", "rounded_frame", "cutout_shadow"}

COPY_SAFE_SIDES = {"left", "right", "top", "bottom"}

WEAK_CONCEPTUAL_TERMS = {
    "transparency",
    "confidence",
    "growth",
    "results",
    "visibility",
    "connection",
    "smart advertising",
    "smart ads",
    "trust",
    "quality",
    "innovation",
    "attention",
    "simplicity",
    "being central",
}

INTERNAL_PLAN_FIELDS = {
    "strategyCandidateScan",
    "conceptualGeneratorScan",
    "campaignSelfCheck",
    "strategyJudgeResult",
    "strategyFamily",
    "strategyScore",
    "campaignExplorationSeed",
    "selectionReason",
    "planningInternals",
    "conceptualGeneratorWhyItExpressesSlogan",
    "embodimentChoice",
    "productVisibilityJustification",
    "brandOwnershipReason",
    "competitorTransferTest",
    "transferRisk",
    "categoryRelevancePatched",
}


@dataclass
class Builder1Palette:
    dominant: str
    secondary: str
    accent: str
    background: str
    text: str


@dataclass
class Builder1CopySafeArea:
    side: str
    width_percent: int


@dataclass
class Builder1GraphicGenerator:
    palette: Builder1Palette
    layout_template: str
    headline_placement: str
    headline_alignment: str
    headline_max_width_percent: int
    brand_block_placement: str
    slogan_placement: str
    copy_safe_area: Builder1CopySafeArea
    typography_style: str
    headline_scale: str
    brand_scale: str
    slogan_scale: str
    image_style: str
    background_treatment: str
    border_treatment: str
    recurring_graphic_device: str
    recurring_graphic_device_rule: str
    shape_language: str
    framing_rule: str
    spacing_rule: str
    slogan_placement_reason: str = ""


@dataclass
class Builder1SeriesGenerator:
    type: str
    principle: str
    progression: str


@dataclass
class Builder1AdPlan:
    index: int
    variation_label: str
    new_contribution: str
    physical_execution: str
    visual_execution: str
    scene_description: str
    conceptual_execution: str
    conceptual_action_proof: str
    headline: Optional[str]
    headline_needed_reason: str
    marketing_text: str


@dataclass
class Builder1SeriesPlan:
    product_name: str
    product_description: str
    format: str
    ad_count: int
    product_name_resolved: str
    detected_language: str
    strategic_problem: str
    strategic_problem_evidence: str
    relative_advantage: str
    relative_advantage_source: str
    relative_advantage_brief_support: str
    relative_advantage_claim_risk: str
    problem_advantage_link: str
    brand_slogan: str
    slogan_derivation: str
    slogan_action: str
    conceptual_generator: str
    conceptual_generator_action: str
    conceptual_generator_input: str
    conceptual_generator_transformation: str
    conceptual_generator_result: str
    conceptual_generator_why_it_expresses_advantage: str
    physical_generator: str
    physical_generator_natural_purpose: str
    physical_generator_campaign_role: str
    graphic_generator: Builder1GraphicGenerator
    series_generator: Builder1SeriesGenerator
    medium_participates: bool
    medium_role: str
    campaign_rationale: str
    ads: List[Builder1AdPlan] = field(default_factory=list)
    planning_internals: Dict[str, Any] = field(default_factory=dict)


def graphic_generator_to_dict(g: Builder1GraphicGenerator) -> Dict[str, Any]:
    return {
        "palette": {
            "dominant": g.palette.dominant,
            "secondary": g.palette.secondary,
            "accent": g.palette.accent,
            "background": g.palette.background,
            "text": g.palette.text,
        },
        "layoutTemplate": g.layout_template,
        "headlinePlacement": g.headline_placement,
        "headlineAlignment": g.headline_alignment,
        "headlineMaxWidthPercent": g.headline_max_width_percent,
        "brandBlockPlacement": g.brand_block_placement,
        "sloganPlacement": g.slogan_placement,
        "sloganPlacementReason": g.slogan_placement_reason,
        "copySafeArea": {
            "side": g.copy_safe_area.side,
            "widthPercent": g.copy_safe_area.width_percent,
        },
        "typographyStyle": g.typography_style,
        "headlineScale": g.headline_scale,
        "brandScale": g.brand_scale,
        "sloganScale": g.slogan_scale,
        "imageStyle": g.image_style,
        "backgroundTreatment": g.background_treatment,
        "borderTreatment": g.border_treatment,
        "recurringGraphicDevice": g.recurring_graphic_device,
        "recurringGraphicDeviceRule": g.recurring_graphic_device_rule,
        "shapeLanguage": g.shape_language,
        "framingRule": g.framing_rule,
        "spacingRule": g.spacing_rule,
    }


def series_generator_to_dict(s: Builder1SeriesGenerator) -> Dict[str, Any]:
    return {"type": s.type, "principle": s.principle, "progression": s.progression}


def campaign_identity_to_dict(plan: Builder1SeriesPlan) -> Dict[str, Any]:
    return {
        "productNameResolved": plan.product_name_resolved,
        "detectedLanguage": plan.detected_language,
        "format": plan.format,
        "adCount": plan.ad_count,
        "strategicProblem": plan.strategic_problem,
        "strategicProblemEvidence": plan.strategic_problem_evidence,
        "relativeAdvantage": plan.relative_advantage,
        "relativeAdvantageSource": plan.relative_advantage_source,
        "relativeAdvantageBriefSupport": plan.relative_advantage_brief_support,
        "relativeAdvantageClaimRisk": plan.relative_advantage_claim_risk,
        "problemAdvantageLink": plan.problem_advantage_link,
        "brandSlogan": plan.brand_slogan,
        "sloganDerivation": plan.slogan_derivation,
        "sloganAction": plan.slogan_action,
        "conceptualGenerator": plan.conceptual_generator,
        "conceptualGeneratorAction": plan.conceptual_generator_action,
        "conceptualGeneratorInput": plan.conceptual_generator_input,
        "conceptualGeneratorTransformation": plan.conceptual_generator_transformation,
        "conceptualGeneratorResult": plan.conceptual_generator_result,
        "conceptualGeneratorWhyItExpressesAdvantage": plan.conceptual_generator_why_it_expresses_advantage,
        "physicalGenerator": plan.physical_generator,
        "physicalGeneratorNaturalPurpose": plan.physical_generator_natural_purpose,
        "physicalGeneratorCampaignRole": plan.physical_generator_campaign_role,
        "graphicGenerator": graphic_generator_to_dict(plan.graphic_generator),
        "seriesGenerator": series_generator_to_dict(plan.series_generator),
        "mediumParticipates": plan.medium_participates,
        "mediumRole": plan.medium_role,
        "campaignRationale": plan.campaign_rationale,
    }


def ad_to_public_api_dict(
    ad: Builder1AdPlan,
    *,
    visual_prompt: str = "",
    image_base64: str = "",
) -> Dict[str, Any]:
    """Public incremental response — no future ad internals."""
    return {
        "index": ad.index,
        "headline": ad.headline,
        "marketingText": ad.marketing_text,
        "visualPrompt": visual_prompt,
        "imageBase64": image_base64,
        "imageContainsFinalCopy": True,
    }


def series_plan_to_store_dict(plan: Builder1SeriesPlan) -> Dict[str, Any]:
    base = {
        "productName": plan.product_name,
        "productDescription": plan.product_description,
        "format": plan.format,
        "adCount": plan.ad_count,
        **campaign_identity_to_dict(plan),
        "ads": [],
    }
    internals = plan.planning_internals or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    for a in plan.ads:
        ad_dict = {
            "index": a.index,
            "variationLabel": a.variation_label,
            "newContribution": a.new_contribution,
            "physicalExecution": a.physical_execution,
            "visualExecution": a.visual_execution,
            "sceneDescription": a.scene_description,
            "conceptualExecution": a.conceptual_execution,
            "conceptualActionProof": a.conceptual_action_proof,
            "headline": a.headline,
            "headlineNeededReason": a.headline_needed_reason,
            "marketingText": a.marketing_text,
        }
        extra = ad_internals.get(a.index) or ad_internals.get(str(a.index))
        if isinstance(extra, dict):
            ad_dict.update(extra)
        base["ads"].append(ad_dict)
    if internals:
        base.update(
            {
                key: value
                for key, value in internals.items()
                if key != "adInternals"
            }
        )
    return base


def _palette_from_dict(raw: Dict[str, Any]) -> Builder1Palette:
    return Builder1Palette(
        dominant=str(raw.get("dominant") or ""),
        secondary=str(raw.get("secondary") or ""),
        accent=str(raw.get("accent") or ""),
        background=str(raw.get("background") or ""),
        text=str(raw.get("text") or ""),
    )


def series_plan_from_store_dict(data: Dict[str, Any]) -> Builder1SeriesPlan:
    from engine.builder1_plan_parser import parse_builder1_series_plan

    return parse_builder1_series_plan(
        data,
        expected_format=str(data.get("format") or "portrait"),
        expected_ad_count=int(data.get("adCount") or 2),
        product_name=str(data.get("productName") or ""),
        product_description=str(data.get("productDescription") or ""),
        require_internal_scans=False,
    )
