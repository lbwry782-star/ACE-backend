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

# Legacy Object A/B schema removed from active production.


@dataclass
class Builder1Typography:
    headline_style: str
    slogan_style: str
    brand_style: str


@dataclass
class Builder1CompositionGrid:
    grid: str
    visual_area: str
    copy_area: str
    alignment: str
    slogan_placement: str
    brand_placement: str


@dataclass
class Builder1GraphicGenerator:
    color_palette: List[str]
    typography: Builder1Typography
    composition: Builder1CompositionGrid
    image_style: str
    spacing: str
    visual_treatment: str
    background_treatment: str


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
    problem_advantage_link: str
    brand_slogan: str
    slogan_derivation: str
    slogan_action: str
    conceptual_generator: str
    conceptual_generator_action: str
    physical_generator: str
    physical_generator_natural_purpose: str
    physical_generator_campaign_role: str
    graphic_generator: Builder1GraphicGenerator
    series_generator: Builder1SeriesGenerator
    medium_participates: bool
    medium_role: str
    campaign_rationale: str
    ads: List[Builder1AdPlan] = field(default_factory=list)


def graphic_generator_to_dict(g: Builder1GraphicGenerator) -> Dict[str, Any]:
    return {
        "colorPalette": list(g.color_palette),
        "typography": {
            "headlineStyle": g.typography.headline_style,
            "sloganStyle": g.typography.slogan_style,
            "brandStyle": g.typography.brand_style,
        },
        "composition": {
            "grid": g.composition.grid,
            "visualArea": g.composition.visual_area,
            "copyArea": g.composition.copy_area,
            "alignment": g.composition.alignment,
            "sloganPlacement": g.composition.slogan_placement,
            "brandPlacement": g.composition.brand_placement,
        },
        "imageStyle": g.image_style,
        "spacing": g.spacing,
        "visualTreatment": g.visual_treatment,
        "backgroundTreatment": g.background_treatment,
    }


def series_generator_to_dict(s: Builder1SeriesGenerator) -> Dict[str, Any]:
    return {
        "type": s.type,
        "principle": s.principle,
        "progression": s.progression,
    }


def campaign_identity_to_dict(plan: Builder1SeriesPlan) -> Dict[str, Any]:
    return {
        "productNameResolved": plan.product_name_resolved,
        "detectedLanguage": plan.detected_language,
        "format": plan.format,
        "adCount": plan.ad_count,
        "strategicProblem": plan.strategic_problem,
        "strategicProblemEvidence": plan.strategic_problem_evidence,
        "relativeAdvantage": plan.relative_advantage,
        "problemAdvantageLink": plan.problem_advantage_link,
        "brandSlogan": plan.brand_slogan,
        "sloganDerivation": plan.slogan_derivation,
        "sloganAction": plan.slogan_action,
        "conceptualGenerator": plan.conceptual_generator,
        "conceptualGeneratorAction": plan.conceptual_generator_action,
        "physicalGenerator": plan.physical_generator,
        "physicalGeneratorNaturalPurpose": plan.physical_generator_natural_purpose,
        "physicalGeneratorCampaignRole": plan.physical_generator_campaign_role,
        "graphicGenerator": graphic_generator_to_dict(plan.graphic_generator),
        "seriesGenerator": series_generator_to_dict(plan.series_generator),
        "mediumParticipates": plan.medium_participates,
        "mediumRole": plan.medium_role,
        "campaignRationale": plan.campaign_rationale,
    }


def ad_plan_to_api_dict(ad: Builder1AdPlan, *, visual_prompt: str = "", image_base64: str = "") -> Dict[str, Any]:
    return {
        "index": ad.index,
        "variationLabel": ad.variation_label,
        "newContribution": ad.new_contribution,
        "physicalExecution": ad.physical_execution,
        "visualExecution": ad.visual_execution,
        "sceneDescription": ad.scene_description,
        "headline": ad.headline,
        "headlineNeededReason": ad.headline_needed_reason,
        "marketingText": ad.marketing_text,
        "visualPrompt": visual_prompt,
        "imageBase64": image_base64,
    }
