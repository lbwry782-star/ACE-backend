"""
Builder1 staged planning contracts — one focused prompt per stage.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from engine.builder1_plan_spec import AD_COUNT_MAX, AD_COUNT_MIN

EXPLORATION_LENSES = [
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
]

STAGE_STRATEGY_SCAN_SYSTEM = """
You are a Builder1 strategy explorer.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidates":[{"id":"S01","lens":"economic","strategicProblem":"...","relativeAdvantage":"...","briefSupport":"...","advantageSource":"explicit_brief","claimRisk":"low"}]}
Rules:
- Exactly 12 candidates with ids S01 through S12.
- Every candidate must be an object, never a string.
- advantageSource: explicit_brief | category_inference | brand_position | observable_product_mechanism
- claimRisk: low | medium | high
- Do not invent surveys, percentages, study names, or statistics.
- briefSupport must cite brief facts or general category reasoning only.
- Do not include slogans, generators, graphics, or ads.
""".strip()

STAGE_STRATEGY_SELECT_SYSTEM = """
You are a Builder1 strategy selector.
Return JSON only. Return exactly this object and no additional top-level keys:
{"selectedCandidateId":"S07","selectionReason":"...","strategyFamily":"...","scores":{"truth":8,"briefSupport":8,"relevance":8,"distinctiveness":7,"brandOwnership":8,"persuasiveStrength":8,"seriesPotential":8,"conceptualActionPotential":8}}
Rules:
- selectedCandidateId must be one of the provided candidate ids.
- Do not rewrite the selected problem or advantage.
- Scores are integers 1-10.
""".strip()

STAGE_CONCEPTUAL_SCAN_SYSTEM = """
You are a Builder1 conceptual-generator explorer.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidates":[{"id":"C01","generator":"...","action":"...","input":"...","transformation":"...","result":"...","whyItExpressesAdvantage":"...","seriesPotential":"..."}]}
Rules:
- Exactly 6 candidates with ids C01 through C06.
- Every candidate must be an object with all string fields non-empty.
- generator must define a repeatable action, not a mood, object, or abstract noun.
- Do not choose physical objects, slogans, colors, layouts, or ads.
""".strip()

STAGE_CONCEPTUAL_SELECT_SYSTEM = """
You are a Builder1 conceptual-generator selector.
Return JSON only. Return exactly this object and no additional top-level keys:
{"selectedCandidateId":"C04","selectionReason":"...","scores":{"advantageConnection":8,"actionClarity":8,"visualPower":8,"seriesPotential":8,"distinctiveness":7,"physicalIndependence":8}}
Rules:
- selectedCandidateId must be one of the provided ids.
- Do not invent a new generator.
- Scores are integers 1-10.
""".strip()

STAGE_BRAND_PHYSICAL_SYSTEM = """
You are a Builder1 brand and physical-system builder.
Return JSON only. Return exactly this object and no additional top-level keys:
{"productNameResolved":"...","brandSlogan":"...","sloganDerivation":"...","sloganAction":"...","physicalGenerator":"...","physicalGeneratorNaturalPurpose":"...","physicalGeneratorCampaignRole":"...","mediumParticipates":false,"mediumRole":"","campaignRationale":"..."}
Rules:
- Do not include graphic generator, series generator, ads, format, adCount, detectedLanguage, or strategy fields.
- mediumParticipates must be JSON boolean true or false, never a string.
- When mediumParticipates is false, mediumRole must be "".
- brandSlogan must be 1-6 words.
""".strip()

STAGE_GRAPHIC_SYSTEM_SYSTEM = """
You are a Builder1 graphic-system builder.
Return JSON only. Return the graphic generator object directly with no wrapper and no additional top-level keys:
{"palette":{"dominant":"#111111","secondary":"#EEEEEE","accent":"#FF5500","background":"#F5F5F5","text":"#222222"},"layoutTemplate":"visual_right_copy_left","headlinePlacement":"top_left","headlineAlignment":"right","headlineMaxWidthPercent":34,"brandBlockPlacement":"bottom_left","sloganPlacement":"bottom_left","copySafeArea":{"side":"left","widthPercent":38},"typographyStyle":"bold_geometric_sans","headlineScale":"large","brandScale":"small","sloganScale":"medium","imageStyle":"editorial_photography","backgroundTreatment":"solid","borderTreatment":"none","recurringGraphicDevice":"...","recurringGraphicDeviceRule":"...","shapeLanguage":"...","framingRule":"...","spacingRule":"..."}
Rules:
- All five palette colors required as #RRGGBB hex.
- Use only valid layout, placement, typography, image, background, and border enum values.
- recurringGraphicDevice must be visibly repeatable across ads.
- Do not return ads, slogan, physical generator, or strategy fields.
""".strip()

STAGE_SERIES_ADS_SYSTEM = f"""
You are a Builder1 series and ads builder.
Return JSON only. Return exactly this object and no additional top-level keys:
{{"seriesGenerator":{{"type":"...","principle":"...","progression":"..."}},"ads":[{{"index":1,"variationLabel":"...","newContribution":"...","conceptualExecution":"...","conceptualActionProof":"...","physicalExecution":"...","visualExecution":"...","sceneDescription":"...","headline":null,"headlineNeededReason":"...","marketingText":"..."}}]}}
Rules:
- seriesGenerator must be an object with type, principle, progression.
- ads must contain exactly the requested ad count ({AD_COUNT_MIN}-{AD_COUNT_MAX}).
- Each ad performs the same conceptual action with a different execution.
- Headlines must be null or very short (max 7 words).
- marketingText must be exactly 50 words in the server target language — one paragraph below the image, not inside it.
- marketingText must be written in the target language provided in the user prompt.
- Do not switch languages inside marketingText.
- Brand names, product names, URLs, numbers, and technical terms may remain in Latin letters when appropriate.
- Do not translate product or brand names unnecessarily.
- Do not invent unsupported claims.
- Do not return brand slogan, physical generator, or graphic generator.
""".strip()


def shuffled_exploration_lens_order() -> List[str]:
    import random

    lenses = list(EXPLORATION_LENSES)
    random.shuffle(lenses)
    return lenses


def build_strategy_scan_user_prompt(
    *,
    product_name: str,
    product_description: str,
    detected_language: str,
    lens_order: List[str],
    exploration_seed: str,
) -> str:
    return (
        f"Product name: {product_name or '(infer from description)'}\n"
        f"Product description: {product_description}\n"
        f"Language context: {detected_language}\n"
        f"Campaign exploration seed: {exploration_seed}\n"
        f"Lens order: {', '.join(lens_order)}\n"
        "Return exactly 12 strategy candidates as objects S01-S12."
    )


def build_strategy_scan_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the candidates array. Return exactly:\n"
        '{"candidates":[{"id":"S01","lens":"...","strategicProblem":"...","relativeAdvantage":"...",'
        '"briefSupport":"...","advantageSource":"explicit_brief","claimRisk":"low"}]}\n'
        f"Validation errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken output:\n{broken_json}\n"
        "Every candidate must be an object. Exactly 12 ids S01-S12."
    )


def build_strategy_select_user_prompt(candidates: List[Dict[str, Any]], exploration_seed: str) -> str:
    return (
        f"Campaign exploration seed: {exploration_seed}\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        "Select the strongest candidate by id. Do not rewrite its problem or advantage."
    )


def build_conceptual_scan_user_prompt(
    *,
    product_description: str,
    strategic_problem: str,
    relative_advantage: str,
    exploration_seed: str,
) -> str:
    return (
        f"Brief: {product_description}\n"
        f"Selected strategic problem: {strategic_problem}\n"
        f"Selected relative advantage: {relative_advantage}\n"
        f"Exploration seed: {exploration_seed}\n"
        "Return exactly 6 conceptual-generator candidates C01-C06 as objects."
    )


def build_conceptual_scan_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the candidates array. Return exactly:\n"
        '{"candidates":[{"id":"C01","generator":"...","action":"...","input":"...",'
        '"transformation":"...","result":"...","whyItExpressesAdvantage":"...","seriesPotential":"..."}]}\n'
        f"Errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_conceptual_select_user_prompt(candidates: List[Dict[str, Any]]) -> str:
    return (
        f"Conceptual candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        "Select one candidate by id."
    )


def build_brand_physical_user_prompt(
    *,
    product_name: str,
    product_description: str,
    detected_language: str,
    format_value: str,
    strategic_problem: str,
    relative_advantage: str,
    conceptual: Dict[str, str],
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> str:
    guidelines = ""
    if brand_guidelines:
        guidelines = "\nBrand guidelines:\n" + json.dumps(brand_guidelines, ensure_ascii=False, indent=2)
    return (
        f"Product name: {product_name or '(infer)'}\n"
        f"Description: {product_description}\n"
        f"Language context: {detected_language}\n"
        f"Format context: {format_value}\n"
        f"Fixed strategic problem: {strategic_problem}\n"
        f"Fixed relative advantage: {relative_advantage}\n"
        f"Fixed conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        "Return brand slogan and physical-generator system only."
        f"{guidelines}"
    )


def build_brand_physical_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the brand/physical JSON object. Return exactly:\n"
        '{"productNameResolved":"...","brandSlogan":"...","sloganDerivation":"...","sloganAction":"...",'
        '"physicalGenerator":"...","physicalGeneratorNaturalPurpose":"...","physicalGeneratorCampaignRole":"...",'
        '"mediumParticipates":false,"mediumRole":"","campaignRationale":"..."}\n'
        f"Missing or invalid fields:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_graphic_system_user_prompt(
    *,
    product_description: str,
    detected_language: str,
    relative_advantage: str,
    conceptual: Dict[str, str],
    brand_physical: Dict[str, Any],
    format_value: str,
) -> str:
    return (
        f"Brief: {product_description}\n"
        f"Language: {detected_language}\n"
        f"Relative advantage: {relative_advantage}\n"
        f"Conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        f"Brand/physical system:\n{json.dumps(brand_physical, ensure_ascii=False, indent=2)}\n"
        f"Format: {format_value}\n"
        "Return the graphic generator object directly."
    )


def build_graphic_system_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the graphic generator object. Return exactly the graphic object with all palette colors, "
        "layout/placement enums, copySafeArea, typography scales, and recurring device rules.\n"
        f"Missing or invalid fields:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_series_ads_user_prompt(
    *,
    ad_count: int,
    format_value: str,
    detected_language: str,
    strategic_problem: str,
    relative_advantage: str,
    conceptual: Dict[str, str],
    brand_physical: Dict[str, Any],
    graphic_generator: Dict[str, Any],
) -> str:
    indexes = ", ".join(str(i) for i in range(1, ad_count + 1))
    lang_name = "Hebrew" if detected_language == "he" else "English"
    return (
        f"Required ad count: {ad_count}\n"
        f"Required ad indexes: {indexes}\n"
        f"Format: {format_value}\n"
        f"TARGET LANGUAGE FOR ALL MARKETING TEXT: {lang_name} ({detected_language})\n"
        "Every marketingText must:\n"
        "- contain exactly 50 words\n"
        "- be written in the target language\n"
        "- use one coherent paragraph\n"
        "- not switch into another language\n"
        "- not contain headings, labels, bullets, or hashtags\n"
        "- not translate the product or brand name unnecessarily\n"
        "- not invent unsupported claims\n"
        f"Strategic problem: {strategic_problem}\n"
        f"Relative advantage: {relative_advantage}\n"
        f"Conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        f"Brand/physical:\n{json.dumps(brand_physical, ensure_ascii=False, indent=2)}\n"
        f"Graphic system:\n{json.dumps(graphic_generator, ensure_ascii=False, indent=2)}\n"
        f"Return seriesGenerator and exactly {ad_count} ads obeying the graphic system."
    )


def build_series_ads_repair_prompt(*, broken_json: str, reasons: List[str], ad_count: int) -> str:
    return (
        "Repair ONLY seriesGenerator and ads. Return exactly:\n"
        '{"seriesGenerator":{"type":"...","principle":"...","progression":"..."},'
        '"ads":[{"index":1,"variationLabel":"...","newContribution":"...",'
        '"conceptualExecution":"...","conceptualActionProof":"...","physicalExecution":"...",'
        '"visualExecution":"...","sceneDescription":"...","headline":null,'
        '"headlineNeededReason":"...","marketingText":"..."}]}\n'
        f"Required ad count: {ad_count}\n"
        f"Errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_final_campaign_repair_prompt(
    *,
    broken_json: str,
    reasons: List[str],
    ad_count: int,
    strategic_problem: str,
    relative_advantage: str,
    conceptual: Dict[str, str],
) -> str:
    return build_series_ads_repair_prompt(broken_json=broken_json, reasons=reasons, ad_count=ad_count)


STAGE_FINAL_CAMPAIGN_SYSTEM = STAGE_BRAND_PHYSICAL_SYSTEM  # legacy alias

