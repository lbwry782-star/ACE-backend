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

STAGE_FINAL_CAMPAIGN_SYSTEM = f"""
You are a Builder1 campaign constructor.
The strategic problem, relative advantage, and conceptual generator are already fixed.
Return JSON only. Return exactly the final creative plan object and no additional top-level keys.
Do NOT include: format, adCount, detectedLanguage, strategicProblem, relativeAdvantage, conceptualGenerator fields, or candidate scans.
Build exactly the requested number of ads ({AD_COUNT_MIN}-{AD_COUNT_MAX}).
Include: productNameResolved, brandSlogan (1-6 words), sloganDerivation, sloganAction,
physicalGenerator fields, graphicGenerator with exact hex palette and concrete layout enums,
seriesGenerator, mediumParticipates, mediumRole, campaignRationale, and ads[].
Each ad needs: index, variationLabel, newContribution, conceptualExecution, conceptualActionProof,
physicalExecution, visualExecution, sceneDescription, headline (null or short string), headlineNeededReason, marketingText.
Images will render brand name, slogan, and optional headline — keep copy very short.
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


def build_final_campaign_user_prompt(
    *,
    product_name: str,
    product_description: str,
    ad_count: int,
    format_value: str,
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    advantage_source: str,
    conceptual: Dict[str, str],
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> str:
    guidelines = ""
    if brand_guidelines:
        guidelines = "\nBrand guidelines:\n" + json.dumps(brand_guidelines, ensure_ascii=False, indent=2)
    return (
        f"Product name: {product_name or '(infer)'}\n"
        f"Description: {product_description}\n"
        f"Format (for image generation context only): {format_value}\n"
        f"Ad count: {ad_count}\n"
        f"Fixed strategic problem: {strategic_problem}\n"
        f"Fixed relative advantage: {relative_advantage}\n"
        f"Brief support: {brief_support}\n"
        f"Advantage source: {advantage_source}\n"
        f"Fixed conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        f"Build the final campaign creative plan with exactly {ad_count} ads."
        f"{guidelines}"
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
    return (
        f"Repair the final campaign creative JSON only.\n"
        f"Keep fixed strategy and conceptual generator unchanged.\n"
        f"Ad count must be exactly {ad_count}.\n"
        f"Fixed problem: {strategic_problem}\n"
        f"Fixed advantage: {relative_advantage}\n"
        f"Fixed conceptual:\n{json.dumps(conceptual, ensure_ascii=False)}\n"
        f"Errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )
