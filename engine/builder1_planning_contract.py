"""
Builder1 campaign-series planning contract (active production prompts).
"""
from __future__ import annotations

import json
import random
import uuid
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

BUILDER1_PLANNING_SYSTEM_PROMPT = f"""
You are Builder1, a senior advertising strategist and creative director.

Produce ONE coherent CAMPAIGN with exactly the requested adCount (between {AD_COUNT_MIN} and {AD_COUNT_MAX}).

METHODOLOGY ONLY — no memory of prior campaigns.

The generated images will be COMPLETE FINAL ADVERTISEMENTS with typeset copy inside the image (brand name, brand slogan, optional headline).

ORDERED PIPELINE:

STEP 1 — STRATEGIC PROBLEM
Output: strategicProblem, strategicProblemEvidence

STEP 2 — STRATEGIC CANDIDATE SCAN (internal, mandatory)
Generate at least 12 genuinely different problem–advantage candidates.
Reject paraphrases of the same transparency/ROI/dashboard idea.
Group candidates into at least 5 materially different strategy families.
Score families on truth, brief support, relevance, distinctiveness, ownership, persuasive strength, series potential, conceptual-action potential.
Identify top 3 quality candidates.
If several are close, use campaignExplorationSeed to break ties — do not always pick the most conventional first candidate.
Output strategyCandidateScan with candidates[], families[], topCandidates[].
Also output: strategyFamily, strategyScore, campaignExplorationSeed, selectionReason

STEP 3 — RELATIVE ADVANTAGE
Do NOT invent dashboards, live reporting, guaranteed results, sales attribution, transparency systems, speed guarantees, automation, or personal service unless the brief supports them.
Positioning inference is allowed; invented product features are not.
Output: relativeAdvantage, relativeAdvantageSource, relativeAdvantageBriefSupport, relativeAdvantageClaimRisk, problemAdvantageLink

STEP 4 — BRAND SLOGAN (1–4 words preferred, max 6, extremely short for image typesetting)

STEP 5 — CONCEPTUAL GENERATOR SCAN (internal, before physical generator)
Generate at least 6 distinct conceptual-generator candidates.
Each must define a repeatable action/transformation: Take X → perform Y → produce Z.
Reject emotions, abstract values, objects, visual styles, categories, or physical generators.
Score on advantage relationship, action clarity, visual power, serial potential, distinctiveness.
Select ONE conceptual generator BEFORE choosing any physical generator.
Output conceptualGeneratorScan with candidates[].

STEP 6 — SELECTED CONCEPTUAL GENERATOR
Output: conceptualGenerator, conceptualGeneratorAction, conceptualGeneratorInput, conceptualGeneratorTransformation, conceptualGeneratorResult, conceptualGeneratorWhyItExpressesAdvantage
Must NOT equal physicalGenerator.

STEP 7 — PHYSICAL GENERATOR (derived FROM concept, not before it)
Output: physicalGenerator, physicalGeneratorNaturalPurpose, physicalGeneratorCampaignRole

STEP 8 — GRAPHIC GENERATOR (dominant visible art direction)
Exact hex palette, layout template, typography style/scales, copy zone, recurring visible device, shape language, spacing rule.
Same identity in EVERY ad. Image model renders final copy inside the ad.

STEP 9 — SERIES + exactly adCount ads
Each ad: conceptualExecution, conceptualActionProof, optional headline (null or 1–5 words preferred, max 7).
Marketing text is NOT rendered in the image.

Return JSON only.
""".strip()


def new_campaign_exploration_seed() -> str:
    return str(uuid.uuid4())


def shuffled_exploration_lens_order() -> List[str]:
    lenses = list(EXPLORATION_LENSES)
    random.shuffle(lenses)
    return lenses


def build_builder1_planning_user_prompt(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    brand_guidelines: Optional[Dict[str, Any]] = None,
    exploration_lens_order: Optional[List[str]] = None,
    campaign_exploration_seed: Optional[str] = None,
) -> str:
    name_line = product_name.strip() or "(not provided — infer from description)"
    lenses = exploration_lens_order or shuffled_exploration_lens_order()
    seed = campaign_exploration_seed or new_campaign_exploration_seed()
    lens_line = ", ".join(lenses)
    guidelines_block = ""
    if brand_guidelines:
        guidelines_block = (
            "\nBrand guidelines (override inferred visual identity where specified):\n"
            + json.dumps(brand_guidelines, ensure_ascii=False, indent=2)
        )
    return f"""
Product name: {name_line}
Product description: {product_description.strip()}
Format: {format_value}
Requested ad count: {ad_count}
Campaign exploration seed: {seed}
Exploration lens order: {lens_line}

Return exactly ONE campaign with exactly {ad_count} ads in ads[].
Use campaignExplorationSeed exactly as provided when breaking ties among strong candidates.

Required JSON shape includes:
- strategyCandidateScan with >=12 candidates and >=5 families
- conceptualGeneratorScan with >=6 candidates BEFORE physicalGenerator
- strategyFamily, strategyScore, campaignExplorationSeed, selectionReason
- graphicGenerator with palette, layoutTemplate, typographyStyle, headlineScale, brandScale, sloganScale, copySafeArea, recurringGraphicDevice, recurringGraphicDeviceRule, shapeLanguage, framingRule, spacingRule
- ads[] with conceptualExecution, conceptualActionProof, headline (null or short string), marketingText
{guidelines_block}
""".strip()


def build_builder1_series_repair_user_prompt(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    broken_plan_json: str,
    rejection_reasons: list[str],
    brand_guidelines: Optional[Dict[str, Any]] = None,
    exploration_lens_order: Optional[List[str]] = None,
    campaign_exploration_seed: Optional[str] = None,
) -> str:
    reasons = "\n".join(f"- {r}" for r in rejection_reasons)
    base = build_builder1_planning_user_prompt(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        ad_count=ad_count,
        brand_guidelines=brand_guidelines,
        exploration_lens_order=exploration_lens_order,
        campaign_exploration_seed=campaign_exploration_seed,
    )
    return f"""
{base}

The previous campaign plan JSON failed validation:
{reasons}

Broken plan:
{broken_plan_json}

Repair the plan to satisfy ALL validation rules. Return corrected JSON only.
""".strip()


def build_builder1_strategy_repair_user_prompt(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    ad_count: int,
    broken_plan_json: str,
    judge_reason_codes: list[str],
    brand_guidelines: Optional[Dict[str, Any]] = None,
    exploration_lens_order: Optional[List[str]] = None,
    campaign_exploration_seed: Optional[str] = None,
) -> str:
    reasons = "\n".join(f"- {c}" for c in judge_reason_codes)
    base = build_builder1_planning_user_prompt(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        ad_count=ad_count,
        brand_guidelines=brand_guidelines,
        exploration_lens_order=exploration_lens_order,
        campaign_exploration_seed=campaign_exploration_seed,
    )
    return f"""
{base}

The strategy judge rejected the previous plan:
{reasons}

Rejected plan:
{broken_plan_json}

Repair strategy, advantage support, conceptual action, and graphic concreteness. Return corrected JSON only.
""".strip()
