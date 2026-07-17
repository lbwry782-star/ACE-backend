"""
Builder1 campaign-series planning contract (active production prompts).
"""
from __future__ import annotations

import json
import random
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

METHODOLOGY ONLY — no memory of prior campaigns, slogans, or executions.

ORDERED PIPELINE — complete every stage before the next.

STEP 1 — STRATEGIC PROBLEM
Output: strategicProblem, strategicProblemEvidence

STEP 2 — STRATEGIC CANDIDATE SCAN (internal)
Before choosing the final advantage, scan at least 10 distinct problem–advantage pairs.
Each candidate must use a different exploration lens.
Do not repeat transparency/ROI/dashboard paraphrases.
Include strategyCandidateScan with candidates[] of {{lens, problem, advantage, briefSupport}}.
This scan is internal methodology — keep it in JSON but it will not be shown to users.

STEP 3 — RELATIVE ADVANTAGE
Choose the strongest supported advantage from the scan.
Do NOT invent product capabilities absent from the brief (dashboards, live reporting, guaranteed ROI, measurable sales lifts).
Output: relativeAdvantage, relativeAdvantageSource, relativeAdvantageBriefSupport, relativeAdvantageClaimRisk, problemAdvantageLink
relativeAdvantageSource must be one of: explicit_brief, category_inference, brand_position, observable_product_mechanism
Generic transparency, quality, service, innovation, trust, or results alone are insufficient.

STEP 4 — BRAND SLOGAN (shared, 1–6 words max)
Output: brandSlogan, sloganDerivation, sloganAction

STEP 5 — CONCEPTUAL GENERATOR (repeatable ACTION, not a theme)
Must be expressible as: Take X → perform Y → produce Z
Output: conceptualGenerator, conceptualGeneratorAction, conceptualGeneratorInput, conceptualGeneratorTransformation, conceptualGeneratorResult, conceptualGeneratorWhyItExpressesAdvantage
Reject vague themes like transparency, confidence, growth, results, visibility, connection.
Must NOT equal the physicalGenerator name.

STEP 6 — PHYSICAL GENERATOR
Embodies the conceptual action. Output: physicalGenerator, physicalGeneratorNaturalPurpose, physicalGeneratorCampaignRole

STEP 7 — GRAPHIC GENERATOR (concrete, machine-renderable)
Required exact hex palette roles, layout template, copy-safe area, headline placement inside the ad canvas, recurring visible device.
Use enums where specified. Same identity in EVERY ad.

STEP 8 — SERIES GENERATOR + exactly adCount ads
Each ad must include conceptualExecution and conceptualActionProof showing how it performs the SAME conceptual action differently.
No ad may only swap objects without performing the shared transformation.

HEADLINE POLICY: optional per ad (null allowed, max 7 words). Headline renders in Frontend overlay inside the ad canvas — never in the image.

Return JSON only. No markdown fences.
""".strip()


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
) -> str:
    name_line = product_name.strip() or "(not provided — infer from description)"
    lenses = exploration_lens_order or shuffled_exploration_lens_order()
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
Exploration lens order for this campaign (use in candidate scan): {lens_line}

Return exactly ONE campaign with exactly {ad_count} ads in ads[].

Required JSON shape:
{{
  "productNameResolved": "...",
  "detectedLanguage": "he|en|...",
  "format": "{format_value}",
  "adCount": {ad_count},
  "strategyCandidateScan": {{
    "candidates": [
      {{"lens": "economic", "problem": "...", "advantage": "...", "briefSupport": "..."}}
    ]
  }},
  "strategicProblem": "...",
  "strategicProblemEvidence": "...",
  "relativeAdvantage": "...",
  "relativeAdvantageSource": "explicit_brief|category_inference|brand_position|observable_product_mechanism",
  "relativeAdvantageBriefSupport": "...",
  "relativeAdvantageClaimRisk": "...",
  "problemAdvantageLink": "...",
  "brandSlogan": "...",
  "sloganDerivation": "...",
  "sloganAction": "...",
  "conceptualGenerator": "...",
  "conceptualGeneratorAction": "...",
  "conceptualGeneratorInput": "...",
  "conceptualGeneratorTransformation": "...",
  "conceptualGeneratorResult": "...",
  "conceptualGeneratorWhyItExpressesAdvantage": "...",
  "physicalGenerator": "...",
  "physicalGeneratorNaturalPurpose": "...",
  "physicalGeneratorCampaignRole": "...",
  "graphicGenerator": {{
    "palette": {{"dominant": "#000000", "secondary": "#FFFFFF", "accent": "#FF0000", "background": "#F5F5F5", "text": "#111111"}},
    "layoutTemplate": "visual_right_copy_left",
    "headlinePlacement": "top_left",
    "headlineAlignment": "right",
    "headlineMaxWidthPercent": 34,
    "headlineColor": "#111111",
    "headlineTreatment": "plain",
    "brandBlockPlacement": "bottom_left",
    "sloganPlacement": "bottom_left",
    "copySafeArea": {{"side": "left", "widthPercent": 38}},
    "imageStyle": "editorial_photography",
    "backgroundTreatment": "solid",
    "borderTreatment": "none",
    "recurringGraphicDevice": "...",
    "recurringGraphicDeviceRule": "...",
    "framingRule": "..."
  }},
  "seriesGenerator": {{"type": "...", "principle": "...", "progression": "..."}},
  "mediumParticipates": false,
  "mediumRole": "",
  "campaignRationale": "...",
  "campaignSelfCheck": {{}},
  "ads": [
    {{
      "index": 1,
      "variationLabel": "...",
      "newContribution": "...",
      "physicalExecution": "...",
      "visualExecution": "...",
      "sceneDescription": "...",
      "conceptualExecution": "...",
      "conceptualActionProof": "...",
      "headline": null,
      "headlineNeededReason": "...",
      "marketingText": "..."
    }}
  ]
}}
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
) -> str:
    reasons = "\n".join(f"- {r}" for r in rejection_reasons)
    base = build_builder1_planning_user_prompt(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        ad_count=ad_count,
        brand_guidelines=brand_guidelines,
        exploration_lens_order=exploration_lens_order,
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
) -> str:
    reasons = "\n".join(f"- {c}" for c in judge_reason_codes)
    base = build_builder1_planning_user_prompt(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        ad_count=ad_count,
        brand_guidelines=brand_guidelines,
        exploration_lens_order=exploration_lens_order,
    )
    return f"""
{base}

The strategy judge rejected the previous plan:
{reasons}

Rejected plan:
{broken_plan_json}

Repair strategy, advantage support, conceptual action, and graphic concreteness. Return corrected JSON only.
""".strip()
