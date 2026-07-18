"""
Builder1 staged planning contracts — one focused prompt per stage.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from engine.builder1_plan_spec import AD_COUNT_MAX, AD_COUNT_MIN
from engine.builder1_no_logo import BUILDER1_NO_LOGO_PLANNING_RULE, brand_guidelines_for_prompt

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

STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM = """
You are a Builder1 product-name resolver for a digital advertising agent.
Return JSON only. Return exactly this object and no additional top-level keys:
{"productNameResolved":"..."}
Rules:
- Invent exactly one concise brand-like product or brand name from the description.
- Do not return strategy, relative advantage, slogan, headline, generators, or ads.
- Do not copy the full product description verbatim.
- Do not return a generic category label such as Shoe Store, Restaurant, חנות נעליים, or מסעדה.
- The name must be distinctive, readable, advertising-ready, and free of unsupported claims.
""".strip()

STAGE_STRATEGY_SCAN_SYSTEM = """
You are a Builder1 strategy explorer for a digital advertising agent.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidates":[{"id":"S01","lens":"economic","strategicProblem":"...","relativeAdvantage":"...","briefSupport":"...","advantageSource":"explicit_brief","claimRisk":"low","campaignExecutableNow":true,"requiresClientConsultation":false,"clientActionLevel":"none","implementationCostLevel":"none","simpleStrategicAction":null}]}
Rules:
- Exactly 12 candidates with ids S01 through S12.
- Every candidate must be an object, never a string.
- advantageSource: explicit_brief | category_inference | brand_position | observable_product_mechanism
- claimRisk: low | medium | high
- campaignExecutableNow: true only when the campaign can run immediately from the brief and current product.
- requiresClientConsultation: false unless the strategy needs workshops, interviews, or consulting to define the offer.
- clientActionLevel: none | simple_optional | complex_required
- implementationCostLevel: none | negligible | material
- simpleStrategicAction: null or one short optional communication action only.
- Do not propose business transformation, new products, new services, pricing changes, guarantees, dashboards, training, or material client investment.
- The relative advantage must already exist or be a perceptual advertising reframing of existing facts.
- briefSupport (brief grounding only): a direct brief fact, faithful brief paraphrase, observable product property, or clearly labeled category inference without empirical claims.
- Do not request, cite, or invent market evidence, studies, statistics, surveys, percentages, interview counts, reports, or factual capabilities not in the brief.
- Do not use wording such as evidence, research, market proof, data shows, or surveys show unless the user explicitly supplied that information.
- clientActionLevel none requires simpleStrategicAction null.
- clientActionLevel simple_optional requires a short non-empty simpleStrategicAction that is optional, immediately executable, and needs no client consultation, operational change, new product, new technology, staff training, or material cost.
- Do not include slogans, generators, graphics, or ads.
""".strip()

STAGE_STRATEGY_SELECT_SYSTEM = """
You are a Builder1 strategy reviewer and selector for a digital advertising agent.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidateReviews":[{"candidateId":"S01","groundedInBrief":true,"advantageCurrentlyTrue":true,"executableNow":true,"requiresMaterialInvestment":false,"requiresClientConsultation":false,"requiresBusinessTransformation":false,"brandOwnable":true,"categoryRelevant":true,"eligible":true,"rejectionCodes":[]}],"selectedCandidateId":"S07","selectionReason":"...","strategyFamily":"...","scores":{"truth":8,"briefSupport":8,"advertisingExecutability":9,"noConsultationDependency":9,"noMaterialImplementationCost":9,"relevance":8,"distinctiveness":7,"brandOwnership":8,"persuasiveStrength":8,"seriesPotential":8,"conceptualActionPotential":8}}
Rules:
- Review every supplied candidate id exactly once in candidateReviews.
- Do not rewrite any candidate fields.
- eligible=true requires rejectionCodes to be [].
- eligible=false requires at least one rejection code from:
  advantage_not_currently_true, relative_advantage_not_currently_true,
  material_client_investment_required, client_consultation_required,
  business_transformation_required, unsupported_future_capability,
  unsupported_evidence_claim, strategy_not_brand_ownable, category_relevance_patched,
  campaign_transferable_to_competitor.
- selectedCandidateId must refer to a candidate with eligible=true.
- Do not select a candidate requiring material investment, client consultation, or business transformation.
- Do not select an advantage that becomes true only in the future.
- Scores are integers 1-10.
""".strip()


STAGE_STRATEGY_STAGE_SYSTEM = """
You are a Builder1 strategy explorer and selector for a digital advertising agent.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidates":[{"id":"S01","lens":"economic","strategicProblem":"...","relativeAdvantage":"...","briefSupport":"...","advantageSource":"explicit_brief","claimRisk":"low","campaignExecutableNow":true,"requiresClientConsultation":false,"clientActionLevel":"none","implementationCostLevel":"none","simpleStrategicAction":null}],"evaluations":[{"candidateId":"S01","groundedInBrief":true,"advantageCurrentlyTrue":true,"executableNow":true,"requiresMaterialInvestment":false,"requiresClientConsultation":false,"requiresBusinessTransformation":false,"brandOwnable":true,"categoryRelevant":true,"eligible":true,"rejectionCodes":[]}],"selectedCandidateId":"S01","selectionReason":"..."}
Internal order:
1. Understand the real advertising/perception problem.
2. Generate exactly 12 serious strategic candidates S01-S12.
3. Evaluate every candidate once in evaluations.
4. Mark each candidate eligible or ineligible.
5. Select one eligible candidate by id.
Rules for candidates:
- Exactly 12 candidates with ids S01 through S12.
- Every candidate must be an object, never a string.
- advantageSource: explicit_brief | category_inference | brand_position | observable_product_mechanism
- claimRisk: low | medium | high
- campaignExecutableNow: true only when the campaign can run immediately from the brief and current product.
- requiresClientConsultation: false unless the strategy needs workshops, interviews, or consulting to define the offer.
- clientActionLevel: none | simple_optional | complex_required
- implementationCostLevel: none | negligible | material
- simpleStrategicAction: null or one short optional communication action only.
- Do not propose business transformation, new products, new services, pricing changes, guarantees, dashboards, training, or material client investment.
- The relative advantage must already exist or be a perceptual advertising reframing of existing facts.
- briefSupport may contain only a direct brief fact, faithful brief paraphrase, observable product property, or clearly labeled category inference without empirical claims.
- Do not request, cite, or invent market evidence, studies, statistics, surveys, percentages, interview counts, reports, or factual capabilities not in the brief.
- clientActionLevel none requires simpleStrategicAction null.
- Do not include slogans, generators, graphics, or ads.
Rules for evaluations:
- Exactly one evaluation per candidate id.
- Do not rewrite candidate fields.
- eligible=true requires rejectionCodes [].
- eligible=false requires at least one rejection code from:
  advantage_not_currently_true, relative_advantage_not_currently_true,
  material_client_investment_required, client_consultation_required,
  business_transformation_required, unsupported_future_capability,
  unsupported_evidence_claim, strategy_not_brand_ownable, category_relevance_patched,
  campaign_transferable_to_competitor.
- selectedCandidateId must refer to an eligible candidate.
- Do not use Creator, Judge, or tournament roles.
""".strip()


STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM = """
You are a Builder1 strategy candidate repair assistant.
Return JSON only. Return exactly this object and no additional top-level keys:
{"replacements":[{"id":"S07","lens":"economic","strategicProblem":"...","relativeAdvantage":"...","briefSupport":"...","advantageSource":"explicit_brief","claimRisk":"low","campaignExecutableNow":true,"requiresClientConsultation":false,"clientActionLevel":"none","implementationCostLevel":"none","simpleStrategicAction":null}]}
Rules:
- Replace ONLY the requested candidate ids.
- briefSupport may contain only a direct brief fact, faithful brief paraphrase, or clearly labeled category inference.
- Do not invent percentages, studies, surveys, dates, interview counts, reports, or factual capabilities.
- Do not request or cite external market evidence.
- clientActionLevel none requires simpleStrategicAction null.
- clientActionLevel simple_optional requires a short non-empty simpleStrategicAction.
- complex_required and material implementation cost are not allowed.
""".strip()

STAGE_SLOGAN_SCAN_SYSTEM = f"""
You are a Builder1 brand-slogan explorer.
Return JSON only. Return exactly this object and no additional top-level keys:
{{"candidates":[{{"id":"L01","brandSlogan":"...","derivationFromAdvantage":"...","impliedAction":"...","whyOwnable":"...","whyNaturalInLanguage":"...","competitorTransferRisk":"low","campaignGenerativePower":"..."}}]}}
Rules:
- Exactly 6 candidates with ids L01 through L06.
- Every candidate must be an object with all string fields non-empty.
- competitorTransferRisk must be low, medium, or high.
- The slogan is a linguistic distillation of the selected relative advantage — not decorative copy, not a generic quality claim, not a product description, not an ad headline.
- Prefer one to four words when natural; allow slightly longer only when brevity would damage clarity.
- Each slogan must imply a visual or conceptual action and support several distinct advertisements.
- Do not invent capabilities that require future client implementation.
- Do not reuse generic slogans that could belong to many unrelated brands.
- Do not include conceptual generators, physical objects, graphics, or ads.
- {BUILDER1_NO_LOGO_PLANNING_RULE}
""".strip()

STAGE_SLOGAN_STAGE_SYSTEM = f"""
You are a Builder1 brand-slogan explorer and selector.
Return JSON only. Return exactly this object and no additional top-level keys:
{{"candidates":[{{"id":"L01","brandSlogan":"...","derivationFromAdvantage":"...","impliedAction":"...","whyOwnable":"...","whyNaturalInLanguage":"...","competitorTransferRisk":"low","campaignGenerativePower":"..."}}],"evaluations":[{{"candidateId":"L01","derivedFromAdvantage":true,"naturalInLanguage":true,"credible":true,"ownable":true,"impliedActionValid":true,"campaignGenerative":true,"eligible":true,"rejectionCodes":[]}}],"selectedCandidateId":"L01","selectionReason":"..."}}
Internal order:
1. Generate exactly six serious brand-slogan candidates from the fixed relative advantage.
2. Compare and evaluate each candidate once in evaluations.
3. Choose the strongest slogan and commit to it via selectedCandidateId.
4. Explain in selectionReason how the chosen slogan distills the relative advantage.
5. Define the implied action the slogan requires.
Rules:
- Exactly 6 candidates with ids L01 through L06.
- The slogan is a direct linguistic distillation of the relative advantage — not decorative copy, not a generic quality claim, not a product description, not an ad headline.
- Prefer a concise slogan, usually one to four words, when natural.
- A longer slogan is allowed when brevity would damage meaning, clarity, distinctiveness, memorability, or natural language.
- Do not obey a mechanical word maximum.
- A strong relative advantage may naturally yield a simple slogan; do not force clever copy.
- Select the slogan before receiving or creating conceptual or visual ideas.
- Product Name is separate from brandSlogan; do not include the product name inside brandSlogan.
- Semantic derivation does not require repeating the same words as the relative advantage.
- Do not include conceptual generators, physical objects, graphics, or ads.
- Do not use Creator, Judge, or tournament roles.
- {BUILDER1_NO_LOGO_PLANNING_RULE}
""".strip()

STAGE_SLOGAN_SELECT_SYSTEM = """
You are a Builder1 brand-slogan selector.
Return JSON only. Return exactly this object and no additional top-level keys:
{"selectedCandidateId":"L04","selectionReason":"...","scores":{"directAdvantageExpression":8,"naturalness":8,"memorability":7,"credibility":9,"brandOwnership":8,"competitorTransferResistance":8,"actionClarity":8,"campaignGenerativePower":9}}
Rules:
- selectedCandidateId must be one of the provided ids.
- Do not rewrite the selected candidate.
- Prioritize direct expression of the relative advantage, credibility, natural language, brand ownership, and clear implied action over cleverness.
- Scores are integers 1-10.
""".strip()

STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM = """
You are a Builder1 slogan quality reviewer.
Return JSON only. Return exactly this object and no additional top-level keys:
{"reviews":[{"candidateId":"L01","derivedFromAdvantage":true,"naturalInLanguage":true,"credible":true,"ownable":true,"impliedActionValid":true,"campaignGenerative":true,"eligible":true,"rejectionCodes":[]}]}
Rules:
- Review every supplied candidate id exactly once.
- Do not rewrite slogans or add candidates.
- The selected relative advantage remains authoritative.
- Semantic derivation does not require repeating the same words as the advantage sentence.
- eligible=true requires rejectionCodes to be [].
- eligible=false requires at least one rejection code from:
  slogan_not_derived_from_advantage, slogan_generic, slogan_descriptive_only, slogan_not_ownable,
  slogan_not_credible, slogan_no_implied_action, slogan_not_campaign_generative,
  slogan_requires_future_capability, slogan_wrong_language, slogan_invalid_structure
""".strip()

STAGE_SLOGAN_CANDIDATE_REPAIR_SYSTEM = """
You are a Builder1 slogan candidate repair agent.
Return JSON only. Return exactly this object and no additional top-level keys:
{"replacements":[{"id":"L03","brandSlogan":"...","derivationFromAdvantage":"...","impliedAction":"...","whyOwnable":"...","whyNaturalInLanguage":"...","competitorTransferRisk":"low","campaignGenerativePower":"..."}]}
Rules:
- Replace ONLY the requested candidate ids.
- Do not include ids that were not requested.
- competitorTransferRisk must be low, medium, or high.
- Every string field must be non-empty.
- Preserve the selected relative advantage as the source of meaning.
""".strip()

STAGE_CONCEPTUAL_SCAN_SYSTEM = """
You are a Builder1 conceptual-generator explorer.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidates":[{"id":"C01","generator":"...","action":"...","input":"...","transformation":"...","result":"...","whyItExpressesSlogan":"...","whyItExpressesAdvantage":"...","seriesPotential":"...","brandOwnershipPotential":"..."}]}
Rules:
- Exactly 6 candidates with ids C01 through C06.
- Every candidate must be an object with all string fields non-empty.
- The conceptual generator must answer: what action or transformation makes the selected brand slogan visible?
- Derive every candidate from the fixed brand slogan and its implied action — not from random objects or attractive scenes.
- generator must define a repeatable action, not a mood, object, or abstract noun.
- Do not choose the physical generator in this stage.
- Do not require client operational change, new products, pricing, or material investment.
- Do not choose slogans, colors, layouts, or ads.
""".strip()

STAGE_CONCEPTUAL_STAGE_SYSTEM = """
You are a Builder1 conceptual-generator explorer and selector.
Return JSON only. Return exactly this object and no additional top-level keys:
{"candidates":[{"id":"C01","generator":"...","action":"...","input":"...","transformation":"...","result":"...","whyItExpressesSlogan":"...","whyItExpressesAdvantage":"...","seriesPotential":"...","brandOwnershipPotential":"..."}],"evaluations":[{"candidateId":"C01","derivedFromSelectedSloganAction":true,"expressesRelativeAdvantage":true,"visuallyClear":true,"seriesGenerative":true,"brandOwnable":true,"categoryRelevant":true,"executableByImageModel":true,"eligible":true,"rejectionCodes":[]}],"selectedCandidateId":"C01","selectionReason":"..."}
Internal order:
1. Generate conceptual-generator candidates from the fixed implied action.
2. Evaluate each candidate once in evaluations.
3. Select one eligible conceptual generator by id.
Rules:
- Exactly 6 candidates with ids C01 through C06.
- Derive every candidate from the fixed brand slogan and its implied action.
- generator must define a repeatable action, not a mood, object, or abstract noun.
- Do not choose the physical generator, graphic system, or ads.
- Do not rewrite the slogan.
- Do not use Creator, Judge, or tournament roles.
""".strip()

STAGE_CONCEPTUAL_SELECT_SYSTEM = """
You are a Builder1 conceptual-generator selector.
Return JSON only. Return exactly this object and no additional top-level keys:
{"selectedCandidateId":"C04","selectionReason":"...","scores":{"sloganConnection":9,"advantageConnection":8,"actionClarity":8,"visualPower":8,"seriesPotential":8,"brandOwnership":8,"categoryRelevance":8,"physicalIndependence":8,"noClientOperationalAction":9}}
Rules:
- selectedCandidateId must be one of the provided ids.
- Do not invent a new generator or slogan.
- Prefer candidates strongly connected to the exact selected slogan and relative advantage.
- Scores are integers 1-10.
""".strip()

STAGE_BRAND_PHYSICAL_SYSTEM = f"""
You are a Builder1 physical-system builder.
Return JSON only. Return exactly this object and no additional top-level keys:
{{"productNameResolved":"...","physicalGenerator":"...","physicalGeneratorNaturalPurpose":"...","physicalGeneratorCampaignRole":"...","physicalGeneratorIsProduct":false,"physicalGeneratorIsPackaging":false,"worksWithoutProductVisible":true,"transferredObject":"...","transferredObjectAction":"...","whyClearerThanShowingProduct":"...","mediumParticipates":false,"mediumRole":"","campaignRationale":"..."}}
Rules:
- Do NOT create, replace, or modify the brand slogan. It is fixed before this stage.
- productNameResolved must exactly match the fixed productNameResolved value provided in the user prompt. Do not rename it.
- Compare at least one serious transferred-object embodiment versus showing the product directly.
- The selected physical generator must NOT be the advertised product, its packaging, or a category package.
- physicalGeneratorIsProduct, physicalGeneratorIsPackaging must be false and worksWithoutProductVisible must be true.
- transferredObject is the external familiar object that performs the fixed slogan action.
- transferredObjectAction is one concrete visual action the transferred object performs.
- whyClearerThanShowingProduct: one sentence maximum.
- The physical generator must derive from: relative advantage → fixed slogan → implied action → selected conceptual generator.
- Do not include graphic generator, series generator, ads, format, adCount, detectedLanguage, strategy fields, or slogan fields.
- mediumParticipates must be JSON boolean true or false, never a string.
- When mediumParticipates is false, mediumRole must be "".
- {BUILDER1_NO_LOGO_PLANNING_RULE}
""".strip()

STAGE_GRAPHIC_SYSTEM_SYSTEM = f"""
You are a Builder1 graphic-system builder.
Return JSON only. Return the graphic generator object directly with no wrapper and no additional top-level keys:
{{"palette":{{"dominant":"#111111","secondary":"#EEEEEE","accent":"#FF5500","background":"#F5F5F5","text":"#222222"}},"layoutTemplate":"visual_right_copy_left","headlinePlacement":"top_left","headlineAlignment":"right","headlineMaxWidthPercent":34,"brandBlockPlacement":"bottom_left","sloganPlacement":"bottom_left","sloganPlacementReason":"","copySafeArea":{{"side":"left","widthPercent":38}},"typographyStyle":"bold_geometric_sans","headlineScale":"large","brandScale":"small","sloganScale":"medium","imageStyle":"editorial_photography","backgroundTreatment":"solid","borderTreatment":"none","recurringGraphicDevice":"...","recurringGraphicDeviceRule":"...","shapeLanguage":"...","framingRule":"...","spacingRule":"..."}}
Rules:
- All five palette colors required as #RRGGBB hex.
- Use only valid layout, placement, typography, image, background, and border enum values.
- recurringGraphicDevice must be visibly repeatable across ads and must remain a campaign composition device, not a product logo or packaging brand mark.
- For Hebrew campaigns default sloganPlacement to bottom_left to preserve RTL reading flow: see visual → understand → read brand interpretation.
- Use another sloganPlacement only when sloganPlacementReason explains the strategic RTL-preserving reason.
- Do not return ads, physical generator, or strategy fields.
- Do not default to billboards, packaging mockups, product-on-pedestal, or split-screen unless the campaign concept requires them.
- {BUILDER1_NO_LOGO_PLANNING_RULE}
""".strip()

STAGE_SERIES_ADS_SYSTEM = f"""
You are a Builder1 series and ads builder.
Return JSON only. Return exactly this object and no additional top-level keys:
{{"seriesGenerator":{{"type":"...","principle":"...","progression":"..."}},"ads":[{{"index":1,"variationLabel":"...","newContribution":"...","conceptualExecution":"...","conceptualActionProof":"...","physicalExecution":"...","visualExecution":"...","sceneDescription":"...","headline":null,"headlineNeededReason":"...","marketingText":"...","familiarExpectation":"...","singleChangedPropertyOrAction":"...","immediateClarityReason":"...","sloganConnection":"...","relativeAdvantageConnection":"...","brandOwnershipReason":"...","categoryRelevanceReason":"...","headlineRequired":false,"headlineReason":"...","sameVisualLawProof":"...","distinctFromOtherAdsReason":"...","noReuseCheck":"..."}}]}}
Rules:
- seriesGenerator must be an object with type, principle, progression.
- ads must contain exactly the requested ad count ({AD_COUNT_MIN}-{AD_COUNT_MAX}).
- The campaign slogan has already been selected upstream and is immutable.
- Do not generate, rewrite, paraphrase, translate, punctuate, or spacing-change the slogan.
- Do not create a different slogan for any advertisement.
- Do not return brandSlogan, slogan, campaignSlogan, physical generator, or graphic generator.
- Use sloganConnection only to explain how each visual execution expresses the fixed slogan.
- The exact displayed slogan will be inserted by the server into every ad.
- Create only distinct ad executions of the fixed conceptual, physical, and graphic laws.
- Default headline to null unless the visual alone cannot communicate the idea.
- Do not use a headline to explain what object changed or what the visual joke means.
- marketingText must be exactly 50 words in the server target language — one paragraph below the image, not inside it.
- marketingText must be written in the target language provided in the user prompt.
- Product identification in rendered ads must use the written product name as plain text only; never request a logo or brand symbol.
- Do not decide whether the product or packaging should appear — the server owns product visibility policy.
- Do not return productVisible, packagingVisible, productVisibilityRequired, or related visibility fields.
- Populate all internal methodology fields on every ad.
- {BUILDER1_NO_LOGO_PLANNING_RULE}
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
        '"briefSupport":"...","advantageSource":"explicit_brief","claimRisk":"low",'
        '"campaignExecutableNow":true,"requiresClientConsultation":false,'
        '"clientActionLevel":"none","implementationCostLevel":"none","simpleStrategicAction":null}]}\n'
        f"Validation errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken output:\n{broken_json}\n"
        "Every candidate must be an object. Exactly 12 ids S01-S12."
    )


def build_strategy_select_user_prompt(candidates: List[Dict[str, Any]], exploration_seed: str) -> str:
    return (
        f"Campaign exploration seed: {exploration_seed}\n"
        "Only immediately executable advertising strategies are provided.\n"
        f"Eligible candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        "Select the strongest candidate by id. Do not rewrite its problem or advantage."
    )


def build_strategy_stage_user_prompt(
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
        "Generate exactly 12 strategy candidates, evaluate each once, and select one eligible candidate.\n"
        "Do not include slogans, conceptual generators, physical generators, graphics, or ads."
    )


def build_slogan_stage_user_prompt(
    *,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
) -> str:
    return (
        f"Product name (fixed): {product_name_resolved}\n"
        f"Brief: {product_description}\n"
        f"Language: {detected_language}\n"
        f"Selected strategic problem: {strategic_problem}\n"
        f"Selected relative advantage: {relative_advantage}\n"
        f"Relative-advantage grounding: {brief_support}\n"
        "Builder1 is a digital advertising agent — the slogan must work from what currently exists.\n"
        "Generate exactly six serious brand-slogan candidates from the fixed relative advantage.\n"
        "Compare them, evaluate each once, and choose the strongest slogan.\n"
        "Prefer a concise slogan, usually one to four words, when natural.\n"
        "A longer slogan is allowed when brevity would damage meaning, clarity, distinctiveness, memorability, or natural language.\n"
        "Do not obey a mechanical word maximum.\n"
        "Select the slogan before conceptual or visual generation.\n"
        "Product Name is fixed separately and must not appear inside brandSlogan.\n"
        "Do not include conceptual generators, physical objects, graphics, or ads."
    )


def build_slogan_scan_user_prompt(
    *,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
) -> str:
    return build_slogan_stage_user_prompt(
        product_name_resolved=product_name_resolved,
        product_description=product_description,
        detected_language=detected_language,
        strategic_problem=strategic_problem,
        relative_advantage=relative_advantage,
        brief_support=brief_support,
    )


def build_slogan_scan_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the candidates array. Return exactly:\n"
        '{"candidates":[{"id":"L01","brandSlogan":"...","derivationFromAdvantage":"...",'
        '"impliedAction":"...","whyOwnable":"...","whyNaturalInLanguage":"...",'
        '"competitorTransferRisk":"low","campaignGenerativePower":"..."}]}\n'
        f"Errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_slogan_select_user_prompt(
    candidates: List[Dict[str, Any]],
    *,
    eligible_candidate_ids: Optional[List[str]] = None,
) -> str:
    eligible = eligible_candidate_ids or [str(c.get("id", "")) for c in candidates]
    return (
        f"Slogan candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        f"eligibleCandidateIds: {json.dumps(eligible, ensure_ascii=False)}\n"
        "Select one candidate by id from eligibleCandidateIds only. Do not rewrite it."
    )


def build_slogan_quality_review_user_prompt(
    *,
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    product_name_resolved: str,
    detected_language: str,
    candidates: List[Dict[str, Any]],
) -> str:
    return (
        f"Product name (fixed): {product_name_resolved}\n"
        f"Language: {detected_language}\n"
        f"Selected strategic problem: {strategic_problem}\n"
        f"Selected relative advantage: {relative_advantage}\n"
        f"Relative-advantage grounding: {brief_support}\n"
        f"Slogan candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        "Review every supplied candidate id exactly once. Do not rewrite slogans or add candidates."
    )


def build_slogan_candidate_repair_user_prompt(
    *,
    invalid_candidate_ids: List[str],
    rejection_codes_by_id: Dict[str, List[str]],
    strategic_problem: str,
    relative_advantage: str,
    brief_support: str,
    product_name_resolved: str,
    detected_language: str,
    candidates: List[Dict[str, Any]],
) -> str:
    code_lines = [
        f"- {cid}: {', '.join(rejection_codes_by_id.get(cid, []))}" for cid in invalid_candidate_ids
    ]
    return (
        f"Repair ONLY these candidate ids: {', '.join(invalid_candidate_ids)}\n"
        f"Product name (fixed): {product_name_resolved}\n"
        f"Language: {detected_language}\n"
        f"Selected strategic problem: {strategic_problem}\n"
        f"Selected relative advantage: {relative_advantage}\n"
        f"Relative-advantage grounding: {brief_support}\n"
        "Rejection codes:\n"
        + "\n".join(code_lines)
        + "\nOriginal invalid candidates:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
        + "\nReturn replacements only for the invalid ids."
    )


def build_conceptual_scan_user_prompt(
    *,
    product_description: str,
    product_name_resolved: str,
    strategic_problem: str,
    relative_advantage: str,
    brand_slogan: str,
    slogan_derivation: str,
    implied_action: str,
    exploration_seed: str,
) -> str:
    return (
        f"Product name (fixed): {product_name_resolved}\n"
        f"Brief: {product_description}\n"
        f"Selected strategic problem: {strategic_problem}\n"
        f"Selected relative advantage: {relative_advantage}\n"
        f"Fixed brand slogan: {brand_slogan}\n"
        f"Slogan derivation: {slogan_derivation}\n"
        f"Implied slogan action: {implied_action}\n"
        f"Exploration seed: {exploration_seed}\n"
        "Every conceptual candidate must derive from the slogan action. "
        "Answer: what action or transformation makes the slogan visible?\n"
        "Return exactly 6 conceptual-generator candidates C01-C06 as objects."
    )


def build_conceptual_stage_user_prompt(
    *,
    product_description: str,
    product_name_resolved: str,
    strategic_problem: str,
    relative_advantage: str,
    brand_slogan: str,
    slogan_derivation: str,
    implied_action: str,
    exploration_seed: str,
) -> str:
    base = build_conceptual_scan_user_prompt(
        product_description=product_description,
        product_name_resolved=product_name_resolved,
        strategic_problem=strategic_problem,
        relative_advantage=relative_advantage,
        brand_slogan=brand_slogan,
        slogan_derivation=slogan_derivation,
        implied_action=implied_action,
        exploration_seed=exploration_seed,
    )
    return (
        f"{base}\n"
        "Generate exactly 6 conceptual candidates, evaluate each once, and select one eligible concept.\n"
        "Do not choose physical generators, graphic systems, or advertisements."
    )


def build_conceptual_scan_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the candidates array. Return exactly:\n"
        '{"candidates":[{"id":"C01","generator":"...","action":"...","input":"...",'
        '"transformation":"...","result":"...","whyItExpressesSlogan":"...",'
        '"whyItExpressesAdvantage":"...","seriesPotential":"...","brandOwnershipPotential":"..."}]}\n'
        f"Errors:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_conceptual_select_user_prompt(candidates: List[Dict[str, Any]]) -> str:
    return (
        f"Conceptual candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n"
        "Select one candidate by id."
    )


def build_product_name_resolution_user_prompt(
    *,
    product_description: str,
    detected_language: str,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> str:
    guidelines = ""
    safe_guidelines = brand_guidelines_for_prompt(brand_guidelines)
    if safe_guidelines:
        guidelines = "\nBrand guidelines:\n" + json.dumps(safe_guidelines, ensure_ascii=False, indent=2)
    language_rule = (
        "The generated name must be English (Latin letters only)."
        if detected_language == "en"
        else "The generated name may be Hebrew or English only."
    )
    return (
        f"Description:\n{product_description.strip()}\n"
        f"Detected language: {detected_language}\n"
        f"{language_rule}\n"
        "Return exactly one productNameResolved value."
        f"{guidelines}"
    )


def build_product_name_resolution_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the product-name JSON object. Return exactly:\n"
        '{"productNameResolved":"..."}\n'
        "Do not copy the description verbatim. Do not return a generic category label.\n"
        f"Missing or invalid fields:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


def build_brand_physical_user_prompt(
    *,
    product_name_resolved: str,
    product_description: str,
    detected_language: str,
    format_value: str,
    strategic_problem: str,
    relative_advantage: str,
    brand_slogan: str,
    slogan_derivation: str,
    implied_action: str,
    conceptual: Dict[str, str],
    brand_guidelines: Optional[Dict[str, Any]] = None,
    visibility_policy: str = "FORBIDDEN",
) -> str:
    guidelines = ""
    safe_guidelines = brand_guidelines_for_prompt(brand_guidelines)
    if safe_guidelines:
        guidelines = "\nBrand guidelines:\n" + json.dumps(safe_guidelines, ensure_ascii=False, indent=2)
    return (
        f"Fixed productNameResolved (echo exactly): {product_name_resolved}\n"
        f"Description: {product_description}\n"
        f"Language context: {detected_language}\n"
        f"Format context: {format_value}\n"
        f"Fixed strategic problem: {strategic_problem}\n"
        f"Fixed relative advantage: {relative_advantage}\n"
        f"Fixed brand slogan (do not change): {brand_slogan}\n"
        f"Fixed slogan derivation: {slogan_derivation}\n"
        f"Fixed implied slogan action: {implied_action}\n"
        f"Fixed conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        f"Server product visibility policy: {visibility_policy}\n"
        "When policy is FORBIDDEN, do not choose the product or its packaging as the physical generator.\n"
        "Compare transferred-object embodiments only. Return physical-generator system only. Do NOT return or modify the brand slogan."
        f"{guidelines}"
    )


def build_brand_physical_repair_prompt(*, broken_json: str, reasons: List[str]) -> str:
    return (
        "Repair ONLY the physical-system JSON object. Return exactly:\n"
        '{"productNameResolved":"...","physicalGenerator":"...","physicalGeneratorNaturalPurpose":"...",'
        '"physicalGeneratorCampaignRole":"...","physicalGeneratorIsProduct":false,'
        '"physicalGeneratorIsPackaging":false,"worksWithoutProductVisible":true,'
        '"transferredObject":"...","transferredObjectAction":"...","whyClearerThanShowingProduct":"...",'
        '"mediumParticipates":false,"mediumRole":"",'
        '"campaignRationale":"..."}\n'
        f"Missing or invalid fields:\n" + "\n".join(f"- {r}" for r in reasons) + "\n"
        f"Broken:\n{broken_json}"
    )


BUILDER1_BRAND_PHYSICAL_IDENTITY_CORRECTION = (
    "The previous selection used the advertised product itself. "
    "Select a different external physical object."
)


def build_brand_physical_identity_retry_prompt(*, base_user_prompt: str) -> str:
    return (
        f"{base_user_prompt}\n\n"
        "=== PHYSICAL GENERATOR CORRECTION (MANDATORY) ===\n"
        f"{BUILDER1_BRAND_PHYSICAL_IDENTITY_CORRECTION}\n"
        "The transferred object must not be the advertised product, its packaging, or the same category unit.\n"
        "=== END PHYSICAL GENERATOR CORRECTION ==="
    )


def build_graphic_system_user_prompt(
    *,
    product_description: str,
    detected_language: str,
    relative_advantage: str,
    brand_slogan: str,
    conceptual: Dict[str, str],
    brand_physical: Dict[str, Any],
    format_value: str,
) -> str:
    hebrew_rule = ""
    if detected_language == "he":
        hebrew_rule = (
            "Hebrew campaign: default main visual right/center, RTL flow, brand slogan at bottom_left "
            "unless sloganPlacementReason provides a strategic RTL-preserving alternative.\n"
        )
    return (
        f"Brief: {product_description}\n"
        f"Language: {detected_language}\n"
        f"Fixed brand slogan: {brand_slogan}\n"
        f"Relative advantage: {relative_advantage}\n"
        f"Conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        f"Physical system:\n{json.dumps(brand_physical, ensure_ascii=False, indent=2)}\n"
        f"Format: {format_value}\n"
        f"{hebrew_rule}"
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
    brand_slogan: str,
    implied_action: str,
    conceptual: Dict[str, str],
    brand_physical: Dict[str, Any],
    graphic_generator: Dict[str, Any],
    visibility_policy: str = "FORBIDDEN",
) -> str:
    indexes = ", ".join(str(i) for i in range(1, ad_count + 1))
    lang_name = "Hebrew" if detected_language == "he" else "English"
    return (
        f"Required ad count: {ad_count}\n"
        f"Required ad indexes: {indexes}\n"
        f"Format: {format_value}\n"
        f"TARGET LANGUAGE FOR ALL MARKETING TEXT: {lang_name} ({detected_language})\n"
        f"Fixed brand slogan across all ads (immutable, server-owned): {brand_slogan}\n"
        f"Fixed implied slogan action: {implied_action}\n"
        "The campaign slogan is already selected. Do not generate, rewrite, paraphrase, translate, "
        "or alter punctuation or spacing for the slogan.\n"
        "Do not return brandSlogan or any per-ad slogan field. Use sloganConnection only to explain "
        "how each ad expresses the fixed slogan. The server inserts the exact slogan into every ad.\n"
        "Every marketingText must:\n"
        "- contain exactly 50 words\n"
        "- be written in the target language\n"
        "- use one coherent paragraph\n"
        "- not switch into another language\n"
        "- not contain headings, labels, bullets, or hashtags\n"
        "- not translate the product or brand name unnecessarily\n"
        "- not invent unsupported claims\n"
        "Headline default: null. Use a headline only when the visual alone cannot communicate the idea.\n"
        "Do not use a headline to explain what changed or what the visual joke means.\n"
        f"Strategic problem: {strategic_problem}\n"
        f"Relative advantage: {relative_advantage}\n"
        f"Conceptual generator:\n{json.dumps(conceptual, ensure_ascii=False, indent=2)}\n"
        f"Physical system:\n{json.dumps(brand_physical, ensure_ascii=False, indent=2)}\n"
        f"Graphic system:\n{json.dumps(graphic_generator, ensure_ascii=False, indent=2)}\n"
        f"Server product visibility policy: {visibility_policy}\n"
        "Do not decide product or packaging visibility in ad output — the server injects visibility fields.\n"
        f"Return seriesGenerator and exactly {ad_count} ads obeying the graphic system and internal methodology fields."
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

