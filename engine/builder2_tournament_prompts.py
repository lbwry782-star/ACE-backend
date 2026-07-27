"""
Builder2 tournament prompt builders — isolated from legacy video_planning prompts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from engine.builder2_creator_core_contract import build_creator_required_keys_prompt_text
from engine.builder2_methodology_contract import (
    CREATIVE_STAGE_ORDER,
    INTEREST_PRIORITY_ORDER,
    METHODOLOGY_VERSION,
    TOURNAMENT_MOTTO,
    VALID_CONTINUITY_RISK,
    VALID_HEADLINE_DECISIONS,
    VALID_HEADLINE_FORMS,
    VALID_STRUCTURE_TYPES,
    VALID_VISUAL_PARALLEL_TYPES,
    prompt_enum_list,
)
from engine.builder2_judge_core_contract import (
    build_judge_required_keys_prompt_text,
    resolve_creator_verbal_decision,
)
from engine.builder2_prototypes import Builder2Prototype
from engine.builder2_runway_config import resolve_builder2_video_duration_seconds
from engine.builder2_strategy_identity import expected_strategy_foundation_id
from engine.builder2_tournament_contracts import (
    CANDIDATE_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    JUDGE_SCORE_RANGES,
    STRATEGY_SCHEMA_VERSION,
    VALID_GROUNDING_TYPES,
    WINNER_PLAN_SCHEMA_VERSION,
)


def build_strategy_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
) -> str:
    grounding_types = ", ".join(sorted(VALID_GROUNDING_TYPES))
    stage_order = " → ".join(CREATIVE_STAGE_ORDER)
    interest_order = " → ".join(INTEREST_PRIORITY_ORDER)
    return (
        "You are the Builder2 Strategy role generating ONE fixed strategic foundation.\n"
        "You are a methodologist, not a copywriter or director.\n"
        "Do NOT choose a prototype. Do NOT create a visual concept, headline, keyword, object pair, or Runway prompt.\n"
        "Do NOT invent statistics, studies, or customer research.\n"
        "Ground the problem in real observable practice, physical reality, market behavior, or professional knowledge.\n"
        "Perceptual buyer problems are valid when grounded in observable practice or market behavior.\n"
        "Identify ONE grounded problem only. Reject generic goals such as wanting more customers or needing awareness.\n"
        f"Creative order for downstream roles: {stage_order}.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n\n"
        f"Return one JSON object only with schemaVersion={STRATEGY_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}, and keys:\n"
        "productNameResolved, language, problemPerception{statement,groundingType,groundingEvidence,whyItMatters}, "
        "relativeAdvantage{statement,derivationFromProblem,truthBoundary,admitsRelevantGap}, "
        "mechanismScan{domainFacts,discoveredMechanism,creativeOpportunity,depthEvidence}.\n"
        f'language must be exactly "he" or "en".\n'
        f"groundingType must be exactly one of: {grounding_types}.\n"
        "groundingEvidence must be a non-empty JSON array of concise qualitative evidence strings; "
        "one strong item is sufficient. Do not invent statistics.\n"
        "domainFacts must be a non-empty JSON array of concise qualitative domain facts; "
        "professional knowledge and common market behavior are acceptable without citations.\n"
        "truthBoundary must explain what the advantage does not falsely claim.\n"
        "admitsRelevantGap must be JSON boolean true or false.\n"
        "depthEvidence must explain why the discovered mechanism is deeper than a first association.\n"
        "Reference-only methods such as market-code inversion may inform mechanism scan guidance but must not become visual concepts here.\n"
        'If you cannot ground a valid problem, return {"planningFailure":"builder2_strategy_not_grounded"} only.'
    )


def build_strategy_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    invalid_output: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    return (
        "You are the Builder2 Strategy repair role.\n"
        "Repair ONLY the listed validation defects in the strategic foundation JSON.\n"
        "Do NOT choose a prototype. Do NOT create a visual concept, headline, or Runway prompt.\n"
        "Do NOT invent statistics, studies, or customer research.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n\n"
        "Original strategy instructions:\n"
        f"{build_strategy_prompt(product_name=product_name, product_description=product_description, language=language)}\n\n"
        "Invalid structured output to repair:\n"
        f"{json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        f"Return one repaired JSON object only with schemaVersion={STRATEGY_SCHEMA_VERSION!r}."
    )


def build_creator_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    attempt_number: int,
    runway_mode: str,
) -> str:
    duration = resolve_builder2_video_duration_seconds()
    structure_types = prompt_enum_list(VALID_STRUCTURE_TYPES)
    continuity_risks = prompt_enum_list(VALID_CONTINUITY_RISK)
    visual_parallel_types = prompt_enum_list(VALID_VISUAL_PARALLEL_TYPES)
    stage_order = " → ".join(CREATIVE_STAGE_ORDER)
    interest_order = " → ".join(INTEREST_PRIORITY_ORDER)
    strategy_id = expected_strategy_foundation_id(strategy_foundation)
    strategy_digest = strategy_foundation.get("strategyFoundationDigest") or ""
    return (
        "You are the Builder2 Creator role generating ONE isolated candidate idea.\n"
        f"Motto: {TOURNAMENT_MOTTO}\n"
        "Interpret the prototype by how it addressed a problem, not by the original problem, object or surface appearance.\n"
        "Do NOT start from a headline, clever word, random object, or visual trick and invent strategy afterward.\n"
        f"Follow this workflow while developing the candidate: {stage_order}.\n"
        "Do not output private reasoning or a narrative of your thought process.\n"
        "Return only the required structured conclusions.\n"
        "The server enforces the workflow contract; do not self-certify internal reasoning order.\n"
        f"Interest priority: {interest_order}.\n"
        "Do not optimize for Runway simplicity before finding an interesting mechanism.\n"
        "Do not use arbitrary weirdness as interest. Freshness must come from a deep mechanism.\n"
        "Realism and silent clarity are mandatory constraints. Everyday is lowest priority, not the starting point.\n"
        "You know ONLY this assigned prototype and this attempt ID.\n"
        "Do NOT reference previous candidates, Judge scores, tournament standings, or other prototypes.\n"
        "Do NOT output a final headline, Runway production prompt, marketing copy, image request, or Judge score.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Attempt number: {attempt_number}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n"
        f"Prototype display name: {prototype.display_name}\n"
        f"Original problem: {prototype.original_problem}\n"
        f"Reusable method: {prototype.reusable_method}\n"
        f"Do not copy: {prototype.must_not_copy}\n"
        f"Creator guidance: {prototype.creator_guidance}\n"
        "The assigned prototype method is supplied by the server. Do not restate or redefine the prototype method.\n"
        "Apply the method to the current Strategy Foundation through the required prototype-specific application object, "
        "visualMechanism, creatorReport.whyParallelExpressesAdvantage, and silent video execution.\n"
        "Return structured application evidence, not a summary of the prototype instructions.\n"
        f"Video duration seconds: {duration}\n"
        f"Runway mode constraint: {runway_mode}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        f"strategyFoundationId (return exactly): {strategy_id!r}\n"
        f"strategyFoundationDigest (reference only, do not recalculate): {strategy_digest!r}\n"
        "Fixed strategic foundation (unchanged for all candidates):\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={CANDIDATE_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}. No Markdown fences. No prose.\n"
        f"{build_creator_required_keys_prompt_text(prototype_id=prototype.prototype_id)}\n"
        f"strategyFoundationId must be exactly {strategy_id!r}.\n"
        f"prototypeId must be exactly {prototype.prototype_id!r}.\n"
        f"structureType must be exactly one of: {structure_types}.\n"
        f"continuityRisk must be exactly one of: {continuity_risks}.\n"
        f"visualParallelType must be exactly one of: {visual_parallel_types}. "
        'If using "other", explain the parallel clearly in creatorReport.whyParallelExpressesAdvantage.\n'
        "verbalPotential.decision must be exactly one of: available, not_needed, not_found.\n"
        "When decision=available provide keywordOrKeyPhrase, visualMeaning, strategicMeaning, and bornFromVisibleMechanism=true.\n"
        "When decision=not_needed or not_found provide a short reason.\n"
        "creatorReport.silentVerification may be a string explanation OR use silentVerification{understandableWithoutAudio,explanation}.\n"
        "runwayFeasibility.generationRisks must be a JSON array (empty array allowed when risk is low).\n"
        f"creatorReport.goldPrototypeUsed must be {prototype.prototype_id!r} or {prototype.display_name!r}.\n"
        "Hebrew free-text fields are allowed. Enum fields must use the exact canonical English tokens above.\n"
        "For think_small identify a real weakness. For essential_pairing avoid appearance-only pairing.\n"
        "For context_collision include a meaningful bridge explanation in creatorReport."
    )


def build_creator_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    attempt_number: int,
    runway_mode: str,
    invalid_output: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    return (
        "You are the Builder2 Creator repair role.\n"
        "Repair ONLY the listed structural/schema defects. Preserve the creative idea.\n"
        "Do NOT reference other candidates, Judge scores, or tournament standings.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n\n"
        "Original Creator instructions:\n"
        f"{build_creator_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate_id=candidate_id, attempt_number=attempt_number, runway_mode=runway_mode)}\n\n"
        "Invalid structured output to repair:\n"
        f"{json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        f"Return one repaired JSON object only with schemaVersion={CANDIDATE_SCHEMA_VERSION!r}."
    )


def build_creator_retry_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate_id: str,
    attempt_number: int,
    runway_mode: str,
    retry_rule: str,
) -> str:
    return (
        "You are the Builder2 Creator role generating ONE fresh isolated candidate idea.\n"
        "This is a clean retry for the same assigned prototype slot.\n"
        "Do NOT reference any previous candidate output, Judge scores, or tournament standings.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Assigned prototype ID: {prototype.prototype_id}\n"
        f"Methodology rule to satisfy: {retry_rule}\n\n"
        f"{build_creator_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate_id=candidate_id, attempt_number=attempt_number, runway_mode=runway_mode)}"
    )


def build_judge_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate: Dict[str, Any],
    candidate_id: str,
) -> str:
    interest_order = " → ".join(INTEREST_PRIORITY_ORDER)
    contract = candidate.get("prototypeMethodContract") or {}
    creator_verbal_decision = resolve_creator_verbal_decision(candidate)
    judge_contract = build_judge_required_keys_prompt_text(
        creator_verbal_decision=creator_verbal_decision,
        candidate_id=candidate_id,
    )
    return (
        "You are the Builder2 Judge role evaluating ONE candidate independently.\n"
        f"Motto: {TOURNAMENT_MOTTO}\n"
        "Do NOT reward prototype prestige, literal similarity, historical fame, or prototype ID.\n"
        "Do NOT reward repetition of the prototype description or any Creator-written methodSummary.\n"
        "Judge whether the actual visual mechanism applied the assigned prototype's problem-solving method "
        "to the current Problem Perception and Relative Advantage. Judge the application itself.\n"
        "Do NOT redesign the idea, generate a replacement advertisement, or compare to unseen candidates.\n"
        "Do NOT infer missing Creator intent beyond the candidate and Creator Report.\n"
        "Assume the idea is wrong until it proves itself. A valid eligible=false judgment is valid.\n"
        f"Interest priority for qualitative assessment: {interest_order}.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Canonical prototype method contract (authoritative):\n"
        f"{json.dumps(contract, ensure_ascii=False)}\n"
        "Assigned prototype definition (instruction context only — not proof of application):\n"
        f"{json.dumps({'prototypeId': prototype.prototype_id, 'displayName': prototype.display_name, 'originalProblem': prototype.original_problem, 'reusableMethod': prototype.reusable_method, 'mustNotCopy': prototype.must_not_copy, 'judgeQualityGuidance': prototype.judge_quality_guidance}, ensure_ascii=False)}\n"
        "Candidate to judge:\n"
        f"{json.dumps(candidate, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={JUDGMENT_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}. No Markdown fences. No prose.\n"
        f"candidateId must be exactly {candidate_id!r}.\n"
        "eligible must be JSON boolean true or false, not a string.\n"
        "disqualifiers must be a JSON array.\n"
        "strengths must be a JSON array.\n"
        "weaknesses must be a JSON array.\n"
        "confidence must be a JSON number from 0.0 to 1.0.\n"
        "Every score must be an integer within its category maximum.\n"
        "Do NOT output totalScore, total, or any authoritative total score field.\n"
        "Hebrew free-text fields are allowed in verdict, strengths, weaknesses and prototypeQualityComparison.\n"
        f"{judge_contract}\n"
        "Required keys: candidateId, eligible, disqualifiers, scores, verdict, strengths, weaknesses, "
        "prototypeQualityComparison, confidence, problemAdvantageAssessment, mechanismDepthAssessment, "
        "prototypeMethodAssessment, visualMechanismAssessment, participationAssessment, visualFamilyAssessment, "
        "silentMovieAssessment, verbalLayerAssessment, headlineNecessityAssessment, advertisingCompletionAssessment.\n"
        "If eligible=false, include at least one disqualifier explaining why."
    )


def build_judge_repair_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate: Dict[str, Any],
    candidate_id: str,
    invalid_output: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    return (
        "You are the Builder2 Judge repair role.\n"
        "Repair ONLY the listed structural defects. Preserve the substantive judgment.\n"
        "Do NOT redesign the candidate or change eligibility merely to satisfy schema.\n"
        f"Candidate ID: {candidate_id}\n\n"
        "Original Judge instructions:\n"
        f"{build_judge_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate=candidate, candidate_id=candidate_id)}\n\n"
        "Invalid structured output to repair:\n"
        f"{json.dumps(invalid_output, ensure_ascii=False)}\n\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        f"Return one repaired JSON object only with schemaVersion={JUDGMENT_SCHEMA_VERSION!r}."
    )


def build_judge_retry_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    prototype: Builder2Prototype,
    candidate: Dict[str, Any],
    candidate_id: str,
    retry_rule: str,
) -> str:
    return (
        "You are the Builder2 Judge role performing ONE clean retry for the same candidate.\n"
        "Do NOT reference any previous Judge response, score, ranking, or unseen candidate.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Violated Judge rule to respect: {retry_rule}\n\n"
        f"{build_judge_prompt(product_name=product_name, product_description=product_description, language=language, strategy_foundation=strategy_foundation, prototype=prototype, candidate=candidate, candidate_id=candidate_id)}"
    )


def build_winner_development_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    prototype: Builder2Prototype,
    runway_mode: str,
    preservation_snapshot: Dict[str, Any],
) -> str:
    duration = resolve_builder2_video_duration_seconds()
    headline_forms = prompt_enum_list(VALID_HEADLINE_FORMS)
    headline_decisions = prompt_enum_list(VALID_HEADLINE_DECISIONS)
    return (
        "You are the Builder2 Winner Developer converting ONE winning candidate into a production-ready video plan.\n"
        "Refine the winning execution only.\n"
        "Do not redesign the advertisement.\n"
        "Do not replace the visual mechanism.\n"
        "Do not solve weaknesses by creating a new concept.\n"
        "Preserve the winning creative mechanism exactly.\n"
        "Do NOT redesign the idea around motion, replace the visual family, replace the visual anchor, "
        "change the strategic problem, or change the relative advantage.\n"
        "Use editing and timing only to strengthen the same mechanism.\n"
        "First ask: How do I preserve the mechanism? Then: How do I express it through seven seconds of video?\n"
        "Generate the headline ONLY now when headlineDecision.decision=use (alias include). Headline remainder max seven words excluding product name.\n"
        "When headlineDecision.decision=omit, leave headline, headlineText, and headlineCoreKeyword empty and set headlineForm=none. "
        "Do not invent a separate in-scene headline; the authoritative end-card slogan is server-owned from the winning Creator candidate.\n"
        "Required advertisingClosure object: required=true, productNameText, sloganText, language, "
        "presentationMode=end_card, durationSeconds=2, headlineSource, noLogo=true.\n"
        f"Video duration seconds: {duration}\n"
        f"Runway mode: {runway_mode}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Winning candidate:\n"
        f"{json.dumps(winning_candidate, ensure_ascii=False)}\n"
        "Valid Judge judgment for this winning candidate only:\n"
        f"{json.dumps(winning_judgment, ensure_ascii=False)}\n"
        "Preservation snapshot (must match preservationReference identity fields exactly):\n"
        f"{json.dumps(preservation_snapshot, ensure_ascii=False)}\n"
        "Prototype method:\n"
        f"{json.dumps({'prototypeId': prototype.prototype_id, 'reusableMethod': prototype.reusable_method}, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={WINNER_PLAN_SCHEMA_VERSION!r}, "
        f"methodologyVersion={METHODOLOGY_VERSION!r}.\n"
        "Required keys: productNameResolved, language, prototypeId, "
        "coreCreativeMechanism, visualParallelType, visualFamily, structureType, headlineDecision, headlineForm, "
        "advertisingClosure, coreVisualIdea, sequence{beginning,development,resolution}, sceneVariations, visualAnchor, "
        "openingFrameDescription, videoPrompt.\n"
        "Optional diagnostic keys: preservationReference, winnerPreservationCheck, headlineDecision.reason.\n"
        "headlineDecision.decision is authoritative (use|omit; include accepted as alias for use). "
        "headlineDecision.reason is optional diagnostic metadata and is not required for validity.\n"
        "Strategic fields (problemPerception, relativeAdvantage, coreCreativeMechanism, prototype identity, "
        "visual anchor foundation) are server-owned and will be restored from the selected winning candidate.\n"
        f'headlineDecision.decision must be one of: {headline_decisions}.\n'
        f'headlineForm must be one of: {headline_forms}. headlineForm="none" requires decision="omit".\n'
        "If preservationReference is returned, it must match the source candidate identity fields.\n"
        'When decision="omit", headline may be empty.\n'
    )


def build_winner_headline_repair_prompt(
    *,
    product_name: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    winning_judgment: Dict[str, Any],
    parsed_winner_plan: Dict[str, Any],
    validation_failures: List[str],
) -> str:
    headline_form = str(parsed_winner_plan.get("headlineForm") or "").strip() or "(unspecified)"
    headline_decision = parsed_winner_plan.get("headlineDecision")
    scene_context = {
        "coreVisualIdea": parsed_winner_plan.get("coreVisualIdea"),
        "coreCreativeMechanism": parsed_winner_plan.get("coreCreativeMechanism"),
        "sequence": parsed_winner_plan.get("sequence"),
        "visualAnchor": parsed_winner_plan.get("visualAnchor"),
        "videoPrompt": parsed_winner_plan.get("videoPrompt"),
        "headlineForm": headline_form,
        "headlineDecision": headline_decision,
    }
    return (
        "You are the Builder2 Winner headline repair role.\n"
        "Repair ONLY the missing in-scene headline fields for an already accepted Winner plan.\n"
        "Preserve the existing advertisement exactly.\n"
        "Do NOT redesign, reinterpret, or replace the creative idea.\n"
        "Do NOT change the visual sequence, videoPrompt, strategy, mechanism, prototype, or advertising closure.\n"
        "Do NOT return a complete Winner plan.\n"
        "Do NOT copy the end-card slogan merely because it exists; the in-scene headline is a separate element.\n"
        "Do NOT depend on a fixed idiom unless the existing contract already permits it.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Winning Creator candidate:\n"
        f"{json.dumps(winning_candidate, ensure_ascii=False)}\n"
        "Winning Judge judgment (headline necessity is authoritative):\n"
        f"{json.dumps(winning_judgment, ensure_ascii=False)}\n"
        "Existing parsed Winner plan scene context (preserve unchanged):\n"
        f"{json.dumps(scene_context, ensure_ascii=False)}\n"
        "Exact validation failures to fix:\n"
        + "\n".join(f"- {item}" for item in validation_failures)
        + "\n\n"
        "Return one JSON object only with exactly these keys:\n"
        '- "headline": non-empty remainder text excluding the product name; maximum seven words in the remainder.\n'
        '- "headlineCoreKeyword": exactly one meaningful word that appears in the headline remainder.\n'
        "The headline must be understandable in the requested language.\n"
        "No explanations outside JSON.\n"
    )
