"""
Builder2 tournament prompt builders — isolated from legacy video_planning prompts.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from engine.builder2_prototypes import Builder2Prototype
from engine.builder2_runway_config import resolve_builder2_video_duration_seconds
from engine.builder2_tournament_contracts import (
    CANDIDATE_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
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
    return (
        "You are the Builder2 Strategy role generating ONE fixed strategic foundation.\n"
        "Do NOT choose a prototype. Do NOT create a visual concept, headline, or Runway prompt.\n"
        "Do NOT invent statistics, studies, or customer research.\n"
        "Ground the problem in real observable practice, physical reality, market behavior, or professional knowledge.\n"
        "Perceptual buyer problems are valid when grounded in observable practice or market behavior.\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n\n"
        f"Return one JSON object only with schemaVersion={STRATEGY_SCHEMA_VERSION!r} and keys:\n"
        "productNameResolved, language, problemPerception{statement,groundingType,groundingEvidence,whyItMatters}, "
        "relativeAdvantage{statement,derivationFromProblem}, "
        "mechanismScan{domainFacts,discoveredMechanism,creativeOpportunity}.\n"
        f'language must be exactly "he" or "en".\n'
        f"groundingType must be exactly one of: {grounding_types}.\n"
        "groundingEvidence must be a non-empty JSON array of concise qualitative evidence strings; "
        "one strong item is sufficient. Do not invent statistics.\n"
        "domainFacts must be a non-empty JSON array of concise qualitative domain facts; "
        "professional knowledge and common market behavior are acceptable without citations.\n"
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
    return (
        "You are the Builder2 Creator role generating ONE isolated candidate idea.\n"
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
        f"Video duration seconds: {duration}\n"
        f"Runway mode constraint: {runway_mode}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation (unchanged for all candidates):\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={CANDIDATE_SCHEMA_VERSION!r}.\n"
        "Required keys: prototypeId, prototypeMethodApplied, coreCreativeMechanism, conceptSummary, "
        "visualParallelType, visualFamily, structureType, sevenSecondStructure{beginning,development,resolution}, "
        "visualAnchor{description,whyEssential}, silentVerification{understandableWithoutAudio,explanation}, "
        "runwayFeasibility{mainSubject,mainAction,location,openingFrame,continuityRisk,generationRisks,whyRunwayShouldUnderstand}, "
        "editingPlan{purpose,reveal,pacing}, creatorReport{problemPerception,relativeAdvantage,mechanismScanSummary,"
        "goldPrototypeUsed,visualParallelType,whyParallelExpressesAdvantage,whyRunwayShouldUnderstand}.\n"
        "structureType must be continuous_event or variation_montage.\n"
        "For think_small identify a real weakness. For essential_pairing avoid appearance-only pairing.\n"
        "For context_collision include a meaningful bridge explanation in creatorReport."
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
    return (
        "You are the Builder2 Judge role evaluating ONE candidate independently.\n"
        "Do NOT redesign the idea, generate a replacement advertisement, or compare to unseen candidates.\n"
        "Do NOT infer missing Creator intent beyond the candidate and Creator Report.\n"
        f"Candidate ID: {candidate_id}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Assigned prototype definition:\n"
        f"{json.dumps({'prototypeId': prototype.prototype_id, 'displayName': prototype.display_name, 'originalProblem': prototype.original_problem, 'reusableMethod': prototype.reusable_method, 'mustNotCopy': prototype.must_not_copy, 'judgeQualityGuidance': prototype.judge_quality_guidance}, ensure_ascii=False)}\n"
        "Candidate to judge:\n"
        f"{json.dumps(candidate, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={JUDGMENT_SCHEMA_VERSION!r}.\n"
        "Required keys: candidateId, eligible, disqualifiers, scores{problemAdvantageIntegrity,mechanismQuality,"
        "prototypeMethodApplication,silentVisualClarity,originalityFreshness,eleganceSimplicity,runwayFeasibility,"
        "editingContribution}, verdict, strengths, weaknesses, prototypeQualityComparison, confidence.\n"
        "Do NOT provide an authoritative total score field."
    )


def build_winner_development_prompt(
    *,
    product_name: str,
    product_description: str,
    language: str,
    strategy_foundation: Dict[str, Any],
    winning_candidate: Dict[str, Any],
    prototype: Builder2Prototype,
    runway_mode: str,
) -> str:
    duration = resolve_builder2_video_duration_seconds()
    return (
        "You are the Builder2 Winner Developer converting ONE winning candidate into a production-ready video plan.\n"
        "Preserve the winning creative mechanism exactly.\n"
        "Do NOT redesign the idea around motion, replace the visual family, replace the visual anchor, "
        "change the strategic problem, or change the relative advantage.\n"
        "Use editing and timing only to strengthen the same mechanism.\n"
        "First ask: How do I preserve the mechanism? Then: How do I express it through seven seconds of video?\n"
        "Generate the headline ONLY now. Headline remainder max seven words excluding product name.\n"
        f"Video duration seconds: {duration}\n"
        f"Runway mode: {runway_mode}\n"
        f"Product name: {product_name or '(empty)'}\n"
        f"Product description: {product_description}\n"
        f"Language: {language}\n"
        "Fixed strategic foundation:\n"
        f"{json.dumps(strategy_foundation, ensure_ascii=False)}\n"
        "Winning candidate:\n"
        f"{json.dumps(winning_candidate, ensure_ascii=False)}\n"
        "Prototype method:\n"
        f"{json.dumps({'prototypeId': prototype.prototype_id, 'reusableMethod': prototype.reusable_method}, ensure_ascii=False)}\n\n"
        f"Return one JSON object only with schemaVersion={WINNER_PLAN_SCHEMA_VERSION!r}.\n"
        "Required keys: productNameResolved, language, problemPerception, relativeAdvantage, prototypeId, "
        "coreCreativeMechanism, visualParallelType, visualFamily, structureType, headline, headlineCoreKeyword, "
        "coreVisualIdea, sequence{beginning,development,resolution}, sceneVariations, visualAnchor, "
        "openingFrameDescription, videoPrompt."
    )
