"""
Builder2 Judge core contract — single source of truth for prompt, validator, and parity tests.
"""
from __future__ import annotations

import json
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from engine.builder2_creator_core_contract import VALID_VERBAL_DECISIONS
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_tournament_contracts import JUDGE_SCORE_RANGES, JUDGMENT_SCHEMA_VERSION
from engine.builder2_advertising_slogan_quality_contract import (
    JUDGE_SLOGAN_ASSESSMENT_KEY,
    build_default_judge_slogan_assessment,
    build_judge_advertising_slogan_prompt_text,
)

VALID_VERBAL_APPLICABILITY: FrozenSet[str] = frozenset({"available", "not_needed", "not_found"})

VERBAL_ASSESSMENT_BOOLEAN_FIELDS: Tuple[str, ...] = (
    "keywordBornFromVisual",
    "visualMeaningIsClear",
    "strategicMeaningIsClear",
    "twoMeaningsReinforceEachOther",
)

JUDGE_METHODOLOGY_TEXT_FIELDS: Tuple[str, ...] = (
    "problemAdvantageAssessment",
    "mechanismDepthAssessment",
    "prototypeMethodAssessment",
    "visualMechanismAssessment",
    "participationAssessment",
    "visualFamilyAssessment",
    "silentMovieAssessment",
)

JUDGE_FACTUAL_GROUNDING_GATE_FIELDS: Tuple[str, ...] = (
    "productClaimFactuallyGrounded",
    "noUnsupportedFeatureClaim",
    "noCategoryConventionPresentedAsProductFact",
    "viewerWouldNotInferUnsupportedCapability",
    "relativeAdvantageEvidenceAccepted",
)

JUDGE_CONCLUSION_BOOLEAN_PREFIXES: Tuple[str, ...] = (
    "verbalLayerAssessment.keywordBornFromVisual",
    "verbalLayerAssessment.visualMeaningIsClear",
    "verbalLayerAssessment.strategicMeaningIsClear",
    "verbalLayerAssessment.twoMeaningsReinforceEachOther",
    "headlineNecessityAssessment.visualWouldWorkWithoutHeadline",
    "headlineNecessityAssessment.headlineNeeded",
    "headlineNecessityAssessment.headlineRecommended",
    "prototypeMethodAssessment.methodActuallyApplied",
    "eligible",
    *tuple(f"factualGroundingAssessment.{key}" for key in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS),
)


def is_judge_factual_grounding_gate_field(field_path: Optional[str]) -> bool:
    if not field_path:
        return False
    if field_path == "factualGroundingAssessment":
        return False
    prefix = "factualGroundingAssessment."
    if not field_path.startswith(prefix):
        return False
    leaf = field_path.split(".", 1)[1]
    return leaf in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS

JUDGE_SEMANTIC_COHERENCE_FIELDS: FrozenSet[str] = frozenset(
    {
        "verbalLayerAssessment.notes",
        "headlineNecessityAssessment.notes",
    }
)

HEADLINE_NEEDED_ALIASES: Tuple[str, ...] = ("headlineNeeded", "headlineRecommended")


def resolve_creator_verbal_decision(candidate: Optional[Dict[str, Any]]) -> str:
    if not isinstance(candidate, dict):
        return "available"
    verbal = candidate.get("verbalPotential")
    if not isinstance(verbal, dict):
        return "available"
    decision = str(verbal.get("decision") or "").strip().lower()
    if decision in VALID_VERBAL_DECISIONS:
        return decision
    if str(verbal.get("keywordOrKeyPhrase") or "").strip():
        return "available"
    return "not_needed"


def resolve_verbal_applicability(
    verbal: Dict[str, Any],
    *,
    creator_verbal_decision: str,
) -> str:
    raw = str(verbal.get("applicability") or "").strip().lower()
    if raw in VALID_VERBAL_APPLICABILITY:
        return raw
    return creator_verbal_decision if creator_verbal_decision in VALID_VERBAL_APPLICABILITY else "available"


def is_judge_conclusion_boolean_field(field_path: str) -> bool:
    if field_path in JUDGE_CONCLUSION_BOOLEAN_PREFIXES:
        return True
    for prefix in JUDGE_CONCLUSION_BOOLEAN_PREFIXES:
        if field_path.startswith(prefix):
            return True
    return False


def is_judge_structural_repair_field(field_path: Optional[str]) -> bool:
    if not field_path:
        return True
    if field_path in JUDGE_SEMANTIC_COHERENCE_FIELDS:
        return False
    if field_path == "eligible":
        return False
    return True


def filter_judge_structural_errors(errors: List[str]) -> List[str]:
    filtered: List[str] = []
    for item in errors:
        field = item.split(":", 1)[-1] if ":" in item else item
        if not is_judge_structural_repair_field(field):
            continue
        filtered.append(item)
    return list(dict.fromkeys(filtered))


def build_judge_verbal_layer_prompt_text(*, creator_verbal_decision: str) -> str:
    if creator_verbal_decision == "available":
        return (
            "verbalLayerAssessment must include applicability='available' and all four boolean assessment fields "
            f"({', '.join(VERBAL_ASSESSMENT_BOOLEAN_FIELDS)}). "
            "Any combination of true and false is valid Judge criticism. "
            "Include notes explaining negative verbal assessments."
        )
    if creator_verbal_decision == "not_needed":
        return (
            "Creator verbalPotential.decision is 'not_needed'. "
            "verbalLayerAssessment must include applicability='not_needed', notes explaining why verbal analysis "
            "is not applicable, and null for the four boolean assessment fields."
        )
    return (
        "Creator verbalPotential.decision is 'not_found'. "
        "verbalLayerAssessment must include applicability='not_found', notes assessing the absence, "
        "and null for the four boolean assessment fields unless you explicitly assess each dimension."
    )


def build_judge_example_json(*, candidate_id: str = "cand-example") -> Dict[str, Any]:
    scores = {name: min(5, high // 2) for name, (_low, high) in JUDGE_SCORE_RANGES.items()}
    return {
        "schemaVersion": JUDGMENT_SCHEMA_VERSION,
        "methodologyVersion": METHODOLOGY_VERSION,
        "candidateId": candidate_id,
        "eligible": True,
        "disqualifiers": [],
        "scores": scores,
        "verdict": "...",
        "strengths": ["..."],
        "weaknesses": ["..."],
        "prototypeQualityComparison": "...",
        "confidence": 0.75,
        "problemAdvantageAssessment": "...",
        "mechanismDepthAssessment": "...",
        "prototypeMethodAssessment": "...",
        "visualMechanismAssessment": "...",
        "participationAssessment": "...",
        "visualFamilyAssessment": "...",
        "silentMovieAssessment": "...",
        "verbalLayerAssessment": {
            "applicability": "available",
            "keywordBornFromVisual": True,
            "visualMeaningIsClear": True,
            "strategicMeaningIsClear": True,
            "twoMeaningsReinforceEachOther": True,
            "notes": "...",
        },
        "headlineNecessityAssessment": {
            "headlineNeeded": False,
            "visualWouldWorkWithoutHeadline": True,
            "notes": "...",
        },
        "advertisingCompletionAssessment": {
            "advertiserIdentifiable": True,
            "productNamePresent": True,
            "relativeAdvantageClosed": True,
            "sloganSpecificToIdea": True,
            "functionsAsAdvertisement": True,
            "notes": "...",
        },
        JUDGE_SLOGAN_ASSESSMENT_KEY: build_default_judge_slogan_assessment(),
        "semanticAlignmentAssessment": {
            "visualMeaning": "...",
            "sloganMeaning": "...",
            "combinedAdvertisingMeaning": "...",
            "sameStrategicPromise": True,
            "sloganCompletesRatherThanChangesVisual": True,
            "understandableWithoutCreatorReport": True,
            "keyWordMeaningsConnected": True,
            "semanticAlignment": True,
            "failureReason": None,
        },
        "prototypeApplicationAssessment": {
            "assignedPrototypeId": "closest",
            "prototypeMethodVisibleInFilm": True,
            "prototypeMethodReinforcedBySlogan": True,
            "applicationFeelsIntrinsic": True,
            "applicationRequiresRetrospectiveExplanation": False,
            "prototypeFitScore": 12,
        },
        "factualGroundingAssessment": _default_factual_grounding_example(),
    }


def _default_factual_grounding_example() -> Dict[str, Any]:
    from engine.builder2_strategy_evidence_grounding_contract import build_default_judge_factual_grounding_assessment

    return build_default_judge_factual_grounding_assessment()


def build_judge_factual_grounding_prompt_text() -> str:
    gates = ", ".join(JUDGE_FACTUAL_GROUNDING_GATE_FIELDS)
    return (
        "factualGroundingAssessment is mandatory and must never be {} or omitted.\n"
        f"Return all five boolean gates ({gates}) plus notes.\n"
        "Every gate must be JSON boolean true or false — false is valid when the candidate is not factually grounded.\n"
        "notes must be a non-empty string explaining the factual-grounding assessment.\n"
        "Never omit factualGroundingAssessment merely because eligible=false.\n"
        "Compare product claims against the original product description, productSemanticBrief explicitFacts, "
        "and licensedImplications — not identical wording.\n"
        "Reject unsupported capabilities listed in restrictedCapabilities unless explicitly supplied.\n"
        "Internal creatorReport analysis and explicit negations are not public product claims."
    )


def build_judge_required_keys_prompt_text(*, creator_verbal_decision: str, candidate_id: str) -> str:
    score_lines = "\n".join(
        f"- {name}: {low}–{high}" for name, (low, high) in sorted(JUDGE_SCORE_RANGES.items())
    )
    example = build_judge_example_json(candidate_id=candidate_id)
    return (
        "Example JSON shape (illustrative values only):\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"{build_judge_verbal_layer_prompt_text(creator_verbal_decision=creator_verbal_decision)}\n"
        "headlineNecessityAssessment must include headlineNeeded (boolean), "
        "visualWouldWorkWithoutHeadline (boolean), and notes.\n"
        "advertisingCompletionAssessment must include advertiserIdentifiable, productNamePresent, "
        "relativeAdvantageClosed, sloganSpecificToIdea, functionsAsAdvertisement (booleans), and notes.\n"
        f"{build_judge_advertising_slogan_prompt_text()}\n"
        "semanticAlignmentAssessment must include visualMeaning, sloganMeaning, combinedAdvertisingMeaning, "
        "sameStrategicPromise, sloganCompletesRatherThanChangesVisual, understandableWithoutCreatorReport, "
        "keyWordMeaningsConnected, semanticAlignment (boolean), and failureReason (null when aligned).\n"
        "prototypeApplicationAssessment must include assignedPrototypeId, prototypeMethodVisibleInFilm, "
        "prototypeMethodReinforcedBySlogan, applicationFeelsIntrinsic, "
        "applicationRequiresRetrospectiveExplanation (booleans), and prototypeFitScore (0-15).\n"
        "metaphoricalEmbodimentAssessment must include literalExecutionDetected, "
        "literalPresentationMeaningfullyTransformed, creativeEmbodimentModeAccepted, "
        "physicalEmbodimentMatchesStrategicRelationship, viewerDiscoveryPresent, sloganOnlyBridgesNotExplains, "
        "creativeEmbodimentAccepted (booleans), and rejectionReason (null when accepted).\n"
        "Judge whether the strategic perception is physically experienced — not merely stated, diagrammed, or "
        "rescued by the slogan. When the product or business-domain object remains visible, assess whether scale, "
        "context, composition, motion, role, absence, or medium transformation creates the advertising idea. "
        "Do not reject merely because the product is visible or because no external metaphorical world exists.\n"
        "Reject ordinary product demonstrations, untransformed dashboards/graphs/reports, and executions whose "
        "meaning exists only in the report or slogan.\n"
        f"{build_judge_factual_grounding_prompt_text()}\n"
        "logoPolicyAssessment must include logoDetectedInPlan, logoDependentMeaning, advertisedLogoRequested, "
        "thirdPartyBrandingDetected, inventedLogoDetected, brandedObjectRiskAccepted, plainTextIdentificationOnly, "
        "logoFreeExecutionAccepted, logoPolicySatisfied (booleans), and rejectionReason (null when accepted). "
        "Reject any real, third-party, invented, or wordmark-style logo. Do not approve logo-dependent concepts.\n"
        "visualBridgeAssessment must include centralVisibleDetail, sloganConnectionToVisibleDetail, "
        "sloganConnectionToRelativeAdvantage, dependsOnEarlierCopy=false, singleSloganContractSatisfied=true.\n"
        "Reject literal graph/report/dashboard execution unless a meaningful conceptual transformation is proven. "
        "Clarity without creative embodiment is insufficient.\n"
        "Required score fields:\n"
        f"{score_lines}\n"
    )
