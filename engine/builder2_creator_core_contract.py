"""
Builder2 Creator core contract — single source of truth for prompt, validator, and parity tests.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_tournament_contracts import CANDIDATE_SCHEMA_VERSION

CREATOR_OWNERSHIP_CREATOR_CORE = "CREATOR_CORE"
CREATOR_OWNERSHIP_SERVER_DERIVED = "SERVER_DERIVED"
CREATOR_OWNERSHIP_JUDGE_SEMANTIC = "JUDGE_SEMANTIC"
CREATOR_OWNERSHIP_OPTIONAL_DIAGNOSTIC = "OPTIONAL_DIAGNOSTIC"
CREATOR_OWNERSHIP_REMOVE_DUPLICATE = "REMOVE_DUPLICATE"

PROTOTYPE_APPLICATION_FIELDS: Dict[str, str] = {
    "winning_card": "winningCardApplication",
    "summer_fan": "summerFanApplication",
    "forgot": "forgotApplication",
    "greenpeace_essential_pairing": "essentialPairingApplication",
    "closest": "closestApplication",
    "think_small": "thinkSmallApplication",
}

PROTOTYPE_APPLICATION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "winning_card": ("winningCardApplication", "prototypeApplication"),
    "summer_fan": ("summerFanApplication", "prototypeApplication"),
    "forgot": ("forgotApplication", "prototypeApplication"),
    "greenpeace_essential_pairing": ("essentialPairingApplication", "prototypeApplication"),
    "closest": ("closestApplication", "prototypeApplication"),
    "think_small": ("thinkSmallApplication", "prototypeApplication"),
}

VALID_VERBAL_DECISIONS: FrozenSet[str] = frozenset({"available", "not_needed", "not_found"})
VALID_VISUAL_ANCHOR_TIMING: FrozenSet[str] = frozenset({"opening", "development", "resolution"})

FIELD_OWNERSHIP: Dict[str, str] = {
    "strategyFoundationId": CREATOR_OWNERSHIP_CREATOR_CORE,
    "prototypeId": CREATOR_OWNERSHIP_CREATOR_CORE,
    "structureType": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualParallelType": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualMechanism": CREATOR_OWNERSHIP_CREATOR_CORE,
    "coreCreativeMechanism": CREATOR_OWNERSHIP_CREATOR_CORE,
    "sevenSecondStructure": CREATOR_OWNERSHIP_CREATOR_CORE,
    "sevenSecondStructure.beginning": CREATOR_OWNERSHIP_CREATOR_CORE,
    "sevenSecondStructure.development": CREATOR_OWNERSHIP_CREATOR_CORE,
    "sevenSecondStructure.resolution": CREATOR_OWNERSHIP_CREATOR_CORE,
    "videoExecution": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.mainSubject": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.mainAction": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.location": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.openingFrame": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.whyRunwayShouldUnderstand": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.continuityRisk": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.generationRisks": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.fitsSevenSeconds": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.requiresImpossibleMorphing": CREATOR_OWNERSHIP_CREATOR_CORE,
    "runwayFeasibility.requiresSubtleUnseenInference": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualAnchor": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualAnchor.description": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualAnchor.whyEssential": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualAnchorTiming": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.problemPerception": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.relativeAdvantage": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.mechanismScanSummary": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.goldPrototypeUsed": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.visualParallelType": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.whyParallelExpressesAdvantage": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.whyRunwayShouldUnderstand": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.silentVerification": CREATOR_OWNERSHIP_CREATOR_CORE,
    "creatorReport.puritySelfCheck": CREATOR_OWNERSHIP_OPTIONAL_DIAGNOSTIC,
    "silentVerification": CREATOR_OWNERSHIP_REMOVE_DUPLICATE,
    "silentVerification.explanation": CREATOR_OWNERSHIP_REMOVE_DUPLICATE,
    "silentVerification.understandableWithoutAudio": CREATOR_OWNERSHIP_REMOVE_DUPLICATE,
    "verbalPotential": CREATOR_OWNERSHIP_CREATOR_CORE,
    "verbalPotential.decision": CREATOR_OWNERSHIP_CREATOR_CORE,
    "verbalPotential.keywordOrKeyPhrase": CREATOR_OWNERSHIP_CREATOR_CORE,
    "verbalPotential.visualMeaning": CREATOR_OWNERSHIP_CREATOR_CORE,
    "verbalPotential.strategicMeaning": CREATOR_OWNERSHIP_CREATOR_CORE,
    "verbalPotential.reason": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualFamilyId": CREATOR_OWNERSHIP_CREATOR_CORE,
    "visualFamilyDefinition": CREATOR_OWNERSHIP_CREATOR_CORE,
    "recurringMotif": CREATOR_OWNERSHIP_CREATOR_CORE,
    "sceneVariations": CREATOR_OWNERSHIP_CREATOR_CORE,
    "replacementCheck": CREATOR_OWNERSHIP_CREATOR_CORE,
    "contextCollisionSafeguard": CREATOR_OWNERSHIP_CREATOR_CORE,
    "essenceExtreme": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "essenceExtreme.advantageEssence": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "essenceExtreme.extremePhysicalExpression": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "essenceExtreme.whyChosenObjectsFollowFromTheEssence": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "participationMechanism": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "participationMechanism.whoOrWhatParticipates": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "participationMechanism.visibleAction": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "participationMechanism.visibleCauseAndEffect": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "participationMechanism.notMerelyAReadyMadeResult": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "anchorPunchlineSeparation": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "anchorPunchlineSeparation.anchor": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "anchorPunchlineSeparation.resolutionOrPunchline": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "anchorPunchlineSeparation.whyTheyAreNotTheSameThing": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualAnchor.appearsBeforeOrDuringResolution": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualFamilyConsistency": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualFamilyConsistency.familyDefinition": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualFamilyConsistency.recurringMotif": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualFamilyConsistency.whyAllVariationsBelongTogether": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualFamilyConsistency.sideBySideFrameTest": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "conceptSummary": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "visualFamily": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "prototypeMethodApplied": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "sourceConcept": CREATOR_OWNERSHIP_SERVER_DERIVED,
    "editingPlan": CREATOR_OWNERSHIP_OPTIONAL_DIAGNOSTIC,
    "prototypeMethodApplication": CREATOR_OWNERSHIP_OPTIONAL_DIAGNOSTIC,
    "creativeOrderConfirmation": CREATOR_OWNERSHIP_OPTIONAL_DIAGNOSTIC,
    "creatorReport.creatorPuritySelfCheck": CREATOR_OWNERSHIP_OPTIONAL_DIAGNOSTIC,
}

for _pid, _field in PROTOTYPE_APPLICATION_FIELDS.items():
    FIELD_OWNERSHIP[_field] = CREATOR_OWNERSHIP_CREATOR_CORE

SERVER_DERIVED_FIELD_PATHS: FrozenSet[str] = frozenset(
    path for path, owner in FIELD_OWNERSHIP.items() if owner == CREATOR_OWNERSHIP_SERVER_DERIVED
)

CREATOR_MODEL_REQUIRED_TOP_LEVEL: Tuple[str, ...] = (
    "strategyFoundationId",
    "prototypeId",
    "structureType",
    "visualParallelType",
    "coreCreativeMechanism",
    "visualMechanism",
    "visualAnchor",
    "runwayFeasibility",
    "creatorReport",
    "verbalPotential",
)

CREATOR_MODEL_REQUIRED_NESTED: Tuple[str, ...] = (
    "sevenSecondStructure.beginning",
    "sevenSecondStructure.development",
    "sevenSecondStructure.resolution",
    "runwayFeasibility.mainSubject",
    "runwayFeasibility.mainAction",
    "runwayFeasibility.location",
    "runwayFeasibility.openingFrame",
    "runwayFeasibility.whyRunwayShouldUnderstand",
    "runwayFeasibility.continuityRisk",
    "runwayFeasibility.fitsSevenSeconds",
    "runwayFeasibility.requiresImpossibleMorphing",
    "runwayFeasibility.requiresSubtleUnseenInference",
    "visualAnchor.description",
    "visualAnchor.whyEssential",
    "creatorReport.problemPerception",
    "creatorReport.relativeAdvantage",
    "creatorReport.mechanismScanSummary",
    "creatorReport.goldPrototypeUsed",
    "creatorReport.visualParallelType",
    "creatorReport.whyParallelExpressesAdvantage",
    "creatorReport.whyRunwayShouldUnderstand",
    "verbalPotential.decision",
)

MONTAGE_REQUIRED_TOP_LEVEL: Tuple[str, ...] = (
    "visualFamilyId",
    "visualFamilyDefinition",
    "recurringMotif",
    "sceneVariations",
)


def prototype_application_field(prototype_id: str) -> str:
    return PROTOTYPE_APPLICATION_FIELDS[prototype_id]


def creator_model_required_field_paths(*, prototype_id: str, structure_type: str = "continuous_event") -> List[str]:
    paths = list(CREATOR_MODEL_REQUIRED_TOP_LEVEL)
    paths.extend(CREATOR_MODEL_REQUIRED_NESTED)
    paths.append(prototype_application_field(prototype_id))
    if structure_type == "variation_montage":
        paths.extend(MONTAGE_REQUIRED_TOP_LEVEL)
    return paths


def is_server_derived_field(field_path: str) -> bool:
    return field_path in SERVER_DERIVED_FIELD_PATHS


def is_creator_owned_structural_field(field_path: str) -> bool:
    owner = FIELD_OWNERSHIP.get(field_path)
    if owner == CREATOR_OWNERSHIP_CREATOR_CORE:
        return True
    if field_path.endswith("Application") or field_path in PROTOTYPE_APPLICATION_FIELDS.values():
        return True
    if field_path.startswith("winningCardApplication.") or field_path.startswith("closestApplication."):
        return True
    if field_path.startswith("summerFanApplication.") or field_path.startswith("forgotApplication."):
        return True
    if field_path.startswith("essentialPairingApplication.") or field_path.startswith("thinkSmallApplication."):
        return True
    return False


def build_creator_required_keys_prompt_text(*, prototype_id: str) -> str:
    app_field = prototype_application_field(prototype_id)
    return (
        f"Required keys: schemaVersion={CANDIDATE_SCHEMA_VERSION!r}, methodologyVersion={METHODOLOGY_VERSION!r}, "
        "strategyFoundationId, prototypeId, structureType, visualParallelType, coreCreativeMechanism, visualMechanism, "
        f"{app_field}, "
        "sevenSecondStructure{beginning,development,resolution}, "
        "runwayFeasibility{mainSubject,mainAction,location,openingFrame,continuityRisk,generationRisks,"
        "whyRunwayShouldUnderstand,fitsSevenSeconds,requiresImpossibleMorphing,requiresSubtleUnseenInference}, "
        "visualAnchor{description,whyEssential,visualAnchorTiming}, "
        "verbalPotential{decision,keywordOrKeyPhrase,visualMeaning,strategicMeaning,reason}, "
        "creatorReport{problemPerception,relativeAdvantage,mechanismScanSummary,goldPrototypeUsed,visualParallelType,"
        "whyParallelExpressesAdvantage,whyRunwayShouldUnderstand,silentVerification,puritySelfCheck}.\n"
        "For structureType=variation_montage also require visualFamilyId, visualFamilyDefinition, recurringMotif, "
        "sceneVariations (2–4 items with description and familyId).\n"
        "Include replacementCheck when visualParallelType=replacement.\n"
        "Include contextCollisionSafeguard when visualParallelType=context_collision.\n"
        "Do NOT output essenceExtreme, participationMechanism, anchorPunchlineSeparation, visualFamilyConsistency, "
        "conceptSummary, visualFamily, prototypeMethodApplied, editingPlan, or headline fields — the server derives those."
    )


def filter_creator_owned_structural_errors(errors: List[str]) -> List[str]:
    filtered: List[str] = []
    for item in errors:
        field = item.split(":", 1)[-1] if ":" in item else item
        if is_server_derived_field(field):
            continue
        if field in {"conceptSummary", "visualFamily", "prototypeMethodApplied", "editingPlan", "sourceConcept"}:
            continue
        if field.startswith("visualFamilyConsistency"):
            continue
        if field.startswith("essenceExtreme"):
            continue
        if field.startswith("participationMechanism"):
            continue
        if field.startswith("anchorPunchlineSeparation"):
            continue
        if field == "visualAnchor.appearsBeforeOrDuringResolution":
            continue
        filtered.append(item)
    return list(dict.fromkeys(filtered))
