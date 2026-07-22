"""
Builder1 planning model caller helpers — optional strict JSON schema for final substages.
"""
from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Dict, Optional

from engine.builder1_conceptual_evaluations import CONCEPTUAL_REJECTION_CODE_LIST
from engine.builder1_physical_evaluations import PHYSICAL_REJECTION_CODE_LIST
from engine.builder1_strict_schema import (
    StrictSchemaConfigurationError,
    find_strict_schema_errors,
    is_invalid_json_schema_api_error,
    normalize_strict_json_schema,
    prepare_strict_json_schema,
)

logger = logging.getLogger(__name__)

STRICT_SCHEMA_STAGES = frozenset(
    {
        "strategy_stage",
        "strategy_scan",
        "strategy_candidate_repair",
        "strategy_slogan_stage",
        "strategy_slogan_repair",
        "slogan_only_repair",
        "slogan_stage",
        "slogan_scan",
        "slogan_quality_review",
        "slogan_candidate_repair",
        "conceptual_stage",
        "conceptual_evaluation_repair",
        "brand_physical",
        "physical_evaluation_repair",
        "graphic_system",
        "series_ads",
        "series_execution_repair",
    }
)

STRATEGY_SCAN_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "lens",
                    "strategicProblem",
                    "relativeAdvantage",
                    "briefSupport",
                    "advantageSource",
                    "claimRisk",
                    "campaignExecutableNow",
                    "requiresClientConsultation",
                    "clientActionLevel",
                    "implementationCostLevel",
                    "simpleStrategicAction",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "lens": {"type": "string"},
                    "strategicProblem": {"type": "string"},
                    "relativeAdvantage": {"type": "string"},
                    "briefSupport": {"type": "string"},
                    "advantageSource": {"type": "string"},
                    "claimRisk": {"type": "string"},
                    "campaignExecutableNow": {"type": "boolean"},
                    "requiresClientConsultation": {"type": "boolean"},
                    "clientActionLevel": {
                        "type": "string",
                        "enum": ["none", "simple_optional", "complex_required"],
                    },
                    "implementationCostLevel": {
                        "type": "string",
                        "enum": ["none", "negligible", "material"],
                    },
                    "simpleStrategicAction": {"type": ["string", "null"]},
                },
            },
        },
    },
}

SLOGAN_SCAN_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "brandSlogan",
                    "derivationFromAdvantage",
                    "impliedAction",
                    "whyOwnable",
                    "whyNaturalInLanguage",
                    "competitorTransferRisk",
                    "campaignGenerativePower",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "brandSlogan": {"type": "string"},
                    "derivationFromAdvantage": {"type": "string"},
                    "impliedAction": {"type": "string"},
                    "whyOwnable": {"type": "string"},
                    "whyNaturalInLanguage": {"type": "string"},
                    "competitorTransferRisk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "campaignGenerativePower": {"type": "string"},
                },
            },
        },
    },
}

SLOGAN_QUALITY_REVIEW_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reviews"],
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidateId",
                    "derivedFromAdvantage",
                    "naturalInLanguage",
                    "credible",
                    "ownable",
                    "impliedActionValid",
                    "campaignGenerative",
                    "eligible",
                    "rejectionCodes",
                ],
                "properties": {
                    "candidateId": {"type": "string"},
                    "derivedFromAdvantage": {"type": "boolean"},
                    "naturalInLanguage": {"type": "boolean"},
                    "credible": {"type": "boolean"},
                    "ownable": {"type": "boolean"},
                    "impliedActionValid": {"type": "boolean"},
                    "campaignGenerative": {"type": "boolean"},
                    "eligible": {"type": "boolean"},
                    "rejectionCodes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}

SLOGAN_CANDIDATE_REPAIR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["replacements"],
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "brandSlogan",
                    "derivationFromAdvantage",
                    "impliedAction",
                    "whyOwnable",
                    "whyNaturalInLanguage",
                    "competitorTransferRisk",
                    "campaignGenerativePower",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "brandSlogan": {"type": "string"},
                    "derivationFromAdvantage": {"type": "string"},
                    "impliedAction": {"type": "string"},
                    "whyOwnable": {"type": "string"},
                    "whyNaturalInLanguage": {"type": "string"},
                    "competitorTransferRisk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "campaignGenerativePower": {"type": "string"},
                },
            },
        },
    },
}

BRAND_PHYSICAL_CANDIDATE_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "externalObject",
        "physicalWorld",
        "physicalAction",
        "perceptionDemonstrated",
        "sloganActionConnection",
        "clearerThanConventionalProductShot",
        "survivesProductRemoval",
        "seriesPotential",
        "whyClearerThanShowingProduct",
    ],
    "properties": {
        "id": {"type": "string"},
        "externalObject": {"type": "string"},
        "physicalWorld": {"type": "string"},
        "physicalAction": {"type": "string"},
        "perceptionDemonstrated": {"type": "string"},
        "sloganActionConnection": {"type": "string"},
        "clearerThanConventionalProductShot": {"type": "boolean"},
        "survivesProductRemoval": {"type": "boolean"},
        "seriesPotential": {"type": "string"},
        "whyClearerThanShowingProduct": {"type": "string"},
    },
}

BRAND_PHYSICAL_EVALUATION_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidateId",
        "clearerThanConventionalProductShot",
        "survivesProductRemoval",
        "supportsTransferredObject",
        "distinctiveToBrand",
        "eligible",
        "rejectionCodes",
    ],
    "properties": {
        "candidateId": {"type": "string"},
        "clearerThanConventionalProductShot": {"type": "boolean"},
        "survivesProductRemoval": {"type": "boolean"},
        "supportsTransferredObject": {"type": "boolean"},
        "distinctiveToBrand": {"type": "boolean"},
        "eligible": {"type": "boolean"},
        "rejectionCodes": {
            "type": "array",
            "items": {"type": "string", "enum": list(PHYSICAL_REJECTION_CODE_LIST)},
        },
    },
}

PHYSICAL_EVALUATION_REPAIR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["physicalEvaluations"],
    "properties": {
        "physicalEvaluations": {
            "type": "array",
            "items": BRAND_PHYSICAL_EVALUATION_ITEM_SCHEMA,
        },
    },
}

BRAND_PHYSICAL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "physicalCandidates",
        "physicalEvaluations",
        "selectedPhysicalCandidateId",
        "productNameResolved",
        "physicalGenerator",
        "physicalGeneratorNaturalPurpose",
        "physicalGeneratorCampaignRole",
        "physicalGeneratorIsProduct",
        "physicalGeneratorIsPackaging",
        "worksWithoutProductVisible",
        "transferredObject",
        "transferredObjectAction",
        "whyClearerThanShowingProduct",
        "clearerThanConventionalProductShot",
        "survivesProductRemoval",
        "productEvidenceRequired",
        "productEvidenceReason",
        "mediumParticipates",
        "mediumRole",
        "campaignRationale",
    ],
    "properties": {
        "physicalCandidates": {
            "type": "array",
            "items": BRAND_PHYSICAL_CANDIDATE_ITEM_SCHEMA,
        },
        "physicalEvaluations": {
            "type": "array",
            "items": BRAND_PHYSICAL_EVALUATION_ITEM_SCHEMA,
        },
        "selectedPhysicalCandidateId": {"type": "string"},
        "productNameResolved": {"type": "string"},
        "physicalGenerator": {"type": "string"},
        "physicalGeneratorNaturalPurpose": {"type": "string"},
        "physicalGeneratorCampaignRole": {"type": "string"},
        "physicalGeneratorIsProduct": {"type": "boolean"},
        "physicalGeneratorIsPackaging": {"type": "boolean"},
        "worksWithoutProductVisible": {"type": "boolean"},
        "transferredObject": {"type": "string"},
        "transferredObjectAction": {"type": "string"},
        "whyClearerThanShowingProduct": {"type": "string"},
        "clearerThanConventionalProductShot": {"type": "boolean"},
        "survivesProductRemoval": {"type": "boolean"},
        "productEvidenceRequired": {"type": "boolean"},
        "productEvidenceReason": {"type": "string"},
        "mediumParticipates": {"type": "boolean"},
        "mediumRole": {"type": "string"},
        "campaignRationale": {"type": "string"},
    },
}


def _build_graphic_system_json_schema() -> Dict[str, Any]:
    from engine.builder1_graphic_contract import graphic_schema_descriptive_properties, graphic_schema_enum_properties
    from engine.builder1_plan_spec import COPY_SAFE_SIDES

    enum_props = graphic_schema_enum_properties()
    desc_props = graphic_schema_descriptive_properties()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "palette",
            "layoutTemplate",
            "headlinePlacement",
            "headlineAlignment",
            "headlineMaxWidthPercent",
            "brandBlockPlacement",
            "sloganPlacement",
            "sloganPlacementReason",
            "copySafeArea",
            "typographyStyle",
            "headlineScale",
            "brandScale",
            "sloganScale",
            "imageStyle",
            "backgroundTreatment",
            "borderTreatment",
            "recurringGraphicDevice",
            "recurringGraphicDeviceRule",
            "shapeLanguage",
            "framingRule",
            "spacingRule",
        ],
        "properties": {
            "palette": {
                "type": "object",
                "additionalProperties": False,
                "required": ["dominant", "secondary", "accent", "background", "text"],
                "properties": {
                    "dominant": {"type": "string"},
                    "secondary": {"type": "string"},
                    "accent": {"type": "string"},
                    "background": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            "headlineMaxWidthPercent": {"type": "integer"},
            "sloganPlacementReason": {"type": "string"},
            "copySafeArea": {
                "type": "object",
                "additionalProperties": False,
                "required": ["side", "widthPercent"],
                "properties": {
                    "side": {"type": "string", "enum": sorted(COPY_SAFE_SIDES)},
                    "widthPercent": {"type": "integer"},
                },
            },
            **enum_props,
            **desc_props,
        },
    }


GRAPHIC_SYSTEM_JSON_SCHEMA: Dict[str, Any] = _build_graphic_system_json_schema()

SERIES_ADS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["seriesGenerator", "ads"],
    "properties": {
        "seriesGenerator": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "principle", "progression"],
            "properties": {
                "type": {"type": "string"},
                "principle": {"type": "string"},
                "progression": {"type": "string"},
            },
        },
        "ads": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer"},
                    "variationLabel": {"type": "string"},
                    "newContribution": {"type": "string"},
                    "conceptualExecution": {"type": "string"},
                    "conceptualActionProof": {"type": "string"},
                    "physicalExecution": {"type": "string"},
                    "visualExecution": {"type": "string"},
                    "sceneDescription": {"type": "string"},
                    "headline": {"type": ["string", "null"]},
                    "headlineNeededReason": {"type": "string"},
                    "marketingText": {"type": "string"},
                    "familiarExpectation": {"type": "string"},
                    "singleChangedPropertyOrAction": {"type": "string"},
                    "immediateClarityReason": {"type": "string"},
                    "sloganConnection": {"type": "string"},
                    "relativeAdvantageConnection": {"type": "string"},
                    "brandOwnershipReason": {"type": "string"},
                    "categoryRelevanceReason": {"type": "string"},
                    "headlineRequired": {"type": "boolean"},
                    "headlineReason": {"type": "string"},
                    "sameVisualLawProof": {"type": "string"},
                    "distinctFromOtherAdsReason": {"type": "string"},
                    "noReuseCheck": {"type": "string"},
                    "executionSubject": {"type": "string"},
                    "executionAction": {"type": "string"},
                    "executionObjectState": {"type": "string"},
                    "executionScene": {"type": "string"},
                    "executionPunchline": {"type": "string"},
                },
                "required": [
                    "index",
                    "variationLabel",
                    "newContribution",
                    "conceptualExecution",
                    "conceptualActionProof",
                    "physicalExecution",
                    "visualExecution",
                    "sceneDescription",
                    "headline",
                    "headlineNeededReason",
                    "marketingText",
                    "familiarExpectation",
                    "singleChangedPropertyOrAction",
                    "immediateClarityReason",
                    "sloganConnection",
                    "relativeAdvantageConnection",
                    "brandOwnershipReason",
                    "categoryRelevanceReason",
                    "headlineRequired",
                    "headlineReason",
                    "sameVisualLawProof",
                    "distinctFromOtherAdsReason",
                    "noReuseCheck",
                    "executionSubject",
                    "executionAction",
                    "executionObjectState",
                    "executionScene",
                    "executionPunchline",
                ],
            },
        },
    },
}

STRATEGY_CANDIDATE_REPAIR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["replacements"],
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "lens",
                    "strategicProblem",
                    "relativeAdvantage",
                    "briefSupport",
                    "advantageSource",
                    "claimRisk",
                    "campaignExecutableNow",
                    "requiresClientConsultation",
                    "clientActionLevel",
                    "implementationCostLevel",
                    "simpleStrategicAction",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "lens": {"type": "string"},
                    "strategicProblem": {"type": "string"},
                    "relativeAdvantage": {"type": "string"},
                    "briefSupport": {"type": "string"},
                    "advantageSource": {"type": "string"},
                    "claimRisk": {"type": "string"},
                    "campaignExecutableNow": {"type": "boolean"},
                    "requiresClientConsultation": {"type": "boolean"},
                    "clientActionLevel": {
                        "type": "string",
                        "enum": ["none", "simple_optional", "complex_required"],
                    },
                    "implementationCostLevel": {
                        "type": "string",
                        "enum": ["none", "negligible", "material"],
                    },
                    "simpleStrategicAction": {"type": ["string", "null"]},
                },
            },
        },
    },
}

STRATEGY_EVALUATION_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidateId",
        "groundedInBrief",
        "advantageCurrentlyTrue",
        "executableNow",
        "requiresMaterialInvestment",
        "requiresClientConsultation",
        "requiresBusinessTransformation",
        "brandOwnable",
        "categoryRelevant",
        "eligible",
        "rejectionCodes",
    ],
    "properties": {
        "candidateId": {"type": "string"},
        "groundedInBrief": {"type": "boolean"},
        "advantageCurrentlyTrue": {"type": "boolean"},
        "executableNow": {"type": "boolean"},
        "requiresMaterialInvestment": {"type": "boolean"},
        "requiresClientConsultation": {"type": "boolean"},
        "requiresBusinessTransformation": {"type": "boolean"},
        "brandOwnable": {"type": "boolean"},
        "categoryRelevant": {"type": "boolean"},
        "eligible": {"type": "boolean"},
        "rejectionCodes": {"type": "array", "items": {"type": "string"}},
    },
}

STRATEGY_CANDIDATE_STAGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates", "evaluations", "selectedCandidateId", "selectionReason"],
    "properties": {
        "candidates": STRATEGY_SCAN_JSON_SCHEMA["properties"]["candidates"],
        "evaluations": {"type": "array", "items": STRATEGY_EVALUATION_ITEM_SCHEMA},
        "selectedCandidateId": {"type": "string"},
        "selectionReason": {"type": "string"},
    },
}

STRATEGY_FINAL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lens",
        "strategicProblem",
        "relativeAdvantage",
        "briefSupport",
        "advantageSource",
        "claimRisk",
        "campaignExecutableNow",
        "requiresClientConsultation",
        "clientActionLevel",
        "implementationCostLevel",
        "simpleStrategicAction",
        "selectionReason",
    ],
    "properties": {
        "lens": {"type": "string"},
        "strategicProblem": {"type": "string"},
        "relativeAdvantage": {"type": "string"},
        "briefSupport": {"type": "string"},
        "advantageSource": {"type": "string"},
        "claimRisk": {"type": "string"},
        "campaignExecutableNow": {"type": "boolean"},
        "requiresClientConsultation": {"type": "boolean"},
        "clientActionLevel": {
            "type": "string",
            "enum": ["none", "simple_optional", "complex_required"],
        },
        "implementationCostLevel": {
            "type": "string",
            "enum": ["none", "negligible", "material"],
        },
        "simpleStrategicAction": {"type": ["string", "null"]},
        "selectionReason": {"type": "string"},
    },
}

STRATEGY_STAGE_JSON_SCHEMA: Dict[str, Any] = STRATEGY_FINAL_JSON_SCHEMA

SLOGAN_EVALUATION_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidateId",
        "derivedFromAdvantage",
        "naturalInLanguage",
        "credible",
        "ownable",
        "impliedActionValid",
        "campaignGenerative",
        "eligible",
        "rejectionCodes",
    ],
    "properties": {
        "candidateId": {"type": "string"},
        "derivedFromAdvantage": {"type": "boolean"},
        "naturalInLanguage": {"type": "boolean"},
        "credible": {"type": "boolean"},
        "ownable": {"type": "boolean"},
        "impliedActionValid": {"type": "boolean"},
        "campaignGenerative": {"type": "boolean"},
        "eligible": {"type": "boolean"},
        "rejectionCodes": {"type": "array", "items": {"type": "string"}},
    },
}

SLOGAN_CANDIDATE_STAGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates", "evaluations", "selectedCandidateId", "selectionReason"],
    "properties": {
        "candidates": SLOGAN_SCAN_JSON_SCHEMA["properties"]["candidates"],
        "evaluations": {"type": "array", "items": SLOGAN_EVALUATION_ITEM_SCHEMA},
        "selectedCandidateId": {"type": "string"},
        "selectionReason": {"type": "string"},
    },
}

SLOGAN_FINAL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "brandSlogan",
        "derivationFromAdvantage",
        "impliedAction",
        "whyOwnable",
        "whyNaturalInLanguage",
        "competitorTransferRisk",
        "campaignGenerativePower",
        "selectionReason",
    ],
    "properties": {
        "brandSlogan": {"type": "string"},
        "derivationFromAdvantage": {"type": "string"},
        "impliedAction": {"type": "string"},
        "whyOwnable": {"type": "string"},
        "whyNaturalInLanguage": {"type": "string"},
        "competitorTransferRisk": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "campaignGenerativePower": {"type": "string"},
        "selectionReason": {"type": "string"},
    },
}

SLOGAN_STAGE_JSON_SCHEMA: Dict[str, Any] = SLOGAN_FINAL_JSON_SCHEMA

CONCEPTUAL_EVALUATION_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "candidateId",
        "perceptionToCreate",
        "impliedPhysicalLaw",
        "derivedFromSelectedSloganAction",
        "expressesRelativeAdvantage",
        "visuallyClear",
        "seriesGenerative",
        "brandOwnable",
        "categoryRelevant",
        "executableByImageModel",
        "survivesProductRemoval",
        "avoidsProductShotBias",
        "supportsTransferredObject",
        "distinctiveToBrand",
        "productEvidenceRequired",
        "productEvidenceReason",
        "eligible",
        "rejectionCodes",
    ],
    "properties": {
        "candidateId": {"type": "string"},
        "perceptionToCreate": {"type": "string"},
        "impliedPhysicalLaw": {"type": "string"},
        "derivedFromSelectedSloganAction": {"type": "boolean"},
        "expressesRelativeAdvantage": {"type": "boolean"},
        "visuallyClear": {"type": "boolean"},
        "seriesGenerative": {"type": "boolean"},
        "brandOwnable": {"type": "boolean"},
        "categoryRelevant": {"type": "boolean"},
        "executableByImageModel": {"type": "boolean"},
        "survivesProductRemoval": {"type": "boolean"},
        "avoidsProductShotBias": {"type": "boolean"},
        "supportsTransferredObject": {"type": "boolean"},
        "distinctiveToBrand": {"type": "boolean"},
        "productEvidenceRequired": {"type": "boolean"},
        "productEvidenceReason": {"type": "string"},
        "eligible": {"type": "boolean"},
        "rejectionCodes": {
            "type": "array",
            "items": {"type": "string", "enum": list(CONCEPTUAL_REJECTION_CODE_LIST)},
        },
    },
}

CONCEPTUAL_EVALUATION_REPAIR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evaluations"],
    "properties": {
        "evaluations": {"type": "array", "items": CONCEPTUAL_EVALUATION_ITEM_SCHEMA},
    },
}

CONCEPTUAL_STAGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates", "evaluations", "selectedCandidateId", "selectionReason"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "generator",
                    "action",
                    "input",
                    "transformation",
                    "result",
                    "perceptionToCreate",
                    "impliedPhysicalLaw",
                    "whyItExpressesSlogan",
                    "whyItExpressesAdvantage",
                    "seriesPotential",
                    "brandOwnershipPotential",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "generator": {"type": "string"},
                    "action": {"type": "string"},
                    "input": {"type": "string"},
                    "transformation": {"type": "string"},
                    "result": {"type": "string"},
                    "perceptionToCreate": {"type": "string"},
                    "impliedPhysicalLaw": {"type": "string"},
                    "whyItExpressesSlogan": {"type": "string"},
                    "whyItExpressesAdvantage": {"type": "string"},
                    "seriesPotential": {"type": "string"},
                    "brandOwnershipPotential": {"type": "string"},
                },
            },
        },
        "evaluations": {"type": "array", "items": CONCEPTUAL_EVALUATION_ITEM_SCHEMA},
        "selectedCandidateId": {"type": "string"},
        "selectionReason": {"type": "string"},
    },
}

STRATEGY_SLOGAN_STAGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["strategy", "slogan"],
    "properties": {
        "strategy": STRATEGY_FINAL_JSON_SCHEMA,
        "slogan": SLOGAN_FINAL_JSON_SCHEMA,
    },
}

STRATEGY_SLOGAN_REPAIR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["strategy", "slogan"],
    "properties": {
        "strategy": STRATEGY_FINAL_JSON_SCHEMA,
        "slogan": SLOGAN_FINAL_JSON_SCHEMA,
    },
}

SERIES_EXECUTION_REPAIR_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ads"],
    "properties": {
        "ads": SERIES_ADS_JSON_SCHEMA["properties"]["ads"],
    },
}

STAGE_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "strategy_stage": STRATEGY_STAGE_JSON_SCHEMA,
    "strategy_slogan_stage": STRATEGY_SLOGAN_STAGE_JSON_SCHEMA,
    "strategy_slogan_repair": STRATEGY_SLOGAN_REPAIR_JSON_SCHEMA,
    "slogan_only_repair": SLOGAN_STAGE_JSON_SCHEMA,
    "strategy_scan": STRATEGY_SCAN_JSON_SCHEMA,
    "strategy_candidate_repair": STRATEGY_CANDIDATE_REPAIR_JSON_SCHEMA,
    "slogan_stage": SLOGAN_STAGE_JSON_SCHEMA,
    "slogan_scan": SLOGAN_SCAN_JSON_SCHEMA,
    "slogan_quality_review": SLOGAN_QUALITY_REVIEW_JSON_SCHEMA,
    "slogan_candidate_repair": SLOGAN_CANDIDATE_REPAIR_JSON_SCHEMA,
    "conceptual_stage": CONCEPTUAL_STAGE_JSON_SCHEMA,
    "conceptual_evaluation_repair": CONCEPTUAL_EVALUATION_REPAIR_JSON_SCHEMA,
    "brand_physical": BRAND_PHYSICAL_JSON_SCHEMA,
    "physical_evaluation_repair": PHYSICAL_EVALUATION_REPAIR_JSON_SCHEMA,
    "graphic_system": GRAPHIC_SYSTEM_JSON_SCHEMA,
    "series_ads": SERIES_ADS_JSON_SCHEMA,
    "series_execution_repair": SERIES_EXECUTION_REPAIR_JSON_SCHEMA,
}

_strict_schema_probe_done = False
_strict_schema_available = False
_strict_schema_probe_logged = False


def _responses_create_supports_text_parameter() -> bool:
    global _strict_schema_probe_logged
    try:
        from openai.resources.responses import Responses

        return "text" in inspect.signature(Responses.create).parameters
    except Exception as exc:
        if not _strict_schema_probe_logged:
            logger.info("BUILDER1_STRICT_SCHEMA probe_failed err=%s", exc)
            _strict_schema_probe_logged = True
        return False


def strict_json_schema_available() -> bool:
    """Probe once whether responses.create accepts text.format json_schema."""
    global _strict_schema_probe_done, _strict_schema_available
    if _strict_schema_probe_done:
        return _strict_schema_available
    _strict_schema_probe_done = True
    if (os.environ.get("BUILDER1_DISABLE_STRICT_SCHEMA") or "").strip().lower() in {"1", "true", "yes"}:
        logger.info("BUILDER1_STRICT_SCHEMA disabled_by_env")
        return False
    _strict_schema_available = _responses_create_supports_text_parameter()
    logger.info("BUILDER1_STRICT_SCHEMA available=%s", _strict_schema_available)
    return _strict_schema_available


def build_text_format_for_stage(stage: Optional[str]) -> Optional[Dict[str, Any]]:
    if not stage or stage not in STRICT_SCHEMA_STAGES:
        return None
    if not strict_json_schema_available():
        return None
    schema = STAGE_JSON_SCHEMAS.get(stage)
    if not schema:
        return None
    prepared = prepare_strict_json_schema(schema)
    return {
        "format": {
            "type": "json_schema",
            "name": f"builder1_{stage}",
            "schema": prepared,
            "strict": True,
        }
    }


def call_planning_model(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    stage: Optional[str] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    parse_json_text: Callable[[str], object],
) -> object:
    from engine.builder1_planning_profile import resolve_stage_reasoning_effort
    from engine.openai_reasoning import build_reasoning_payload

    if reasoning is not None:
        reasoning_payload = reasoning or None
    elif reasoning_effort is not None:
        reasoning_payload = build_reasoning_payload(effort=reasoning_effort)
    else:
        effort = resolve_stage_reasoning_effort(stage, model)
        reasoning_payload = build_reasoning_payload(effort=effort) if effort else None

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    }
    if reasoning_payload is not None:
        kwargs["reasoning"] = reasoning_payload
    cache_key = (os.environ.get("BUILDER1_PLANNING_PROMPT_CACHE_KEY") or "builder1-planning-v1").strip()
    if cache_key and stage in STRICT_SCHEMA_STAGES:
        kwargs["prompt_cache_key"] = cache_key
    try:
        text_format = build_text_format_for_stage(stage)
    except StrictSchemaConfigurationError as exc:
        logger.error(
            "BUILDER1_STRICT_SCHEMA_INVALID stage=%s paths=%s",
            stage,
            exc.errors[:5],
        )
        raise
    if text_format:
        kwargs["text"] = text_format
        logger.info("BUILDER1_STRICT_SCHEMA stage=%s enabled=true", stage)
    elif stage in STRICT_SCHEMA_STAGES:
        logger.info("BUILDER1_STRICT_SCHEMA stage=%s enabled=false", stage)

    logger.info(
        "BUILDER1_STAGE_MODEL stage=%s model=%s reasoningMode=%s reasoningEffort=%s",
        stage or "",
        model,
        (reasoning_payload or {}).get("mode", "none") if reasoning_payload else "none",
        (reasoning_payload or {}).get("effort", "none") if reasoning_payload else "none",
    )

    try:
        response = client.responses.create(**kwargs)
    except StrictSchemaConfigurationError:
        raise
    except TypeError as exc:
        if "prompt_cache_key" in kwargs:
            kwargs.pop("prompt_cache_key", None)
            logger.info("BUILDER1_PROMPT_CACHE stage=%s supported=false", stage or "")
            response = client.responses.create(**kwargs)
        else:
            raise
    except Exception as exc:
        if is_invalid_json_schema_api_error(exc):
            logger.error("BUILDER1_STRICT_SCHEMA_INVALID stage=%s err=%s", stage, exc)
            raise StrictSchemaConfigurationError([str(exc)]) from exc
        raise
    out_text = getattr(response, "output_text", None) or ""
    if not out_text and hasattr(response, "output"):
        parts: list[str] = []
        for item in response.output or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", "") or "")
        out_text = "".join(parts)

    try:
        from engine.builder1_planning_metrics import get_planning_metrics

        metrics = get_planning_metrics()
        if metrics is not None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "input_tokens", None)
                if prompt_tokens is None:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if output_tokens is None:
                    output_tokens = getattr(usage, "completion_tokens", None)
                metrics.record_token_usage(
                    prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
                    output_tokens=int(output_tokens) if output_tokens is not None else None,
                )
            cache_status = getattr(getattr(response, "usage", None), "prompt_cache_status", None)
            if cache_status is not None:
                logger.info(
                    "BUILDER1_PROMPT_CACHE stage=%s status=%s",
                    stage or "",
                    cache_status,
                )
    except Exception:
        pass

    return parse_json_text(out_text)
