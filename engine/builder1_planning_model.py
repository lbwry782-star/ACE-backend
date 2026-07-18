"""
Builder1 planning model caller helpers — optional strict JSON schema for final substages.
"""
from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Dict, Optional

from engine.builder1_strict_schema import (
    StrictSchemaConfigurationError,
    find_strict_schema_errors,
    is_invalid_json_schema_api_error,
    normalize_strict_json_schema,
    prepare_strict_json_schema,
)

logger = logging.getLogger(__name__)

STRICT_SCHEMA_STAGES = frozenset({"strategy_scan", "slogan_scan", "slogan_quality_review", "slogan_candidate_repair", "brand_physical", "graphic_system", "series_ads"})

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

BRAND_PHYSICAL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "productNameResolved",
        "physicalGenerator",
        "physicalGeneratorNaturalPurpose",
        "physicalGeneratorCampaignRole",
        "embodimentChoice",
        "productVisibilityJustification",
        "mediumParticipates",
        "mediumRole",
        "campaignRationale",
    ],
    "properties": {
        "productNameResolved": {"type": "string"},
        "physicalGenerator": {"type": "string"},
        "physicalGeneratorNaturalPurpose": {"type": "string"},
        "physicalGeneratorCampaignRole": {"type": "string"},
        "embodimentChoice": {"type": "string", "enum": ["literal", "transferred"]},
        "productVisibilityJustification": {"type": "string"},
        "mediumParticipates": {"type": "boolean"},
        "mediumRole": {"type": "string"},
        "campaignRationale": {"type": "string"},
    },
}

GRAPHIC_SYSTEM_JSON_SCHEMA: Dict[str, Any] = {
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
        "layoutTemplate": {"type": "string"},
        "headlinePlacement": {"type": "string"},
        "headlineAlignment": {"type": "string"},
        "headlineMaxWidthPercent": {"type": "integer"},
        "brandBlockPlacement": {"type": "string"},
        "sloganPlacement": {"type": "string"},
        "sloganPlacementReason": {"type": "string"},
        "copySafeArea": {
            "type": "object",
            "additionalProperties": False,
            "required": ["side", "widthPercent"],
            "properties": {"side": {"type": "string"}, "widthPercent": {"type": "integer"}},
        },
        "typographyStyle": {"type": "string"},
        "headlineScale": {"type": "string"},
        "brandScale": {"type": "string"},
        "sloganScale": {"type": "string"},
        "imageStyle": {"type": "string"},
        "backgroundTreatment": {"type": "string"},
        "borderTreatment": {"type": "string"},
        "recurringGraphicDevice": {"type": "string"},
        "recurringGraphicDeviceRule": {"type": "string"},
        "shapeLanguage": {"type": "string"},
        "framingRule": {"type": "string"},
        "spacingRule": {"type": "string"},
    },
}

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
                    "productVisibilityRequired": {"type": "boolean"},
                    "productVisibilityReason": {"type": "string"},
                    "sameVisualLawProof": {"type": "string"},
                    "distinctFromOtherAdsReason": {"type": "string"},
                    "noReuseCheck": {"type": "string"},
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
                    "productVisibilityRequired",
                    "productVisibilityReason",
                    "sameVisualLawProof",
                    "distinctFromOtherAdsReason",
                    "noReuseCheck",
                ],
            },
        },
    },
}

STAGE_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "strategy_scan": STRATEGY_SCAN_JSON_SCHEMA,
    "slogan_scan": SLOGAN_SCAN_JSON_SCHEMA,
    "slogan_quality_review": SLOGAN_QUALITY_REVIEW_JSON_SCHEMA,
    "slogan_candidate_repair": SLOGAN_CANDIDATE_REPAIR_JSON_SCHEMA,
    "brand_physical": BRAND_PHYSICAL_JSON_SCHEMA,
    "graphic_system": GRAPHIC_SYSTEM_JSON_SCHEMA,
    "series_ads": SERIES_ADS_JSON_SCHEMA,
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
    parse_json_text: Callable[[str], object],
) -> object:
    combined = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": combined,
        "reasoning": reasoning or {"effort": "low"},
    }
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

    try:
        response = client.responses.create(**kwargs)
    except StrictSchemaConfigurationError:
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
    return parse_json_text(out_text)
