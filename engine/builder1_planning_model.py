"""
Builder1 planning model caller helpers — optional strict JSON schema for final substages.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

STRICT_SCHEMA_STAGES = frozenset({"brand_physical", "graphic_system", "series_ads"})

BRAND_PHYSICAL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "productNameResolved",
        "brandSlogan",
        "sloganDerivation",
        "sloganAction",
        "physicalGenerator",
        "physicalGeneratorNaturalPurpose",
        "physicalGeneratorCampaignRole",
        "mediumParticipates",
        "mediumRole",
        "campaignRationale",
    ],
    "properties": {
        "productNameResolved": {"type": "string"},
        "brandSlogan": {"type": "string"},
        "sloganDerivation": {"type": "string"},
        "sloganAction": {"type": "string"},
        "physicalGenerator": {"type": "string"},
        "physicalGeneratorNaturalPurpose": {"type": "string"},
        "physicalGeneratorCampaignRole": {"type": "string"},
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
                "additionalProperties": True,
                "required": [
                    "variationLabel",
                    "newContribution",
                    "conceptualExecution",
                    "conceptualActionProof",
                    "physicalExecution",
                    "visualExecution",
                    "sceneDescription",
                    "headlineNeededReason",
                    "marketingText",
                ],
                "properties": {
                    "index": {"type": ["integer", "null"]},
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
                },
            },
        },
    },
}

STAGE_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "brand_physical": BRAND_PHYSICAL_JSON_SCHEMA,
    "graphic_system": GRAPHIC_SYSTEM_JSON_SCHEMA,
    "series_ads": SERIES_ADS_JSON_SCHEMA,
}

_strict_schema_probe_done = False
_strict_schema_available = False


def _responses_create_supports_text_parameter() -> bool:
    try:
        from openai import OpenAI

        return "text" in inspect.signature(OpenAI.responses.create).parameters
    except Exception as exc:
        logger.info("BUILDER1_STRICT_SCHEMA probe_failed err=%s", exc)
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
    return {
        "format": {
            "type": "json_schema",
            "name": f"builder1_{stage}",
            "schema": schema,
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
    text_format = build_text_format_for_stage(stage)
    if text_format:
        kwargs["text"] = text_format
        logger.info("BUILDER1_STRICT_SCHEMA stage=%s enabled=true", stage)
    elif stage in STRICT_SCHEMA_STAGES:
        logger.info("BUILDER1_STRICT_SCHEMA stage=%s enabled=false", stage)

    response = client.responses.create(**kwargs)
    out_text = getattr(response, "output_text", None) or ""
    if not out_text and hasattr(response, "output"):
        parts: list[str] = []
        for item in response.output or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", "") or "")
        out_text = "".join(parts)
    return parse_json_text(out_text)
