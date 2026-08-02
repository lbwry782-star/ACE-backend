"""
Builder2 Judge factual-grounding structured-output schema — Responses API contract.
"""
from __future__ import annotations

import inspect
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from engine.builder2_strict_json_schema import StrictSchemaConfigurationError, prepare_strict_json_schema
from engine.builder2_advertising_slogan_quality_contract import (
    JUDGE_SLOGAN_ASSESSMENT_KEY,
    build_default_judge_slogan_assessment,
)
from engine.builder2_judge_core_contract import (
    JUDGE_FACTUAL_GROUNDING_GATE_FIELDS,
    build_judge_example_json,
    build_judge_factual_grounding_prompt_text,
)
from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_evidence_grounding_contract import (
    build_default_judge_factual_grounding_assessment,
    requires_strategy_evidence_grounding,
)
from engine.builder2_tournament_contracts import JUDGMENT_SCHEMA_VERSION, JUDGE_SCORE_RANGES

logger = logging.getLogger(__name__)

BUILDER2_JUDGE_FACTUAL_GROUNDING_OUTPUT_SCHEMA_V1 = "builder2_judge_factual_grounding_output_schema_v1"

REQUIRED_FACTUAL_GROUNDING_FIELD_NAMES: Tuple[str, ...] = tuple(JUDGE_FACTUAL_GROUNDING_GATE_FIELDS) + ("notes",)

_strict_schema_probe_done = False
_strict_schema_available = False


def build_factual_grounding_assessment_json_schema() -> Dict[str, Any]:
    properties: Dict[str, Any] = {gate: {"type": "boolean"} for gate in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS}
    properties["notes"] = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": properties,
        "required": list(REQUIRED_FACTUAL_GROUNDING_FIELD_NAMES),
        "additionalProperties": False,
    }


def _nullable_string_schema() -> Dict[str, Any]:
    return {"type": ["string", "null"]}


def _example_value_to_schema(value: Any) -> Dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return _nullable_string_schema()
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if not value:
            return {"type": "array", "items": {"type": "string"}}
        item_schema = _example_value_to_schema(value[0])
        return {"type": "array", "items": item_schema}
    if isinstance(value, dict):
        properties = {key: _example_value_to_schema(nested) for key, nested in value.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        }
    return {"type": "string"}


def _build_judge_output_schema_template(*, factual_grounding_required: bool) -> Dict[str, Any]:
    template = build_judge_example_json(candidate_id="schema-template")
    template["methodologyVersion"] = METHODOLOGY_VERSION
    template["scores"] = {name: min(5, high // 2) for name, (_low, high) in JUDGE_SCORE_RANGES.items()}
    template.setdefault(JUDGE_SLOGAN_ASSESSMENT_KEY, build_default_judge_slogan_assessment())
    template.setdefault(
        "metaphoricalEmbodimentAssessment",
        {
            "literalExecutionDetected": False,
            "literalPresentationMeaningfullyTransformed": True,
            "creativeEmbodimentModeAccepted": True,
            "physicalEmbodimentMatchesStrategicRelationship": True,
            "viewerDiscoveryPresent": True,
            "sloganOnlyBridgesNotExplains": True,
            "creativeEmbodimentAccepted": True,
            "rejectionReason": None,
        },
    )
    template.setdefault(
        "visualBridgeAssessment",
        {
            "centralVisibleDetail": "Visible mechanism detail",
            "sloganConnectionToVisibleDetail": "Slogan follows the visible detail",
            "sloganConnectionToRelativeAdvantage": "Slogan closes the relative advantage",
            "dependsOnEarlierCopy": False,
            "singleSloganContractSatisfied": True,
        },
    )
    template.setdefault(
        "logoPolicyAssessment",
        {
            "logoDetectedInPlan": False,
            "logoDependentMeaning": False,
            "advertisedLogoRequested": False,
            "thirdPartyBrandingDetected": False,
            "inventedLogoDetected": False,
            "brandedObjectRiskAccepted": False,
            "plainTextIdentificationOnly": True,
            "logoFreeExecutionAccepted": True,
            "logoPolicySatisfied": True,
            "rejectionReason": None,
        },
    )
    if factual_grounding_required:
        template["factualGroundingAssessment"] = build_default_judge_factual_grounding_assessment()
    return template


def build_judge_output_json_schema(*, factual_grounding_required: bool) -> Dict[str, Any]:
    template = _build_judge_output_schema_template(factual_grounding_required=factual_grounding_required)
    schema = _example_value_to_schema(template)
    if factual_grounding_required:
        schema["properties"]["factualGroundingAssessment"] = build_factual_grounding_assessment_json_schema()
        required = list(schema.get("required") or [])
        if "factualGroundingAssessment" not in required:
            required.append("factualGroundingAssessment")
        schema["required"] = required
    return schema


def assert_judge_factual_grounding_output_schema_contract(schema: Dict[str, Any]) -> None:
    errors: List[str] = []
    root_props = schema.get("properties")
    if not isinstance(root_props, dict):
        errors.append("root:missing_properties")
    else:
        assessment = root_props.get("factualGroundingAssessment")
        if not isinstance(assessment, dict):
            errors.append("factualGroundingAssessment:missing")
        else:
            props = assessment.get("properties")
            if not isinstance(props, dict) or not props:
                errors.append("factualGroundingAssessment.properties:missing")
            required = assessment.get("required")
            if not isinstance(required, list) or not required:
                errors.append("factualGroundingAssessment.required:missing")
            else:
                for gate in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
                    if gate not in required:
                        errors.append(f"factualGroundingAssessment.required.{gate}:missing")
                    gate_schema = (props or {}).get(gate)
                    if not isinstance(gate_schema, dict) or gate_schema.get("type") != "boolean":
                        errors.append(f"factualGroundingAssessment.properties.{gate}:not_boolean")
                if "notes" not in required:
                    errors.append("factualGroundingAssessment.required.notes:missing")
            if assessment.get("additionalProperties") is not False:
                errors.append("factualGroundingAssessment.additionalProperties:not_false")
    if errors:
        raise StrictSchemaConfigurationError(errors)


def judge_output_schema_requires_factual_grounding_fields(*, factual_grounding_required: bool) -> bool:
    return bool(factual_grounding_required)


def judge_schema_contract_metadata(*, factual_grounding_required: bool) -> Dict[str, Any]:
    return {
        "judgeOutputSchemaVersion": BUILDER2_JUDGE_FACTUAL_GROUNDING_OUTPUT_SCHEMA_V1 if factual_grounding_required else None,
        "normalJudgeSchemaRequiresFactualGroundingFields": judge_output_schema_requires_factual_grounding_fields(
            factual_grounding_required=factual_grounding_required
        ),
        "repairJudgeSchemaRequiresFactualGroundingFields": judge_output_schema_requires_factual_grounding_fields(
            factual_grounding_required=factual_grounding_required
        ),
        "requiredFactualGroundingFieldNames": list(REQUIRED_FACTUAL_GROUNDING_FIELD_NAMES),
    }


def factual_grounding_object_empty(parsed: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(parsed, dict):
        return True
    assessment = parsed.get("factualGroundingAssessment")
    return isinstance(assessment, dict) and not assessment


def actual_factual_grounding_field_names(parsed: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(parsed, dict):
        return []
    assessment = parsed.get("factualGroundingAssessment")
    if not isinstance(assessment, dict):
        return []
    return sorted(str(key) for key in assessment.keys())


def schema_contract_mismatch_detected(parsed: Optional[Dict[str, Any]], *, factual_grounding_required: bool) -> bool:
    if not factual_grounding_required:
        return False
    if not isinstance(parsed, dict):
        return True
    assessment = parsed.get("factualGroundingAssessment")
    if not isinstance(assessment, dict):
        return True
    if not assessment:
        return True
    for gate in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
        if not isinstance(assessment.get(gate), bool):
            return True
    if not str(assessment.get("notes") or "").strip():
        return True
    return False


def _responses_create_supports_text_parameter() -> bool:
    try:
        from openai.resources.responses import Responses

        return "text" in inspect.signature(Responses.create).parameters
    except Exception:
        return False


def judge_strict_json_schema_available() -> bool:
    global _strict_schema_probe_done, _strict_schema_available
    if _strict_schema_probe_done:
        return _strict_schema_available
    _strict_schema_probe_done = True
    if (os.environ.get("BUILDER2_DISABLE_JUDGE_OUTPUT_SCHEMA") or "").strip().lower() in {"1", "true", "yes"}:
        logger.info("BUILDER2_JUDGE_OUTPUT_SCHEMA disabled_by_env")
        _strict_schema_available = False
        return False
    _strict_schema_available = _responses_create_supports_text_parameter()
    logger.info("BUILDER2_JUDGE_OUTPUT_SCHEMA available=%s", _strict_schema_available)
    return _strict_schema_available


def build_judge_responses_api_text_format(
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
    call_type: str,
) -> Optional[Dict[str, Any]]:
    del call_type  # normal and repair share the same contract
    factual_grounding_required = requires_strategy_evidence_grounding(
        strategy=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    if not factual_grounding_required:
        return None
    if not judge_strict_json_schema_available():
        return None
    schema = build_judge_output_json_schema(factual_grounding_required=True)
    prepared = prepare_strict_json_schema(schema)
    assert_judge_factual_grounding_output_schema_contract(prepared)
    return {
        "format": {
            "type": "json_schema",
            "name": BUILDER2_JUDGE_FACTUAL_GROUNDING_OUTPUT_SCHEMA_V1,
            "schema": prepared,
            "strict": True,
        }
    }


def serialized_judge_output_schema_for_tests(
    *,
    strategy_foundation: Optional[Dict[str, Any]] = None,
    compatibility_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    factual_grounding_required = requires_strategy_evidence_grounding(
        strategy=strategy_foundation,
        compatibility_mode=compatibility_mode,
    )
    if not factual_grounding_required:
        return None
    prepared = prepare_strict_json_schema(build_judge_output_json_schema(factual_grounding_required=True))
    assert_judge_factual_grounding_output_schema_contract(prepared)
    return deepcopy(prepared)


def factual_grounding_assessment_satisfies_schema_contract(assessment: Any) -> bool:
    if not isinstance(assessment, dict) or not assessment:
        return False
    for gate in JUDGE_FACTUAL_GROUNDING_GATE_FIELDS:
        if not isinstance(assessment.get(gate), bool):
            return False
    if not str(assessment.get("notes") or "").strip():
        return False
    return True


__all__ = [
    "BUILDER2_JUDGE_FACTUAL_GROUNDING_OUTPUT_SCHEMA_V1",
    "REQUIRED_FACTUAL_GROUNDING_FIELD_NAMES",
    "actual_factual_grounding_field_names",
    "assert_judge_factual_grounding_output_schema_contract",
    "build_factual_grounding_assessment_json_schema",
    "build_judge_output_json_schema",
    "build_judge_responses_api_text_format",
    "factual_grounding_assessment_satisfies_schema_contract",
    "factual_grounding_object_empty",
    "judge_output_schema_requires_factual_grounding_fields",
    "judge_schema_contract_metadata",
    "judge_strict_json_schema_available",
    "schema_contract_mismatch_detected",
    "serialized_judge_output_schema_for_tests",
]
