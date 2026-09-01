"""
Builder1 stage parser ↔ API strict-schema contract preflight.

Zero-cost deterministic check before paid planning model calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class StageParserApiContract:
    """Fields the stage parser requires that must appear in the API strict schema."""

    top_level_required: Tuple[str, ...] = ()
    nested_required: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


STAGE_PARSER_API_CONTRACTS: Dict[str, StageParserApiContract] = {
    "brand_physical": StageParserApiContract(
        top_level_required=("directProductRouteAssessment",),
        nested_required=(
            (
                "directProductRouteAssessment",
                (
                    "productOrCategoryImmediatelyReadable",
                    "relativeAdvantageDirectlyExpressibleWithProduct",
                    "productLedAdvertisingMechanismAvailable",
                    "productLedMechanismSummary",
                    "externalAnalogyAddsUniquePersuasiveGain",
                    "externalAnalogyUniqueGain",
                    "additionalTranslationCost",
                    "recommendedRoute",
                    "routeDecisionReason",
                ),
            ),
        ),
    ),
}


def _schema_node(schema: Mapping[str, Any], field: str) -> Dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    node = properties.get(field)
    return dict(node) if isinstance(node, dict) else {}


def verify_stage_parser_api_contract(
    stage: str,
    schema: Mapping[str, Any],
    *,
    contract: StageParserApiContract | None = None,
) -> List[str]:
    """Return errors when API schema omits parser-required fields."""
    spec = contract or STAGE_PARSER_API_CONTRACTS.get(stage)
    if spec is None:
        return []

    errors: List[str] = []
    properties = schema.get("properties")
    required_raw = schema.get("required")
    if not isinstance(properties, dict):
        errors.append(f"{stage}:schema_missing_properties")
        return errors
    required = [str(item) for item in required_raw] if isinstance(required_raw, list) else []

    for field in spec.top_level_required:
        if field not in properties:
            errors.append(f"{stage}:properties.{field}:missing")
        if field not in required:
            errors.append(f"{stage}:required.{field}:missing")

    for parent, nested_fields in spec.nested_required:
        parent_schema = _schema_node(schema, parent)
        if not parent_schema:
            errors.append(f"{stage}:properties.{parent}:missing")
            continue
        parent_props = parent_schema.get("properties")
        parent_required_raw = parent_schema.get("required")
        if not isinstance(parent_props, dict):
            errors.append(f"{stage}:properties.{parent}:missing_properties")
            continue
        parent_required = (
            [str(item) for item in parent_required_raw]
            if isinstance(parent_required_raw, list)
            else []
        )
        for nested in nested_fields:
            if nested not in parent_props:
                errors.append(f"{stage}:properties.{parent}.properties.{nested}:missing")
            if nested not in parent_required:
                errors.append(f"{stage}:properties.{parent}.required.{nested}:missing")

    return list(dict.fromkeys(errors))


def verify_all_registered_stage_contracts(
    stage_schemas: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    for stage, contract in STAGE_PARSER_API_CONTRACTS.items():
        schema = stage_schemas.get(stage)
        if not isinstance(schema, dict):
            report[stage] = [f"{stage}:schema_not_registered"]
            continue
        errors = verify_stage_parser_api_contract(stage, schema, contract=contract)
        if errors:
            report[stage] = errors
    return report
