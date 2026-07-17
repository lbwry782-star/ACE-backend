"""
Builder1 strict JSON-schema normalization and local validation for Responses API.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Union


class StrictSchemaConfigurationError(ValueError):
    """Raised when a Builder1 strict schema is invalid before or after API submission."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("schema_configuration_error")


SchemaNode = Dict[str, Any]


def _schema_types(node: SchemaNode) -> List[str]:
    raw = node.get("type")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(t) for t in raw if isinstance(t, str)]
    return []


def _is_object_node(node: SchemaNode) -> bool:
    return "object" in _schema_types(node)


def _unsupported_json_values(node: Any, path: str, errors: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            _unsupported_json_values(value, child_path, errors)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            _unsupported_json_values(value, f"{path}[{idx}]", errors)
    elif isinstance(node, (set, tuple, bytes)):
        errors.append(f"{path}:unsupported_value_type")


def find_strict_schema_errors(schema: Any, *, path: str = "") -> List[str]:
    """Return schema paths that violate strict Responses API requirements."""
    errors: List[str] = []
    _unsupported_json_values(schema, path or "$", errors)
    if not isinstance(schema, dict):
        errors.append(f"{path or '$'}:not_object")
        return errors

    node_path = path or "$"
    if _is_object_node(schema):
        if schema.get("additionalProperties") is not False:
            errors.append(f"{node_path}:missing_additionalProperties_false")
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            errors.append(f"{node_path}:missing_properties")
        else:
            required = schema.get("required")
            if isinstance(required, list):
                for field in required:
                    if field not in properties:
                        errors.append(f"{node_path}.required.{field}:not_in_properties")
            for name, subschema in properties.items():
                sub_path = f"{node_path}.properties.{name}"
                errors.extend(find_strict_schema_errors(subschema, path=sub_path))

    if schema.get("type") == "array":
        if "items" not in schema:
            errors.append(f"{node_path}:missing_items")
        else:
            errors.extend(find_strict_schema_errors(schema["items"], path=f"{node_path}.items"))

    for combinator in ("anyOf", "oneOf", "allOf"):
        options = schema.get(combinator)
        if isinstance(options, list):
            for idx, option in enumerate(options):
                errors.extend(find_strict_schema_errors(option, path=f"{node_path}.{combinator}[{idx}]"))

    return list(dict.fromkeys(errors))


def normalize_strict_json_schema(schema: Any) -> Any:
    """Recursively enforce strict object rules while preserving enums and nullable types."""
    if isinstance(schema, list):
        return [normalize_strict_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized: SchemaNode = copy.deepcopy(schema)

    if _is_object_node(normalized):
        normalized["additionalProperties"] = False
        properties = normalized.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            normalized["properties"] = properties
        normalized["properties"] = {
            key: normalize_strict_json_schema(value) for key, value in properties.items()
        }

    if normalized.get("type") == "array" and "items" in normalized:
        normalized["items"] = normalize_strict_json_schema(normalized["items"])

    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in normalized and isinstance(normalized[combinator], list):
            normalized[combinator] = [
                normalize_strict_json_schema(option) for option in normalized[combinator]
            ]

    return normalized


def prepare_strict_json_schema(schema: SchemaNode) -> SchemaNode:
    """Normalize and validate a schema before sending it to the Responses API."""
    normalized = normalize_strict_json_schema(schema)
    if not isinstance(normalized, dict):
        raise StrictSchemaConfigurationError(["schema_not_object"])
    errors = find_strict_schema_errors(normalized)
    if errors:
        raise StrictSchemaConfigurationError(errors)
    return normalized


def is_invalid_json_schema_api_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "invalid_json_schema" in message:
        return True
    if "invalid schema for response_format" in message:
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() == "invalid_json_schema":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict) and str(err.get("code", "")).lower() == "invalid_json_schema":
            return True
    return False
