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
            if not isinstance(required, list):
                errors.append(f"{node_path}:missing_required")
            else:
                for field in required:
                    if field not in properties:
                        errors.append(f"{node_path}.required.{field}:not_in_properties")
                for name in properties:
                    if name not in required:
                        errors.append(f"{node_path}.properties.{name}:not_in_required")
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
        if normalized["properties"]:
            prop_keys = list(normalized["properties"].keys())
            required = normalized.get("required")
            if not isinstance(required, list):
                normalized["required"] = prop_keys
            else:
                extra_required = set(required) - set(prop_keys)
                missing_required = set(prop_keys) - set(required)
                if extra_required:
                    pass
                elif missing_required:
                    normalized["required"] = prop_keys

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
    if "json_schema" in message and "schema" in message:
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() in {"invalid_json_schema", "invalid_request_error"}:
        param = str(getattr(exc, "param", "") or "").lower()
        if code.lower() == "invalid_json_schema" or "schema" in message or "response_format" in message:
            return True
        if param.startswith("text") or "format" in param or "schema" in param:
            return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            err_code = str(err.get("code", "")).lower()
            err_param = str(err.get("param", "") or "").lower()
            err_message = str(err.get("message", "")).lower()
            if err_code == "invalid_json_schema":
                return True
            if "invalid schema for response_format" in err_message:
                return True
            if err_code == "invalid_request_error" and (
                "schema" in err_message
                or "response_format" in err_message
                or "json_schema" in err_message
                or err_param.startswith("text")
                or "format" in err_param
            ):
                return True
    return False


def extract_openai_api_error_details(exc: BaseException) -> Dict[str, Any]:
    """Extract safe OpenAI API error fields for logging — never includes secrets."""
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    body = getattr(exc, "body", None)
    error: Dict[str, Any] = {}
    if isinstance(body, dict):
        raw_error = body.get("error")
        if isinstance(raw_error, dict):
            error = raw_error

    response = getattr(exc, "response", None)
    if request_id is None and response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id") or headers.get("X-Request-Id")

    error_type = str(error.get("type") or type(exc).__name__)
    error_code = str(error.get("code") or getattr(exc, "code", "") or "")
    error_param = str(error.get("param") or getattr(exc, "param", "") or "")
    error_message = str(error.get("message") or str(exc) or "")

    return {
        "statusCode": status_code,
        "errorType": error_type,
        "errorCode": error_code,
        "errorParam": error_param,
        "errorMessage": error_message,
        "requestId": request_id or "",
    }


def classify_compliance_api_error(exc: BaseException) -> str:
    """Map an OpenAI API exception to a Builder1 compliance reason code."""
    details = extract_openai_api_error_details(exc)
    status_code = details.get("statusCode")
    name = type(exc).__name__

    if name in {"APITimeoutError", "TimeoutError"} or "timeout" in str(exc).lower():
        return "review_timeout"
    if name == "RateLimitError" or status_code == 429:
        return "review_rate_limited"
    if name == "AuthenticationError" or status_code == 401:
        return "review_auth_error"
    if isinstance(status_code, int) and status_code >= 500:
        return "review_server_error"
    if status_code == 400 or name == "BadRequestError":
        return "request_rejected"
    if name in {"APIConnectionError"}:
        return "review_timeout"
    return "review_server_error"
