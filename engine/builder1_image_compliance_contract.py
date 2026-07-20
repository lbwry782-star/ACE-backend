"""
Canonical Builder1 image-compliance response contract — schema, prompt, parser.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from engine.builder1_compliance_adjudication import ComplianceEvidenceItem, parse_compliance_evidence

logger = logging.getLogger(__name__)

COMPLIANCE_SCHEMA_VERSION = "builder1_image_compliance_v2"

IMAGE_COMPLIANCE_VIOLATION_CODES = frozenset(
    {
        "invented_product_logo",
        "supplied_logo_displayed",
        "logo_like_brand_symbol",
        "packaging_contains_brand_mark",
        "campaign_device_used_as_logo",
        "product_name_rendered_as_logo",
        "product_visible_without_explicit_request",
        "packaging_visible_without_explicit_request",
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
    }
)

IMAGE_COMPLIANCE_CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})

COMPLIANCE_RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reviewStatus": {"type": "string", "enum": ["completed"]},
        "hardViolations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "advisories": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidenceType": {"type": "string"},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "relationshipToBrandText": {"type": "string"},
                    "symbolDescription": {"type": "string"},
                    "symbolLocation": {"type": "string"},
                    "relationshipToProductName": {"type": "string"},
                    "relationshipToSlogan": {"type": "string"},
                    "compactAndIsolated": {"type": "boolean"},
                    "enclosedAsBadgeOrSeal": {"type": "boolean"},
                    "repeatedAsBrandSignature": {"type": "boolean"},
                },
                "required": ["code"],
            },
        },
        "overallConfidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["reviewStatus", "hardViolations", "advisories", "evidence", "overallConfidence"],
}


class ImageComplianceResponseError(ValueError):
    """Malformed compliance reviewer JSON — must never be treated as pass."""


@dataclass
class NormalizedCompliancePayload:
    candidate_violations: List[str] = field(default_factory=list)
    hard_violations_raw: List[str] = field(default_factory=list)
    advisories_raw: List[str] = field(default_factory=list)
    evidence_items: List[ComplianceEvidenceItem] = field(default_factory=list)
    overall_confidence: str = "high"
    legacy_normalized: bool = False
    legacy_shape: str = "canonical"
    reviewer_pass: bool = True


def compliance_prompt_json_instructions() -> str:
    return """
Return JSON only using this exact canonical structure:
{
  "reviewStatus": "completed",
  "hardViolations": [],
  "advisories": [],
  "evidence": [
    {
      "code": "possible_logo_like_shape",
      "confidence": "low",
      "evidenceType": "visual_context",
      "description": "short factual description",
      "location": "top_left",
      "relationshipToBrandText": "none"
    }
  ],
  "overallConfidence": "high"
}

Rules:
- reviewStatus must be "completed".
- hardViolations lists only objective visible violations with sufficient confidence.
- advisories lists ambiguous or low-confidence findings only.
- evidence is an array; use [] when no evidence items are needed.
- overallConfidence is high, medium, or low.
- Do not include hidden reasoning or prose outside JSON.
""".strip()


def _string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _allowed_code(code: str) -> bool:
    return (
        code in IMAGE_COMPLIANCE_VIOLATION_CODES
        or code.startswith("possible_")
        or code.startswith("low_confidence_")
    )


def _validate_codes(codes: Sequence[str], *, field_name: str) -> None:
    for code in codes:
        if not _allowed_code(code):
            raise ImageComplianceResponseError(f"invalid_violation_code:{field_name}:{code}")


def _normalize_confidence(value: object, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise ImageComplianceResponseError("confidence_missing")
        return "high"
    normalized = str(value).strip().lower()
    if normalized not in IMAGE_COMPLIANCE_CONFIDENCE_VALUES:
        raise ImageComplianceResponseError("invalid_confidence")
    return normalized


def diagnose_payload(data: object) -> Dict[str, Any]:
    if isinstance(data, dict):
        top_level_keys = sorted(str(key) for key in data.keys())
        field_types = {str(key): type(value).__name__ for key, value in data.items()}
        return {
            "responseType": "dict",
            "topLevelKeys": top_level_keys,
            "fieldTypes": field_types,
        }
    if isinstance(data, str):
        return {
            "responseType": "str",
            "topLevelKeys": [],
            "fieldTypes": {},
            "outputTextPreview": _truncate_preview(data),
        }
    return {
        "responseType": type(data).__name__,
        "topLevelKeys": [],
        "fieldTypes": {},
    }


def _truncate_preview(text: str, limit: int = 240) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def coerce_review_dict(raw_payload: object) -> dict:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ImageComplianceResponseError("compliance_output_not_object")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ImageComplianceResponseError("compliance_output_invalid_json") from exc
        if not isinstance(obj, dict):
            raise ImageComplianceResponseError("compliance_output_not_object")
        return obj
    raise ImageComplianceResponseError("compliance_output_not_object")


def normalize_compliance_payload(data: dict) -> NormalizedCompliancePayload:
    has_pass = "pass" in data and isinstance(data.get("pass"), bool)
    has_canonical = any(
        key in data for key in ("reviewStatus", "hardViolations", "advisories", "overallConfidence")
    )

    hard_raw = _string_list(data.get("hardViolations"))
    advisory_raw = _string_list(data.get("advisories"))
    violations_raw = _string_list(data.get("violations"))
    overall_confidence = _normalize_confidence(data.get("overallConfidence", data.get("confidence")))
    evidence_items = parse_compliance_evidence(data.get("evidence"))

    if has_pass and not has_canonical:
        legacy_pass = bool(data["pass"])
        candidate = list(dict.fromkeys(violations_raw))
        _validate_codes(candidate, field_name="violations")
        if legacy_pass and candidate:
            raise ImageComplianceResponseError("pass_true_with_violations")
        if not legacy_pass and not candidate:
            raise ImageComplianceResponseError("pass_false_without_violations")
        logger.info(
            "BUILDER1_IMAGE_COMPLIANCE_LEGACY_NORMALIZED shape=pass_violations pass=%s violationCount=%s",
            legacy_pass,
            len(candidate),
        )
        return NormalizedCompliancePayload(
            candidate_violations=candidate,
            hard_violations_raw=candidate if not legacy_pass else [],
            advisories_raw=[],
            evidence_items=evidence_items,
            overall_confidence=overall_confidence,
            legacy_normalized=True,
            legacy_shape="pass_violations",
            reviewer_pass=legacy_pass,
        )

    for field in COMPLIANCE_RESPONSE_JSON_SCHEMA["required"]:
        if field not in data:
            raise ImageComplianceResponseError(f"missing_required_field:{field}")

    review_status = str(data.get("reviewStatus") or "").strip().lower()
    if review_status and review_status != "completed":
        raise ImageComplianceResponseError("invalid_review_status")

    _validate_codes(hard_raw, field_name="hardViolations")
    _validate_codes(advisory_raw, field_name="advisories")
    _validate_codes(violations_raw, field_name="violations")

    candidate = list(dict.fromkeys(hard_raw + advisory_raw + violations_raw))
    if has_pass and bool(data["pass"]) and candidate:
        raise ImageComplianceResponseError("pass_true_with_violations")

    legacy_normalized = has_pass or bool(violations_raw and not has_canonical)
    if legacy_normalized:
        logger.info(
            "BUILDER1_IMAGE_COMPLIANCE_LEGACY_NORMALIZED shape=mixed canonicalKeys=%s violationCount=%s",
            sorted(k for k in data.keys() if k in {"pass", "violations", "confidence"}),
            len(candidate),
        )

    reviewer_pass = len(hard_raw) == 0 and len(violations_raw) == 0
    if has_pass and not reviewer_pass and bool(data["pass"]):
        raise ImageComplianceResponseError("pass_true_with_violations")
    if has_pass and reviewer_pass and not bool(data["pass"]) and not candidate:
        raise ImageComplianceResponseError("pass_false_without_violations")
    if has_pass:
        reviewer_pass = bool(data["pass"]) and reviewer_pass

    return NormalizedCompliancePayload(
        candidate_violations=candidate,
        hard_violations_raw=hard_raw or ([c for c in candidate if c not in advisory_raw] if candidate else []),
        advisories_raw=advisory_raw,
        evidence_items=evidence_items,
        overall_confidence=overall_confidence,
        legacy_normalized=legacy_normalized,
        legacy_shape="canonical",
        reviewer_pass=reviewer_pass,
    )


def response_rejection_details(
    exc: Exception,
    *,
    raw_payload: object = None,
    parsed_data: object = None,
) -> Dict[str, Any]:
    reason_code = str(exc).split(":", 1)[0] if isinstance(exc, ImageComplianceResponseError) else type(exc).__name__
    details = diagnose_payload(parsed_data if parsed_data is not None else raw_payload)
    details["reasonCode"] = reason_code
    details["parseError"] = str(exc)
    if isinstance(raw_payload, str):
        details["outputTextPreview"] = _truncate_preview(raw_payload)
    missing_fields: List[str] = []
    unexpected_fields: List[str] = []
    if isinstance(parsed_data, dict):
        expected = set(COMPLIANCE_RESPONSE_JSON_SCHEMA["required"])
        missing_fields = sorted(expected - set(parsed_data.keys()))
        allowed = set(COMPLIANCE_RESPONSE_JSON_SCHEMA["properties"].keys()) | {"pass", "violations", "confidence"}
        unexpected_fields = sorted(set(parsed_data.keys()) - allowed)
    elif reason_code in {"compliance_output_invalid_json", "compliance_output_not_object", "compliance_output_empty"}:
        if isinstance(raw_payload, str):
            missing_fields = ["reviewStatus", "hardViolations", "advisories", "evidence", "overallConfidence"]
    details["missingFields"] = missing_fields
    details["unexpectedFields"] = unexpected_fields
    return details


def build_text_format_for_compliance() -> Optional[Dict[str, Any]]:
    from engine.builder1_planning_model import strict_json_schema_available
    from engine.builder1_strict_schema import prepare_strict_json_schema

    if not strict_json_schema_available():
        return None
    prepared = prepare_strict_json_schema(COMPLIANCE_RESPONSE_JSON_SCHEMA)
    return {
        "format": {
            "type": "json_schema",
            "name": COMPLIANCE_SCHEMA_VERSION,
            "schema": prepared,
            "strict": True,
        }
    }


def compliance_repair_user_prompt(*, parse_error: str, rejected_preview: str) -> str:
    schema_text = json.dumps(COMPLIANCE_RESPONSE_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    return (
        "Your previous compliance JSON failed validation.\n"
        f"Parse failure: {parse_error}\n"
        f"Rejected preview: {rejected_preview}\n"
        "Return ONLY corrected JSON matching this canonical schema exactly:\n"
        f"{schema_text}\n"
        "Do not regenerate commentary. Do not change the image. Return the compliance review only."
    )
