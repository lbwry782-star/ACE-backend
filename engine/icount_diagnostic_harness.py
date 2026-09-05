"""
Temporary iCount Payment Page IPN / return URL protocol diagnostic harness.

DIAGNOSTIC ONLY — no payment state, entitlements, or checkout mutation.
Remove this module after protocol discovery; do not use as production IPN.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SYNTHETIC_MARKER_PREFIX = "ACE_ICOUNT_TEST_"

NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# Field names whose values may be logged in full when clearly non-PII.
_FULL_VALUE_NAME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^amount$", re.I),
    re.compile(r"^sum$", re.I),
    re.compile(r"^total$", re.I),
    re.compile(r"currency", re.I),
    re.compile(r"status", re.I),
    re.compile(r"payment_status", re.I),
    re.compile(r"^cp$", re.I),
    re.compile(r"docnum", re.I),
    re.compile(r"doctype", re.I),
    re.compile(r"transaction", re.I),
    re.compile(r"payment_page", re.I),
    re.compile(r"page_id", re.I),
    re.compile(r"pageid", re.I),
    re.compile(r"confirmation", re.I),
    re.compile(r"invoice", re.I),
    re.compile(r"^ref$", re.I),
    re.compile(r"reference", re.I),
    re.compile(r"ace_checkout_ref", re.I),
    re.compile(r"custom", re.I),
)

# Field names that must never have values logged (names only if present).
_SENSITIVE_NAME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"card", re.I),
    re.compile(r"cvv|cvc|ccv", re.I),
    re.compile(r"pan", re.I),
    re.compile(r"credit", re.I),
    re.compile(r"email", re.I),
    re.compile(r"phone", re.I),
    re.compile(r"mobile", re.I),
    re.compile(r"address", re.I),
    re.compile(r"\bname\b", re.I),
    re.compile(r"customer", re.I),
    re.compile(r"client_name", re.I),
    re.compile(r"password", re.I),
    re.compile(r"secret", re.I),
    re.compile(r"cookie", re.I),
    re.compile(r"teudat", re.I),
    re.compile(r"id_number", re.I),
    re.compile(r"personal", re.I),
    re.compile(r"holder", re.I),
)

# Header names: always log name; value metadata only for auth/signature discovery.
_HEADER_VALUE_METADATA_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"signature", re.I),
    re.compile(r"hash", re.I),
    re.compile(r"authorization", re.I),
    re.compile(r"hmac", re.I),
    re.compile(r"digest", re.I),
)

# Correlation hint buckets — field NAME presence only (no undocumented semantics).
_CORRELATION_BUCKETS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "transactionIdFieldNames": (
        re.compile(r"transaction", re.I),
        re.compile(r"^tx_?id$", re.I),
        re.compile(r"trans_id", re.I),
        re.compile(r"payment_id", re.I),
    ),
    "documentIdFieldNames": (
        re.compile(r"docnum", re.I),
        re.compile(r"document", re.I),
        re.compile(r"invoice", re.I),
        re.compile(r"receipt", re.I),
    ),
    "paymentPageIdFieldNames": (
        re.compile(r"payment_page", re.I),
        re.compile(r"page_id", re.I),
        re.compile(r"^cp$", re.I),
        re.compile(r"checkout_page", re.I),
    ),
    "amountFieldNames": (
        re.compile(r"^amount$", re.I),
        re.compile(r"^sum$", re.I),
        re.compile(r"^total$", re.I),
        re.compile(r"payment_amount", re.I),
    ),
    "currencyFieldNames": (re.compile(r"currency", re.I),),
    "statusFieldNames": (
        re.compile(r"status", re.I),
        re.compile(r"payment_status", re.I),
        re.compile(r"paid", re.I),
        re.compile(r"success", re.I),
    ),
    "customReferenceFieldNames": (
        re.compile(r"ace_checkout_ref", re.I),
        re.compile(r"custom", re.I),
        re.compile(r"comment", re.I),
        re.compile(r"reference", re.I),
        re.compile(r"^ref$", re.I),
        re.compile(r"order", re.I),
    ),
    "authSignatureFieldNames": (
        re.compile(r"signature", re.I),
        re.compile(r"hash", re.I),
        re.compile(r"hmac", re.I),
        re.compile(r"auth", re.I),
        re.compile(r"token", re.I),
    ),
}


def validate_diagnostic_nonce(nonce: str) -> bool:
    return bool(NONCE_PATTERN.match(str(nonce or "").strip()))


def _sha256_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _is_sensitive_field_name(name: str) -> bool:
    return any(pattern.search(name) for pattern in _SENSITIVE_NAME_PATTERNS)


def _allows_full_value_for_name(name: str) -> bool:
    if _is_sensitive_field_name(name):
        return False
    return any(pattern.search(name) for pattern in _FULL_VALUE_NAME_PATTERNS)


def _looks_like_pii_value(text: str) -> bool:
    lowered = text.casefold()
    if "@" in text and "." in text:
        return True
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        return True
    return False


def sanitize_scalar_value(field_name: str, value: Any) -> Any:
    if value is None:
        return {"type": "null", "length": 0}

    if isinstance(value, (list, dict)):
        return {"type": _value_type_name(value), "length": len(value)}

    text = str(value)
    if text.startswith(SYNTHETIC_MARKER_PREFIX):
        return text

    if _allows_full_value_for_name(field_name) and not _looks_like_pii_value(text):
        return text

    return {
        "type": _value_type_name(value),
        "length": len(text),
        "sha256": _sha256_digest(text),
    }


def sanitize_field_map(fields: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in sorted(fields.keys(), key=lambda item: str(item).casefold()):
        name = str(key)
        if _is_sensitive_field_name(name):
            out[name] = {"redacted": True, "reason": "sensitive_field_name"}
            continue
        out[name] = sanitize_scalar_value(name, fields.get(key))
    return out


def collect_field_names(fields: Mapping[str, Any]) -> List[str]:
    return sorted((str(key) for key in fields.keys()), key=str.casefold)


def _flatten_field_names(prefix: str, value: Any, out: List[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.append(path)
            _flatten_field_names(path, nested, out)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        for idx, item in enumerate(value):
            _flatten_field_names(f"{prefix}[{idx}]", item, out)


def collect_nested_field_names(payload: Any) -> List[str]:
    names: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            names.append(str(key))
            _flatten_field_names(str(key), value, names)
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        for idx, item in enumerate(payload):
            _flatten_field_names(f"[{idx}]", item, names)
    return sorted(set(names), key=str.casefold)


def sanitize_headers(headers: Mapping[str, str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    names = sorted((str(key) for key in headers.keys()), key=str.casefold)
    sanitized: List[Dict[str, Any]] = []
    for name in names:
        entry: Dict[str, Any] = {"name": name}
        lower = name.casefold()
        if lower in {"cookie", "set-cookie"}:
            entry["value"] = {"redacted": True, "reason": "cookie_header"}
        elif any(pattern.search(name) for pattern in _HEADER_VALUE_METADATA_PATTERNS):
            raw = str(headers.get(name) or "")
            entry["value"] = {
                "type": "string",
                "length": len(raw),
                "sha256": _sha256_digest(raw),
            }
        sanitized.append(entry)
    return names, sanitized


def find_synthetic_markers(fields: Mapping[str, Any]) -> List[str]:
    markers: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith(SYNTHETIC_MARKER_PREFIX):
            markers.append(value)

    walk(dict(fields))
    return list(dict.fromkeys(markers))


def build_correlation_hints(all_field_names: Sequence[str]) -> Dict[str, Any]:
    hints: Dict[str, Any] = {}
    for bucket, patterns in _CORRELATION_BUCKETS.items():
        matched = sorted(
            {
                name
                for name in all_field_names
                if any(pattern.search(name) for pattern in patterns)
            },
            key=str.casefold,
        )
        hints[bucket] = matched
    return hints


def build_observation_report(
    *,
    channel: str,
    diagnostic_nonce: str,
    method: str,
    content_type: str,
    query_fields: Mapping[str, Any],
    form_fields: Mapping[str, Any],
    json_payload: Any,
    header_map: Mapping[str, str],
) -> Dict[str, Any]:
    json_field_names = collect_nested_field_names(json_payload) if json_payload is not None else []
    query_names = collect_field_names(query_fields)
    form_names = collect_field_names(form_fields)

    sanitized_json: Any = None
    if isinstance(json_payload, dict):
        sanitized_json = sanitize_field_map(json_payload)
    elif isinstance(json_payload, list):
        sanitized_json = [
            sanitize_field_map(item) if isinstance(item, dict) else sanitize_scalar_value("item", item)
            for item in json_payload
        ]

    combined_names = sorted(
        set(query_names) | set(form_names) | set(json_field_names),
        key=str.casefold,
    )
    markers = find_synthetic_markers({**dict(query_fields), **dict(form_fields)})
    if isinstance(json_payload, dict):
        markers.extend(find_synthetic_markers(json_payload))
    markers = list(dict.fromkeys(markers))

    header_names, sanitized_headers = sanitize_headers(header_map)
    correlation = build_correlation_hints(combined_names + header_names)

    report: Dict[str, Any] = {
        "channel": channel,
        "diagnosticNonce": diagnostic_nonce,
        "method": method,
        "contentType": content_type or "",
        "queryFieldNames": query_names,
        "formFieldNames": form_names,
        "jsonFieldNames": json_field_names,
        "sanitizedQuery": sanitize_field_map(query_fields),
        "sanitizedForm": sanitize_field_map(form_fields),
        "sanitizedJson": sanitized_json,
        "headerNames": header_names,
        "sanitizedHeaders": sanitized_headers,
        "syntheticMarkers": markers,
        "correlationHints": correlation,
    }
    return report


def log_icount_diagnostic_observation(report: Dict[str, Any]) -> None:
    channel = str(report.get("channel") or "").upper()
    nonce = report.get("diagnosticNonce") or ""
    method = report.get("method") or ""
    if channel == "IPN":
        logger.info(
            "ICOUNT_DIAG_IPN_RECEIVED nonce=%s method=%s contentType=%s payload=%s",
            nonce,
            method,
            report.get("contentType") or "",
            json.dumps(report, ensure_ascii=False, sort_keys=True),
        )
    elif channel == "RETURN":
        logger.info(
            "ICOUNT_DIAG_RETURN_RECEIVED nonce=%s method=%s contentType=%s payload=%s",
            nonce,
            method,
            report.get("contentType") or "",
            json.dumps(report, ensure_ascii=False, sort_keys=True),
        )
    else:
        logger.info(
            "ICOUNT_DIAG_OBSERVATION channel=%s nonce=%s payload=%s",
            channel,
            nonce,
            json.dumps(report, ensure_ascii=False, sort_keys=True),
        )

    logger.info(
        "ICOUNT_DIAG_CORRELATION nonce=%s channel=%s syntheticMarkers=%s hints=%s",
        nonce,
        channel,
        report.get("syntheticMarkers") or [],
        json.dumps(report.get("correlationHints") or {}, ensure_ascii=False, sort_keys=True),
    )


def ipn_success_response_body() -> Tuple[str, int, Dict[str, str]]:
    """Minimal HTTP 200 body many IPN providers accept as successful delivery."""
    return "OK", 200, {"Content-Type": "text/plain; charset=utf-8"}


def resolve_return_redirect_target() -> str:
    import os

    explicit = (os.environ.get("ACE_ICOUNT_DIAG_RETURN_URL") or "").strip()
    if explicit:
        return explicit
    public_base = (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if public_base:
        return f"{public_base}/"
    return "/"
