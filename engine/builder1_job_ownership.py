"""
Builder1 job/campaign ownership — separate from Builder2 ownership fields.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

OWNER_CONTEXT_VERSION = "ace_owner_v1"
BUILDER1_BUILDER_TAG = "builder1"

_OWNERSHIP_KEYS = (
    "ownerContextRef",
    "ownerContextVersion",
    "ownerContextPresent",
    "owner_context_ref",
    "owner_context_version",
    "owner_context_present",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _hash_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_fingerprint(payload: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(payload, Mapping):
        return ""
    canonical = {
        "productDescription": _clean(payload.get("productDescription")),
        "productName": _clean(payload.get("productName")),
    }
    return _hash_token(json.dumps(canonical, sort_keys=True, ensure_ascii=False))


def extract_owner_context_from_request(request: Any) -> Dict[str, str]:
    headers = getattr(request, "headers", {}) or {}
    batch_state = _clean(headers.get("X-ACE-Batch-State"))
    authorization = _clean(headers.get("Authorization"))
    auth_ref = _hash_token(authorization) if authorization else ""
    parts = [OWNER_CONTEXT_VERSION, f"batch={batch_state}", f"auth={auth_ref}"]
    owner_ref = _hash_token("|".join(parts))
    present = "1" if (batch_state or authorization) else "0"
    return {
        "ownerContextRef": owner_ref,
        "ownerContextVersion": OWNER_CONTEXT_VERSION,
        "ownerContextPresent": present,
        "batchStatePresent": "1" if batch_state else "0",
        "authorizationPresent": "1" if authorization else "0",
    }


def ownership_fields_for_builder1_create(
    request: Any,
    payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    fields = extract_owner_context_from_request(request)
    fields["builder"] = BUILDER1_BUILDER_TAG
    fields["builder1ContractVersion"] = "builder1_production_v1"
    fields["originalRequestFingerprint"] = request_fingerprint(payload)
    return fields


def owner_context_present(record: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(record, Mapping):
        return False
    if _clean(record.get("ownerContextPresent")) in {"1", "true", "True"}:
        return True
    if _clean(record.get("owner_context_present")) in {"1", "true", "True"}:
        return True
    return bool(_clean(record.get("ownerContextRef") or record.get("owner_context_ref")))


def is_historical_without_ownership(record: Optional[Mapping[str, Any]]) -> bool:
    return not owner_context_present(record)


def verify_owner_context(
    record: Optional[Mapping[str, Any]],
    request: Any,
    *,
    allow_historical: bool = False,
) -> tuple[bool, Optional[str]]:
    if not isinstance(record, Mapping) or not record:
        return False, "not_found"
    if is_historical_without_ownership(record):
        if allow_historical:
            return True, None
        return False, "ownership_required"
    expected_ref = _clean(record.get("ownerContextRef") or record.get("owner_context_ref"))
    if not expected_ref:
        return False, "ownership_required"
    current = extract_owner_context_from_request(request)
    if current.get("ownerContextRef") != expected_ref:
        return False, "ownership_mismatch"
    return True, None


def public_owner_fields(record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"ownerContextPresent": owner_context_present(record)}


def ownership_denied_response(error_code: str) -> tuple[Dict[str, Any], int]:
    return (
        {
            "ok": False,
            "error": error_code,
        },
        403,
    )
