"""
Builder2 job ownership — bind immutable jobs to existing request/session context.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional

OWNER_CONTEXT_VERSION = "ace_owner_v1"

_OWNERSHIP_FIELD_NAMES = (
    "ownerContextRef",
    "ownerContextVersion",
    "ownerContextPresent",
    "owner_context_ref",
    "owner_context_version",
    "owner_context_present",
    "user_id",
    "userId",
    "session_id",
    "sessionId",
    "owner_id",
    "ownerId",
    "account_id",
    "accountId",
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
    batch_state = _clean(getattr(request, "headers", {}).get("X-ACE-Batch-State"))
    authorization = _clean(getattr(request, "headers", {}).get("Authorization"))
    auth_ref = _hash_token(authorization) if authorization else ""
    parts = [OWNER_CONTEXT_VERSION, f"batch={batch_state}", f"auth={auth_ref}"]
    owner_ref = _hash_token("|".join(parts))
    return {
        "ownerContextRef": owner_ref,
        "ownerContextVersion": OWNER_CONTEXT_VERSION,
        "ownerContextPresent": "1" if (batch_state or authorization) else "0",
        "batchStatePresent": "1" if batch_state else "0",
        "authorizationPresent": "1" if authorization else "0",
    }


def ownership_fields_for_job_create(request: Any, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    fields = extract_owner_context_from_request(request)
    fields["builder"] = "builder2"
    fields["builder2ResumeContractVersion"] = "builder2_resume_v1"
    fields["originalRequestFingerprint"] = request_fingerprint(payload)
    return fields


def owner_context_present_in_job(job_hash: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(job_hash, Mapping):
        return False
    if _clean(job_hash.get("ownerContextPresent")) in {"1", "true", "True"}:
        return True
    if _clean(job_hash.get("owner_context_present")) in {"1", "true", "True"}:
        return True
    if _clean(job_hash.get("ownerContextRef") or job_hash.get("owner_context_ref")):
        return True
    for key in _OWNERSHIP_FIELD_NAMES:
        if key in {"ownerContextPresent", "owner_context_present", "ownerContextRef", "owner_context_version", "ownerContextVersion"}:
            continue
        if _clean(job_hash.get(key)):
            return True
    return False


def is_historical_job_without_ownership(job_hash: Optional[Mapping[str, Any]]) -> bool:
    return not owner_context_present_in_job(job_hash)


def verify_owner_context(
    job_hash: Optional[Mapping[str, Any]],
    request: Any,
    *,
    allow_historical_admin: bool = False,
) -> tuple[bool, Optional[str]]:
    if not isinstance(job_hash, Mapping) or not job_hash:
        return False, "job_not_found"
    if is_historical_job_without_ownership(job_hash):
        if allow_historical_admin:
            return True, None
        return False, "ownership_required_historical_job"
    expected_ref = _clean(job_hash.get("ownerContextRef") or job_hash.get("owner_context_ref"))
    if not expected_ref:
        return False, "ownership_required"
    current = extract_owner_context_from_request(request)
    if current.get("ownerContextRef") != expected_ref:
        return False, "ownership_mismatch"
    return True, None


def public_owner_fields(job_hash: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "ownerContextPresent": owner_context_present_in_job(job_hash),
    }
