"""
Builder2 Judge response ledger — immutable attempt records before validation.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

JUDGE_RESPONSE_LEDGER_KEY = "judgeResponseLedgerByCandidate"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def response_fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def parsed_response_fingerprint(parsed: Dict[str, Any]) -> str:
    return response_fingerprint(json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def ledger_entries(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    ledger = state.get(JUDGE_RESPONSE_LEDGER_KEY) or {}
    entries = ledger.get(candidate_id) or []
    return [item for item in entries if isinstance(item, dict)]


def find_attempt_by_id(state: Dict[str, Any], *, candidate_id: str, attempt_id: str) -> Optional[Dict[str, Any]]:
    for entry in ledger_entries(state, candidate_id):
        if _clean(entry.get("attemptId")) == _clean(attempt_id):
            return entry
    return None


def find_latest_attempt(
    state: Dict[str, Any],
    candidate_id: str,
    *,
    call_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    entries = ledger_entries(state, candidate_id)
    if call_type:
        entries = [item for item in entries if _clean(item.get("callType") or "normal") == _clean(call_type)]
    return entries[-1] if entries else None


def append_judge_response_attempt(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    judgment_id: str,
    call_type: str,
    response_text: str,
    parsed: Dict[str, Any],
    source_judgment_id: Optional[str] = None,
    source_attempt_id: Optional[str] = None,
    response_available: bool = True,
) -> str:
    attempt_id = f"judge-attempt-{candidate_id}-{uuid.uuid4().hex[:12]}"
    entry: Dict[str, Any] = {
        "attemptId": attempt_id,
        "candidateId": candidate_id,
        "judgmentId": judgment_id,
        "callType": call_type,
        "sourceAttemptId": _clean(source_attempt_id) or None,
        "sourceJudgmentId": _clean(source_judgment_id) or None,
        "responseReceived": bool(response_available and response_text),
        "rawResponseAvailable": bool(response_text),
        "responseCharacterCount": len(response_text or ""),
        "responseAvailable": bool(response_available),
        "responseFingerprint": response_fingerprint(response_text),
        "parsedResponseAvailable": bool(parsed),
        "parsedResponseFingerprint": parsed_response_fingerprint(parsed) if parsed else "",
        "parsedResponse": dict(parsed) if isinstance(parsed, dict) else {},
        "structuralValidationAttempted": False,
        "structuralValidationAccepted": False,
        "substantiveEligibilityApplied": False,
        "reportedEligible": parsed.get("eligible") if isinstance(parsed.get("eligible"), bool) else None,
        "deterministicEligible": None,
        "validationFailureField": None,
        "validationFailureReason": None,
        "structuralErrors": [],
        "accepted": False,
        "persistedAt": _utc_now_iso(),
        "recordedAt": _utc_now_iso(),
    }
    ledger = state.setdefault(JUDGE_RESPONSE_LEDGER_KEY, {})
    entries = ledger.setdefault(candidate_id, [])
    if not isinstance(entries, list):
        entries = []
        ledger[candidate_id] = entries
    entries.append(entry)
    return attempt_id


def update_judge_response_attempt(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    attempt_id: str,
    **fields: Any,
) -> None:
    entry = find_attempt_by_id(state, candidate_id=candidate_id, attempt_id=attempt_id)
    if entry is None:
        return
    entry.update(fields)
    entry["updatedAt"] = _utc_now_iso()


def finalize_judge_response_validation(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    attempt_id: str,
    judgment: Optional[Dict[str, Any]] = None,
    validation_failure_field: Optional[str] = None,
    validation_failure_reason: Optional[str] = None,
    structural_errors: Optional[List[str]] = None,
    deterministic_eligible: Optional[bool] = None,
    accepted: bool = False,
) -> None:
    parsed = judgment if isinstance(judgment, dict) else None
    update_judge_response_attempt(
        state,
        candidate_id=candidate_id,
        attempt_id=attempt_id,
        structuralValidationAttempted=True,
        structuralValidationAccepted=accepted,
        substantiveEligibilityApplied=accepted or deterministic_eligible is not None,
        reportedEligible=parsed.get("eligible") if isinstance(parsed, dict) and isinstance(parsed.get("eligible"), bool) else None,
        deterministicEligible=deterministic_eligible,
        validationFailureField=validation_failure_field,
        validationFailureReason=validation_failure_reason,
        structuralErrors=list(structural_errors or []),
        accepted=accepted,
        parsedResponse=dict(parsed) if isinstance(parsed, dict) else find_attempt_by_id(state, candidate_id=candidate_id, attempt_id=attempt_id).get("parsedResponse"),
        parsedResponseFingerprint=parsed_response_fingerprint(parsed) if isinstance(parsed, dict) and parsed else None,
    )


def repair_attempts(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    return [item for item in ledger_entries(state, candidate_id) if _clean(item.get("callType")) == "repair"]


def normal_attempts(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    return [item for item in ledger_entries(state, candidate_id) if _clean(item.get("callType") or "normal") == "normal"]
