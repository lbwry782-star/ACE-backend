"""
Builder2 Winner response ledger — persist paid responses before validation.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.builder2_judge_response_ledger import parsed_response_fingerprint, response_fingerprint
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY

WINNER_DEVELOPMENT_RESPONSE_LEDGER_KEY = "winnerDevelopmentResponseLedger"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger_entries(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    ledger = state.get(WINNER_DEVELOPMENT_RESPONSE_LEDGER_KEY) or {}
    entries = ledger.get(candidate_id) or []
    return [item for item in entries if isinstance(item, dict)]


def find_latest_winner_attempt(
    state: Dict[str, Any],
    candidate_id: str,
    *,
    call_type: str = "normal",
) -> Optional[Dict[str, Any]]:
    entries = [
        item
        for item in ledger_entries(state, candidate_id)
        if _clean(item.get("callType") or "normal") == _clean(call_type or "normal")
    ]
    return entries[-1] if entries else None


def resolve_winner_parsed_response_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    stored = _clean(payload.get("parsedResponseFingerprint"))
    parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
    derived = parsed_response_fingerprint(parsed) if parsed else ""
    effective = stored or derived
    return {
        "stored": stored or None,
        "derived": derived or None,
        "effective": effective or None,
        "storedPresent": bool(stored),
        "derivationPossible": bool(parsed) and bool(derived),
        "derivedMatchesStored": bool(stored and derived and stored == derived),
    }


def resolve_winner_response_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
    stored = _clean(payload.get("responseFingerprint"))
    raw_text = _clean(payload.get("rawResponseText"))
    derived = response_fingerprint(raw_text) if raw_text else ""
    effective = stored or derived
    return {
        "stored": stored or None,
        "derived": derived or None,
        "effective": effective or None,
        "storedPresent": bool(stored),
        "derivationPossible": bool(raw_text) and bool(derived),
        "derivedMatchesStored": bool(stored and derived and stored == derived),
    }


def backfill_winner_parsed_response_fingerprints(payload: Dict[str, Any]) -> str:
    resolved = resolve_winner_parsed_response_fingerprint(payload)
    effective = _clean(resolved.get("effective"))
    if effective and not _clean(payload.get("parsedResponseFingerprint")):
        payload["parsedResponseFingerprint"] = effective
    return effective


def backfill_winner_response_fingerprints(payload: Dict[str, Any]) -> str:
    resolved = resolve_winner_response_fingerprint(payload)
    effective = _clean(resolved.get("effective"))
    if effective and not _clean(payload.get("responseFingerprint")):
        payload["responseFingerprint"] = effective
    return effective


def record_winner_parsed_response_received(
    state: Dict[str, Any],
    *,
    parsed: Dict[str, Any],
    candidate_id: str,
    prototype_id: str,
    top_level_keys: Optional[List[str]] = None,
    response_char_count: int = 0,
    response_text: str = "",
    call_type: str = "normal",
) -> str:
    attempt_id = f"winner-attempt-{candidate_id}-{uuid.uuid4().hex[:12]}"
    parsed_fp = parsed_response_fingerprint(parsed) if isinstance(parsed, dict) else ""
    raw_fp = response_fingerprint(response_text) if response_text else ""
    payload = {
        "parsed": deepcopy(parsed),
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "topLevelKeys": list(top_level_keys or sorted(parsed.keys())),
        "topLevelKeyCount": len(parsed),
        "responseCharCount": response_char_count,
        "rawResponseText": response_text or None,
        "rawResponseAvailable": bool(response_text),
        "responseFingerprint": raw_fp or None,
        "parsedResponseFingerprint": parsed_fp or None,
        "attemptId": attempt_id,
        "callType": _clean(call_type) or "normal",
        "responseLocation": PARSED_WINNER_RESPONSE_KEY,
        "recordedAt": _utc_now_iso(),
    }
    state[PARSED_WINNER_RESPONSE_KEY] = payload
    entry = {
        "attemptId": attempt_id,
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "callType": _clean(call_type) or "normal",
        "responseReceived": True,
        "rawResponseAvailable": bool(response_text),
        "responseCharacterCount": response_char_count or len(response_text or ""),
        "responseFingerprint": raw_fp,
        "parsedResponseAvailable": isinstance(parsed, dict) and bool(parsed),
        "parsedResponseFingerprint": parsed_fp,
        "validationAttempted": False,
        "validationAccepted": False,
        "validationFailureStage": None,
        "validationFailureFieldPath": None,
        "validationFailureReason": None,
        "recordedAt": _utc_now_iso(),
    }
    ledger = state.setdefault(WINNER_DEVELOPMENT_RESPONSE_LEDGER_KEY, {})
    entries = ledger.setdefault(candidate_id, [])
    if not isinstance(entries, list):
        entries = []
        ledger[candidate_id] = entries
    entries.append(entry)
    state["winnerDevelopmentResponseReceived"] = True
    state["winnerDevelopmentParsed"] = True
    state["winnerDevelopmentLatestAttemptId"] = attempt_id
    return attempt_id


def record_winner_validation_outcome(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    attempt_id: str = "",
    accepted: bool,
    failure_stage: Optional[str] = None,
    failure_field_path: Optional[str] = None,
    failure_reason: Optional[str] = None,
    exception_class: Optional[str] = None,
) -> None:
    attempt = attempt_id or _clean(state.get("winnerDevelopmentLatestAttemptId"))
    for entry in ledger_entries(state, candidate_id):
        if attempt and _clean(entry.get("attemptId")) != attempt:
            continue
        entry["validationAttempted"] = True
        entry["validationAccepted"] = accepted
        entry["validationFailureStage"] = failure_stage
        entry["validationFailureFieldPath"] = failure_field_path
        entry["validationFailureReason"] = failure_reason
        entry["validationExceptionClass"] = exception_class
        entry["validatedAt"] = _utc_now_iso()
        break
    payload = state.get(PARSED_WINNER_RESPONSE_KEY)
    if isinstance(payload, dict) and _clean(payload.get("candidateId")) == _clean(candidate_id):
        payload["validationAttempted"] = True
        payload["validationAccepted"] = accepted
        if failure_stage:
            payload["validationFailureStage"] = failure_stage
        if failure_field_path:
            payload["validationFailureFieldPath"] = failure_field_path
        if failure_reason:
            payload["validationFailureReason"] = failure_reason
        if exception_class:
            payload["validationExceptionClass"] = exception_class
