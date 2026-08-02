"""
Builder2 pending Judge structural repair — resume-safe ledger-backed state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.builder2_judge_response_ledger import (
    find_latest_attempt,
    ledger_entries,
    normal_attempts,
    repair_attempts,
)
from engine.builder2_judge_structural_repair_classifier import (
    collect_repairable_structural_failures,
    is_judge_structural_repairable,
    structural_errors_are_repairable,
)
from engine.builder2_tournament_completion_gate import accepted_judgment_index

PENDING_JUDGE_REPAIR_KEY = "pendingJudgeRepair"

REPAIR_STAGE_NORMAL_RECEIVED = "normal_response_received"
REPAIR_STAGE_NORMAL_STRUCTURAL_FAILED = "normal_response_structurally_failed"
REPAIR_STAGE_REPAIR_REQUIRED = "repair_required"
REPAIR_STAGE_REPAIR_DISPATCHED = "repair_dispatched"
REPAIR_STAGE_REPAIR_RESPONSE_RECEIVED = "repair_response_received"
REPAIR_STAGE_REPAIR_RESPONSE_PARSED = "repair_response_parsed"
REPAIR_STAGE_REPAIR_RESPONSE_VALIDATED = "repair_response_validated"
REPAIR_STAGE_REPAIR_ACCEPTED = "repair_accepted"
REPAIR_STAGE_REPAIR_FAILED_STRUCTURALLY = "repair_failed_structurally"
REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE = "repair_response_unrecoverable"

RESUME_BLOCKED_REPAIR_UNAVAILABLE = "builder2_judge_repair_response_unavailable"

_REPAIR_LEDGER_AUTHORITATIVE_KEYS = frozenset(
    {
        "repairDispatched",
        "repairAttemptId",
        "repairJudgmentId",
        "repairResponseReceived",
        "repairResponseAvailable",
        "repairResponseParsed",
        "repairResponseFingerprint",
        "repairParsedResponseFingerprint",
        "repairValidationFailureField",
        "repairValidationFailureReason",
        "lifecycleStage",
        "repairResponseAccepted",
        "repairRequired",
        "repairOutcomeUnrecoverable",
    }
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _accepted_judgment_exists(state: Dict[str, Any], candidate_id: str) -> bool:
    index = accepted_judgment_index(state, read_only=True)
    if candidate_id in index:
        return True
    record = _candidate_record(state, candidate_id)
    return bool(_clean(record.get("judgmentId")))


def _stored_pending(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = _candidate_record(state, candidate_id)
    pending = record.get(PENDING_JUDGE_REPAIR_KEY)
    return dict(pending) if isinstance(pending, dict) else {}


def _write_pending(state: Dict[str, Any], *, candidate_id: str, pending: Dict[str, Any]) -> Dict[str, Any]:
    record = _candidate_record(state, candidate_id)
    if record is not state.setdefault("candidates", {}).setdefault(candidate_id, {}):
        state.setdefault("candidates", {})[candidate_id] = record
    record[PENDING_JUDGE_REPAIR_KEY] = dict(pending)
    return record[PENDING_JUDGE_REPAIR_KEY]


def build_pending_judge_repair_state(
    *,
    normal_entry: Dict[str, Any],
    structural_failures: List[str],
    repair_dispatched: bool = False,
    repair_response_accepted: bool = False,
    lifecycle_stage: Optional[str] = None,
) -> Dict[str, Any]:
    parsed = normal_entry.get("parsedResponse") if isinstance(normal_entry.get("parsedResponse"), dict) else {}
    stage = lifecycle_stage or (
        REPAIR_STAGE_REPAIR_ACCEPTED
        if repair_response_accepted
        else REPAIR_STAGE_REPAIR_DISPATCHED
        if repair_dispatched
        else REPAIR_STAGE_REPAIR_REQUIRED
    )
    return {
        "lifecycleStage": stage,
        "normalResponsePersisted": True,
        "normalResponseParsed": bool(normal_entry.get("parsedResponseAvailable")) and bool(parsed),
        "structuralFailureDetected": True,
        "repairRequired": not repair_response_accepted,
        "repairDispatched": bool(repair_dispatched),
        "repairResponseAccepted": bool(repair_response_accepted),
        "repairResponseAvailable": False,
        "repairOutcomeUnrecoverable": False,
        "sourceJudgmentId": _clean(normal_entry.get("judgmentId")),
        "sourceAttemptId": _clean(normal_entry.get("attemptId")),
        "sourceResponseFingerprint": _clean(normal_entry.get("responseFingerprint")),
        "sourceParsedResponseFingerprint": _clean(normal_entry.get("parsedResponseFingerprint")),
        "structuralFailures": list(structural_failures or []),
        "repairAttemptId": None,
        "repairJudgmentId": None,
        "repairDispatchedAt": None,
        "repairResponseReceived": False,
        "repairResponseParsed": False,
        "repairValidationFailureField": None,
        "repairValidationFailureReason": None,
    }


def persist_pending_judge_repair(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    normal_entry: Dict[str, Any],
    structural_failures: List[str],
    repair_dispatched: bool = False,
) -> Dict[str, Any]:
    pending = build_pending_judge_repair_state(
        normal_entry=normal_entry,
        structural_failures=structural_failures,
        repair_dispatched=repair_dispatched,
        lifecycle_stage=REPAIR_STAGE_NORMAL_STRUCTURAL_FAILED if not repair_dispatched else REPAIR_STAGE_REPAIR_DISPATCHED,
    )
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_REQUIRED if not repair_dispatched else REPAIR_STAGE_REPAIR_DISPATCHED
    return _write_pending(state, candidate_id=candidate_id, pending=pending)


def _merge_pending_with_ledger(state: Dict[str, Any], candidate_id: str, pending: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(pending)
    repair = find_latest_attempt(state, candidate_id=candidate_id, call_type="repair")
    if repair:
        merged["repairDispatched"] = True
        merged["repairAttemptId"] = _clean(repair.get("attemptId")) or merged.get("repairAttemptId")
        merged["repairJudgmentId"] = _clean(repair.get("judgmentId")) or merged.get("repairJudgmentId")
        merged["repairResponseReceived"] = bool(repair.get("responseReceived"))
        merged["repairResponseAvailable"] = bool(repair.get("rawResponseAvailable") or repair.get("responseAvailable"))
        merged["repairResponseParsed"] = bool(repair.get("parsedResponseAvailable"))
        merged["repairResponseFingerprint"] = _clean(repair.get("responseFingerprint"))
        merged["repairParsedResponseFingerprint"] = _clean(repair.get("parsedResponseFingerprint"))
        merged["repairValidationFailureField"] = _clean(repair.get("validationFailureField")) or merged.get("repairValidationFailureField")
        merged["repairValidationFailureReason"] = _clean(repair.get("validationFailureReason")) or merged.get("repairValidationFailureReason")
        if repair.get("accepted"):
            merged["lifecycleStage"] = REPAIR_STAGE_REPAIR_ACCEPTED
            merged["repairResponseAccepted"] = True
            merged["repairRequired"] = False
        elif merged.get("repairResponseParsed"):
            merged["lifecycleStage"] = (
                REPAIR_STAGE_REPAIR_FAILED_STRUCTURALLY
                if repair.get("validationFailureReason")
                else REPAIR_STAGE_REPAIR_RESPONSE_PARSED
            )
    metrics = state.get("metrics") or {}
    if int(metrics.get("judgeRepairCalls") or 0) > 0 and merged.get("repairDispatched") and not repair:
        merged["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE
        merged["repairOutcomeUnrecoverable"] = True
        merged["repairResponseAvailable"] = False
    return merged


def resolve_pending_judge_repair(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    if _accepted_judgment_exists(state, candidate_id):
        return None

    stored = _stored_pending(state, candidate_id)
    normal = find_latest_attempt(state, candidate_id=candidate_id, call_type="normal")
    if normal is None and not stored.get("repairRequired"):
        return None

    if stored.get("repairResponseAccepted") or stored.get("lifecycleStage") == REPAIR_STAGE_REPAIR_ACCEPTED:
        return None

    if not stored and normal is None:
        return None

    if normal is not None:
        parsed = normal.get("parsedResponse") if isinstance(normal.get("parsedResponse"), dict) else {}
        if not parsed:
            return None
        structural_errors = list(normal.get("structuralErrors") or [])
        failures = collect_repairable_structural_failures(structural_errors, parsed=parsed)
        if not failures and not structural_errors_are_repairable(structural_errors, parsed=parsed):
            return None
        pending = build_pending_judge_repair_state(
            normal_entry=normal,
            structural_failures=failures or structural_errors,
            repair_dispatched=bool(stored.get("repairDispatched")),
            repair_response_accepted=bool(stored.get("repairResponseAccepted")),
            lifecycle_stage=_clean(stored.get("lifecycleStage")) or REPAIR_STAGE_REPAIR_REQUIRED,
        )
    else:
        pending = dict(stored)

    pending["candidateId"] = candidate_id
    pending = _merge_pending_with_ledger(state, candidate_id, pending)
    repair = find_latest_attempt(state, candidate_id=candidate_id, call_type="repair")
    for key in stored:
        if stored.get(key) in (None, "", [], {}):
            continue
        if repair and key in _REPAIR_LEDGER_AUTHORITATIVE_KEYS:
            continue
        pending[key] = stored[key]
    if pending.get("repairResponseAccepted") or pending.get("lifecycleStage") == REPAIR_STAGE_REPAIR_ACCEPTED:
        return None
    if pending.get("repairOutcomeUnrecoverable"):
        return pending
    if pending.get("repairDispatched") and repair and not repair.get("accepted"):
        return None
    if pending.get("repairDispatched"):
        return pending
    if pending.get("repairRequired"):
        return pending
    return None


def resolve_unresolved_judge_repair(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    if _accepted_judgment_exists(state, candidate_id):
        return None
    stored = _stored_pending(state, candidate_id)
    if not stored and not ledger_entries(state, candidate_id):
        return None
    normal = find_latest_attempt(state, candidate_id=candidate_id, call_type="normal")
    pending = dict(stored)
    if normal is not None and not pending:
        parsed = normal.get("parsedResponse") if isinstance(normal.get("parsedResponse"), dict) else {}
        failures = collect_repairable_structural_failures(list(normal.get("structuralErrors") or []), parsed=parsed)
        if failures or structural_errors_are_repairable(list(normal.get("structuralErrors") or []), parsed=parsed):
            pending = build_pending_judge_repair_state(normal_entry=normal, structural_failures=failures or list(normal.get("structuralErrors") or []))
    if not pending:
        return None
    pending["candidateId"] = candidate_id
    pending = _merge_pending_with_ledger(state, candidate_id, pending)
    if pending.get("repairResponseAccepted") or pending.get("lifecycleStage") == REPAIR_STAGE_REPAIR_ACCEPTED:
        return None
    if not pending.get("repairDispatched"):
        return None

    repair = find_latest_attempt(state, candidate_id=candidate_id, call_type="repair")
    if repair and repair.get("accepted"):
        return None

    if repair and pending.get("repairResponseParsed") and not repair.get("accepted"):
        from engine.builder2_judge_repair_offline_salvage import assess_repair_attempt_salvage

        salvage = assess_repair_attempt_salvage(state, candidate_id=candidate_id, entry=repair)
        pending["offlineSalvagePossible"] = bool(salvage.get("offlineSalvagePossible"))
        pending["repairOutcome"] = salvage.get("repairOutcome")
        pending["lifecycleStage"] = (
            REPAIR_STAGE_REPAIR_FAILED_STRUCTURALLY
            if salvage.get("repairOutcome") == "failed_structurally"
            else REPAIR_STAGE_REPAIR_RESPONSE_PARSED
        )
        return pending

    if pending.get("repairOutcomeUnrecoverable") or pending.get("lifecycleStage") == REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE:
        pending["repairOutcome"] = "unrecoverable"
        pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE
        return pending

    return pending


def resolve_judge_repair_resume_context(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    from engine.builder2_judge_unavailable_resolution_contract import has_operator_judgment_unavailable_resolution

    if _accepted_judgment_exists(state, candidate_id):
        return {"kind": "none", "candidateId": candidate_id}
    if has_operator_judgment_unavailable_resolution(state, candidate_id):
        return {
            "kind": "none",
            "candidateId": candidate_id,
            "operatorResolutionApplied": True,
        }
    unresolved = resolve_unresolved_judge_repair(state, candidate_id)
    if unresolved:
        outcome = _clean(unresolved.get("repairOutcome"))
        if outcome == "salvageable" or unresolved.get("offlineSalvagePossible"):
            return {"kind": "unresolved_salvageable", "candidateId": candidate_id, "pending": unresolved}
        if outcome == "failed_structurally":
            return {"kind": "unresolved_failed", "candidateId": candidate_id, "pending": unresolved}
        if outcome == "unrecoverable" or unresolved.get("repairOutcomeUnrecoverable"):
            return {
                "kind": "unrecoverable",
                "candidateId": candidate_id,
                "pending": unresolved,
                "resumeBlockedReason": RESUME_BLOCKED_REPAIR_UNAVAILABLE,
            }
        if unresolved.get("repairResponseParsed"):
            return {"kind": "unresolved_failed", "candidateId": candidate_id, "pending": unresolved}
        if unresolved.get("repairDispatched"):
            return {
                "kind": "unrecoverable",
                "candidateId": candidate_id,
                "pending": unresolved,
                "resumeBlockedReason": RESUME_BLOCKED_REPAIR_UNAVAILABLE,
            }
    pending = resolve_pending_judge_repair(state, candidate_id)
    if pending:
        return {"kind": "pending_dispatch", "candidateId": candidate_id, "pending": pending}
    return {"kind": "none", "candidateId": candidate_id}


def pending_judge_repair_candidate_ids(state: Dict[str, Any]) -> List[str]:
    pending_ids: List[str] = []
    seen = set()
    for candidate_id in sorted((state.get("candidates") or {}).keys()):
        cid = str(candidate_id)
        ctx = resolve_judge_repair_resume_context(state, cid)
        if ctx.get("kind") in {"pending_dispatch", "unresolved_salvageable", "unresolved_failed", "unrecoverable"}:
            if cid not in seen:
                seen.add(cid)
                pending_ids.append(cid)
    return pending_ids


def pending_judge_repair_prototype_ids(state: Dict[str, Any]) -> List[str]:
    prototypes: List[str] = []
    for candidate_id in pending_judge_repair_candidate_ids(state):
        prototype_id = _clean(_candidate_record(state, candidate_id).get("prototypeId"))
        if prototype_id:
            prototypes.append(prototype_id)
    return sorted(set(prototypes))


def mark_pending_repair_dispatched(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    repair_attempt_id: Optional[str] = None,
    repair_judgment_id: Optional[str] = None,
) -> None:
    pending = dict(_stored_pending(state, candidate_id) or resolve_pending_judge_repair(state, candidate_id) or {})
    if not pending:
        normal = find_latest_attempt(state, candidate_id=candidate_id, call_type="normal")
        if normal is None:
            return
        pending = build_pending_judge_repair_state(
            normal_entry=normal,
            structural_failures=list(normal.get("structuralErrors") or []),
        )
    pending["repairDispatched"] = True
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_DISPATCHED
    pending["repairDispatchedAt"] = _utc_now_iso()
    if repair_attempt_id:
        pending["repairAttemptId"] = repair_attempt_id
    if repair_judgment_id:
        pending["repairJudgmentId"] = repair_judgment_id
    _write_pending(state, candidate_id=candidate_id, pending=pending)


def mark_pending_repair_response_received(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    repair_attempt_id: str,
    repair_judgment_id: str,
) -> None:
    pending = dict(_stored_pending(state, candidate_id) or {})
    pending["repairDispatched"] = True
    pending["repairResponseReceived"] = True
    pending["repairResponseAvailable"] = True
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_RECEIVED
    pending["repairAttemptId"] = repair_attempt_id
    pending["repairJudgmentId"] = repair_judgment_id
    _write_pending(state, candidate_id=candidate_id, pending=pending)


def mark_pending_repair_response_parsed(state: Dict[str, Any], *, candidate_id: str) -> None:
    pending = dict(_stored_pending(state, candidate_id) or {})
    pending["repairResponseParsed"] = True
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_PARSED
    _write_pending(state, candidate_id=candidate_id, pending=pending)


def mark_pending_repair_failed(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    validation_failure_field: Optional[str],
    validation_failure_reason: Optional[str],
) -> None:
    pending = dict(_stored_pending(state, candidate_id) or {})
    pending["repairDispatched"] = True
    pending["repairResponseReceived"] = True
    pending["repairResponseParsed"] = True
    pending["repairValidationFailureField"] = validation_failure_field
    pending["repairValidationFailureReason"] = validation_failure_reason
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_FAILED_STRUCTURALLY
    _write_pending(state, candidate_id=candidate_id, pending=pending)


def mark_pending_repair_unrecoverable(state: Dict[str, Any], *, candidate_id: str) -> None:
    pending = dict(_stored_pending(state, candidate_id) or {})
    pending["repairDispatched"] = True
    pending["repairOutcomeUnrecoverable"] = True
    pending["repairResponseAvailable"] = False
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE
    _write_pending(state, candidate_id=candidate_id, pending=pending)


def mark_pending_repair_accepted(state: Dict[str, Any], *, candidate_id: str) -> None:
    pending = dict(_stored_pending(state, candidate_id) or {})
    pending["repairDispatched"] = True
    pending["repairResponseAccepted"] = True
    pending["repairRequired"] = False
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_ACCEPTED
    _write_pending(state, candidate_id=candidate_id, pending=pending)
    record = _candidate_record(state, candidate_id)
    record.pop(PENDING_JUDGE_REPAIR_KEY, None)


def repair_dispatch_blocked_reason(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    pending: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    from engine.builder2_judge_circuit_breaker import is_current_judge_contract_circuit_breaker_tripped

    ctx = resolve_judge_repair_resume_context(state, candidate_id)
    if ctx.get("kind") == "unrecoverable":
        return "repair_response_unrecoverable"
    if ctx.get("kind") in {"unresolved_failed", "unresolved_salvageable"}:
        return "repair_already_dispatched"
    pending = pending or ctx.get("pending") or resolve_pending_judge_repair(state, candidate_id)
    if not pending:
        return None
    if pending.get("repairDispatched"):
        return "repair_already_dispatched"
    if is_current_judge_contract_circuit_breaker_tripped(state):
        return "current_contract_circuit_breaker_tripped"
    return None


def normal_judge_call_must_not_repeat(state: Dict[str, Any], candidate_id: str) -> bool:
    ctx = resolve_judge_repair_resume_context(state, candidate_id)
    return ctx.get("kind") != "none"


def repair_judge_call_must_not_repeat(state: Dict[str, Any], candidate_id: str) -> bool:
    ctx = resolve_judge_repair_resume_context(state, candidate_id)
    return ctx.get("kind") in {"unrecoverable", "unresolved_failed", "unresolved_salvageable"}


def candidate_judgment_unresolved(state: Dict[str, Any], candidate_id: str) -> bool:
    ctx = resolve_judge_repair_resume_context(state, candidate_id)
    return ctx.get("kind") in {"unresolved_failed", "unrecoverable", "unresolved_salvageable"}
