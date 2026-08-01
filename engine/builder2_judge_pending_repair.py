"""
Builder2 pending Judge structural repair — resume-safe ledger-backed state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.builder2_judge_structural_repair_classifier import (
    collect_repairable_structural_failures,
    is_judge_structural_repairable,
    structural_errors_are_repairable,
)
from engine.builder2_tournament_completion_gate import accepted_judgment_index

PENDING_JUDGE_REPAIR_KEY = "pendingJudgeRepair"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ledger_entries(state: Dict[str, Any], candidate_id: str) -> List[Dict[str, Any]]:
    ledger = state.get("judgeResponseLedgerByCandidate") or {}
    entries = ledger.get(candidate_id) or []
    return [item for item in entries if isinstance(item, dict)]


def _latest_normal_entry(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normals = [item for item in entries if _clean(item.get("callType") or "normal") == "normal"]
    return normals[-1] if normals else None


def _repair_entry(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    repairs = [item for item in entries if _clean(item.get("callType")) == "repair"]
    return repairs[-1] if repairs else None


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def _accepted_judgment_exists(state: Dict[str, Any], candidate_id: str) -> bool:
    index = accepted_judgment_index(state, read_only=True)
    if candidate_id in index:
        return True
    record = _candidate_record(state, candidate_id)
    return bool(_clean(record.get("judgmentId")))


def build_pending_judge_repair_state(
    *,
    normal_entry: Dict[str, Any],
    structural_failures: List[str],
    repair_dispatched: bool = False,
    repair_response_accepted: bool = False,
) -> Dict[str, Any]:
    parsed = normal_entry.get("parsedResponse") if isinstance(normal_entry.get("parsedResponse"), dict) else {}
    return {
        "normalResponsePersisted": True,
        "normalResponseParsed": bool(normal_entry.get("parsedResponseAvailable")) and bool(parsed),
        "structuralFailureDetected": True,
        "repairRequired": True,
        "repairDispatched": bool(repair_dispatched),
        "repairResponseAccepted": bool(repair_response_accepted),
        "sourceJudgmentId": _clean(normal_entry.get("judgmentId")),
        "sourceResponseFingerprint": _clean(normal_entry.get("responseFingerprint")),
        "sourceParsedResponseFingerprint": _clean(normal_entry.get("parsedResponseFingerprint")),
        "structuralFailures": list(structural_failures or []),
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
    )
    record = _candidate_record(state, candidate_id)
    if record is not state.setdefault("candidates", {}).setdefault(candidate_id, {}):
        state.setdefault("candidates", {})[candidate_id] = record
    record[PENDING_JUDGE_REPAIR_KEY] = pending
    return pending


def resolve_pending_judge_repair(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    if _accepted_judgment_exists(state, candidate_id):
        return None

    record = _candidate_record(state, candidate_id)
    stored = record.get(PENDING_JUDGE_REPAIR_KEY)
    entries = _ledger_entries(state, candidate_id)
    repair = _repair_entry(entries)
    if repair and repair.get("repairResponseAccepted"):
        return None
    if repair and _clean(repair.get("judgmentId")):
        return None

    normal = _latest_normal_entry(entries)
    if normal is None and isinstance(stored, dict) and stored.get("repairRequired"):
        pending = dict(stored)
        pending.setdefault("candidateId", candidate_id)
        return pending

    if normal is None:
        return None

    parsed = normal.get("parsedResponse") if isinstance(normal.get("parsedResponse"), dict) else {}
    if not parsed:
        return None

    structural_errors = list(normal.get("structuralErrors") or [])
    failures = collect_repairable_structural_failures(structural_errors, parsed=parsed)
    if not failures and not structural_errors_are_repairable(structural_errors, parsed=parsed):
        return None

    repair_dispatched = repair is not None or bool((stored or {}).get("repairDispatched"))
    if repair_dispatched and repair is None:
        repair_dispatched = False

    pending = build_pending_judge_repair_state(
        normal_entry=normal,
        structural_failures=failures or structural_errors,
        repair_dispatched=repair_dispatched,
    )
    pending["candidateId"] = candidate_id
    if isinstance(stored, dict):
        pending["repairDispatched"] = bool(stored.get("repairDispatched")) or repair_dispatched
    return pending


def pending_judge_repair_candidate_ids(state: Dict[str, Any]) -> List[str]:
    pending: List[str] = []
    seen = set()
    for candidate_id in sorted((state.get("candidates") or {}).keys()):
        if resolve_pending_judge_repair(state, str(candidate_id)):
            cid = str(candidate_id)
            if cid not in seen:
                seen.add(cid)
                pending.append(cid)
    return pending


def pending_judge_repair_prototype_ids(state: Dict[str, Any]) -> List[str]:
    prototypes: List[str] = []
    for candidate_id in pending_judge_repair_candidate_ids(state):
        prototype_id = _clean(_candidate_record(state, candidate_id).get("prototypeId"))
        if prototype_id:
            prototypes.append(prototype_id)
    return sorted(set(prototypes))


def mark_pending_repair_dispatched(state: Dict[str, Any], *, candidate_id: str) -> None:
    pending = resolve_pending_judge_repair(state, candidate_id)
    if not pending:
        return
    pending = dict(pending)
    pending["repairDispatched"] = True
    record = _candidate_record(state, candidate_id)
    record[PENDING_JUDGE_REPAIR_KEY] = pending


def mark_pending_repair_accepted(state: Dict[str, Any], *, candidate_id: str) -> None:
    record = _candidate_record(state, candidate_id)
    pending = dict(record.get(PENDING_JUDGE_REPAIR_KEY) or resolve_pending_judge_repair(state, candidate_id) or {})
    pending["repairDispatched"] = True
    pending["repairResponseAccepted"] = True
    pending["repairRequired"] = False
    record[PENDING_JUDGE_REPAIR_KEY] = pending
    record.pop(PENDING_JUDGE_REPAIR_KEY, None)


def repair_dispatch_blocked_reason(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    pending: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    from engine.builder2_judge_circuit_breaker import (
        is_current_judge_contract_circuit_breaker_tripped,
        legacy_breaker_evidence_excluded_count,
    )

    pending = pending or resolve_pending_judge_repair(state, candidate_id)
    if not pending:
        return None
    if pending.get("repairDispatched"):
        return "repair_already_dispatched"
    if is_current_judge_contract_circuit_breaker_tripped(state):
        return "current_contract_circuit_breaker_tripped"
    if legacy_breaker_evidence_excluded_count(state) > 0 and not is_current_judge_contract_circuit_breaker_tripped(state):
        return None
    return None


def normal_judge_call_must_not_repeat(state: Dict[str, Any], candidate_id: str) -> bool:
    return resolve_pending_judge_repair(state, candidate_id) is not None
