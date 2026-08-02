"""
Builder2 Judge unavailable resolution contract — operator-authorized terminal outcome.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from engine.builder2_accepted_creator_store import ACCEPTED_CREATOR_INDEX_KEY
from engine.builder2_accepted_judgment_store import derive_accepted_judgment_index
from engine.builder2_tournament_config import resolve_builder2_active_prototype_ids

REPAIR_RESPONSE_UNRECOVERABLE_STAGE = "repair_response_unrecoverable"

BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_CONTRACT_VERSION = "builder2_judge_unavailable_resolution_v1"
CANDIDATE_JUDGMENT_RESOLUTION_KEY = "candidateJudgmentResolutionByCandidate"
OPERATOR_RESOLVED_JUDGMENT_UNAVAILABLE_STAGE = "operator_resolved_judgment_unavailable"
JUDGMENT_UNAVAILABLE_STATUS = "judgment_unavailable"
OPERATOR_DECISION_EXCLUDE_AND_CONTINUE = "exclude_candidate_and_continue"
RESOLUTION_REASON_REPAIR_UNAVAILABLE = "builder2_judge_repair_response_unavailable"

PRODUCTION_JOB_ID = "e369b792-9988-4087-b054-38a713966918"
PRODUCTION_TOURNAMENT_ID = "9d789e1e-7e4a-4ef4-b72e-642da8083788"
PRODUCTION_CLOSEST_CANDIDATE_ID = "cand-1-closest-1-c4ba148f"
PRODUCTION_CLOSEST_PROTOTYPE_ID = "closest"
PRODUCTION_SOURCE_JUDGMENT_ID = "judge-cand-1-closest-1-c4ba148f-017d6914"
PRODUCTION_SOURCE_RESPONSE_FINGERPRINT = (
    "cfb93d941c08dee73a47f7060007b4ce13ed8a7ea27f29d4f18423c552a47b10"
)
PRODUCTION_SOURCE_PARSED_RESPONSE_FINGERPRINT = (
    "4c110a1e7d95b05ef8f9c5abf6def5762f3d06fc7b25ed0f00d84194b23725ee"
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_record(state: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    record = (state.get("candidates") or {}).get(candidate_id) or {}
    return record if isinstance(record, dict) else {}


def prototype_id_for_candidate(state: Dict[str, Any], candidate_id: str) -> str:
    return _clean(_candidate_record(state, candidate_id).get("prototypeId"))


def judgment_resolution_index(state: Dict[str, Any]) -> Dict[str, Any]:
    index = state.get(CANDIDATE_JUDGMENT_RESOLUTION_KEY)
    return dict(index) if isinstance(index, dict) else {}


def get_candidate_judgment_resolution(state: Dict[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    resolution = judgment_resolution_index(state).get(candidate_id)
    return dict(resolution) if isinstance(resolution, dict) else None


def has_operator_judgment_unavailable_resolution(state: Dict[str, Any], candidate_id: str) -> bool:
    resolution = get_candidate_judgment_resolution(state, candidate_id)
    return bool(resolution and _clean(resolution.get("status")) == JUDGMENT_UNAVAILABLE_STATUS)


def assigned_prototype_ids(state: Dict[str, Any]) -> List[str]:
    active = state.get("initialActivePrototypeIds") or state.get("activePrototypeIds")
    if isinstance(active, list) and active:
        return [str(pid).strip() for pid in active if str(pid).strip()]
    return list(resolve_builder2_active_prototype_ids())


def _terminal_judge_prototype_ids(state: Dict[str, Any]) -> Set[str]:
    terminal: Set[str] = set()
    assigned = set(assigned_prototype_ids(state))
    for rec in (state.get("candidates") or {}).values():
        if not isinstance(rec, dict):
            continue
        prototype_id = _clean(rec.get("prototypeId"))
        if prototype_id not in assigned:
            continue
        if rec.get("validationStatus") == "judge_unavailable" or rec.get("status") == "judge_unavailable":
            terminal.add(prototype_id)
    return terminal


def accepted_judgment_count(state: Dict[str, Any], *, read_only: bool = False) -> int:
    return len(derive_accepted_judgment_index(state))


def unavailable_judgment_count(state: Dict[str, Any], *, read_only: bool = False) -> int:
    count = 0
    seen: Set[str] = set()
    for candidate_id, resolution in judgment_resolution_index(state).items():
        if _clean(resolution.get("status")) != JUDGMENT_UNAVAILABLE_STATUS:
            continue
        cid = _clean(candidate_id)
        if cid and cid not in seen:
            seen.add(cid)
            count += 1
    if count:
        return count
    return len(_terminal_judge_prototype_ids(state))


def resolved_judgment_outcome_count(state: Dict[str, Any], *, read_only: bool = False) -> int:
    return accepted_judgment_count(state, read_only=read_only) + unavailable_judgment_count(state, read_only=read_only)


def judgment_unavailable_candidate_ids(state: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()
    for candidate_id, resolution in judgment_resolution_index(state).items():
        if _clean(resolution.get("status")) != JUDGMENT_UNAVAILABLE_STATUS:
            continue
        cid = _clean(candidate_id)
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)
    if ids:
        return sorted(ids)
    for candidate_id, record in (state.get("candidates") or {}).items():
        if not isinstance(record, dict):
            continue
        if record.get("validationStatus") == "judge_unavailable" or record.get("status") == "judge_unavailable":
            cid = _clean(candidate_id)
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return sorted(ids)


def excluded_from_winner_candidate_ids(state: Dict[str, Any]) -> List[str]:
    excluded: List[str] = []
    for candidate_id in judgment_unavailable_candidate_ids(state):
        resolution = get_candidate_judgment_resolution(state, candidate_id)
        if resolution and resolution.get("excludedFromWinnerSelection") is False:
            continue
        excluded.append(candidate_id)
    return sorted(set(excluded))


def operator_resolution_required_candidate_ids(state: Dict[str, Any]) -> List[str]:
    from engine.builder2_judge_pending_repair import resolve_judge_repair_resume_context

    required: List[str] = []
    for candidate_id in sorted((state.get("candidates") or {}).keys()):
        cid = _clean(candidate_id)
        if not cid or has_operator_judgment_unavailable_resolution(state, cid):
            continue
        ctx = resolve_judge_repair_resume_context(state, cid)
        if ctx.get("kind") == "unrecoverable":
            required.append(cid)
    return required


def build_judgment_unavailable_resolution_record(
    *,
    candidate_id: str,
    prototype_id: str,
    source_judgment_id: str,
    source_response_fingerprint: str,
    source_parsed_response_fingerprint: str,
    repair_dispatch_recorded: bool,
    repair_response_available: bool,
    repair_outcome_unrecoverable: bool,
    repair_call_count_observed: int,
) -> Dict[str, Any]:
    return {
        "contractVersion": BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_CONTRACT_VERSION,
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "status": JUDGMENT_UNAVAILABLE_STATUS,
        "resolutionReason": RESOLUTION_REASON_REPAIR_UNAVAILABLE,
        "operatorDecision": OPERATOR_DECISION_EXCLUDE_AND_CONTINUE,
        "operatorAuthorized": True,
        "sourceJudgmentId": source_judgment_id,
        "sourceResponseFingerprint": source_response_fingerprint,
        "sourceParsedResponseFingerprint": source_parsed_response_fingerprint,
        "repairDispatchRecorded": bool(repair_dispatch_recorded),
        "repairResponseAvailable": bool(repair_response_available),
        "repairOutcomeUnrecoverable": bool(repair_outcome_unrecoverable),
        "repairCallCountObserved": int(repair_call_count_observed or 0),
        "excludedFromWinnerSelection": True,
        "additionalPaidCallAuthorized": False,
        "resolvedAt": _utc_now_iso(),
    }


def persist_candidate_judgment_resolution(state: Dict[str, Any], *, candidate_id: str, resolution: Dict[str, Any]) -> Dict[str, Any]:
    index = state.setdefault(CANDIDATE_JUDGMENT_RESOLUTION_KEY, {})
    if not isinstance(index, dict):
        index = {}
        state[CANDIDATE_JUDGMENT_RESOLUTION_KEY] = index
    index[candidate_id] = dict(resolution)
    return index[candidate_id]


def is_creator_accepted_for_candidate(state: Dict[str, Any], candidate_id: str) -> bool:
    record = _candidate_record(state, candidate_id)
    if record.get("validationStatus") == "accepted" or record.get("creatorAcceptanceStatus") == "accepted":
        return True
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    entry = index.get(candidate_id)
    return isinstance(entry, dict) and _clean(entry.get("validationStatus") or "accepted") == "accepted"


def accepted_judgment_exists_for_candidate(state: Dict[str, Any], candidate_id: str) -> bool:
    if _clean(_candidate_record(state, candidate_id).get("judgmentId")):
        return True
    return candidate_id in derive_accepted_judgment_index(state)


def is_reasoning_recovery_paused_state(state: Dict[str, Any]) -> bool:
    status = _clean(state.get("status")).lower()
    progress = _clean(state.get("progressStage"))
    if status in {"failed", "error", "paused_for_reasoning_resume"}:
        return True
    return progress in {"mixed_partial_reasoning", "judge_generation", "creator_generation"}


def media_started(state: Dict[str, Any]) -> bool:
    if bool(state.get("mediaStarted")):
        return True
    media = state.get("mediaResume")
    if not isinstance(media, dict):
        return False
    return bool(_clean(media.get("startImageArtifact")) or _clean(media.get("runwayTaskId")))


def winner_or_media_started(state: Dict[str, Any]) -> bool:
    if media_started(state):
        return True
    if _clean(state.get("winnerCandidateId")):
        return True
    if _clean(state.get("winningCandidateId")):
        return True
    return bool(state.get("winnerDevelopmentPlan"))


def is_reasoning_complete_for_winner_selection(state: Dict[str, Any], *, read_only: bool = False) -> bool:
    from engine.builder2_tournament_completion_gate import (
        is_tournament_ready_for_winner_selection,
        missing_actionable_judge_prototype_ids,
        missing_creator_prototype_ids,
    )

    if len(assigned_prototype_ids(state)) < 6:
        return False
    if missing_creator_prototype_ids(state, read_only=read_only):
        return False
    if missing_actionable_judge_prototype_ids(state, read_only=read_only):
        return False
    return is_tournament_ready_for_winner_selection(state, read_only=read_only)


def pending_repair_is_unrecoverable(state: Dict[str, Any], candidate_id: str) -> bool:
    from engine.builder2_judge_pending_repair import resolve_judge_repair_resume_context

    ctx = resolve_judge_repair_resume_context(state, candidate_id)
    if ctx.get("kind") != "unrecoverable":
        return False
    pending = ctx.get("pending") or {}
    return bool(
        pending.get("repairOutcomeUnrecoverable")
        or pending.get("lifecycleStage") == REPAIR_RESPONSE_UNRECOVERABLE_STAGE
    )
