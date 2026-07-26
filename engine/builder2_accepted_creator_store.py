"""
Builder2 accepted Creator candidate persistence — immutable snapshots before Judge.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from engine.builder2_methodology_contract import METHODOLOGY_VERSION
from engine.builder2_strategy_identity import expected_strategy_foundation_id
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import load_tournament_state

logger = logging.getLogger(__name__)

ACCEPTED_CREATOR_INDEX_KEY = "acceptedCreatorCandidates"


class AcceptedCreatorCandidate(TypedDict, total=False):
    candidateId: str
    prototypeId: str
    roundIndex: int
    attemptNumber: int
    validationStatus: str
    acceptedAt: str
    strategyFoundationId: str
    creatorOutput: Dict[str, Any]
    creativeOrderContract: Dict[str, Any]
    prototypeMethodContract: Dict[str, Any]
    methodologyVersion: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_index(state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY)
    if not isinstance(index, dict):
        index = {}
        state[ACCEPTED_CREATOR_INDEX_KEY] = index
    return index


def build_accepted_creator_snapshot(
    *,
    candidate_id: str,
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    creator_output: Dict[str, Any],
    strategy_foundation: Optional[Dict[str, Any]] = None,
) -> AcceptedCreatorCandidate:
    output = deepcopy(creator_output)
    snapshot: AcceptedCreatorCandidate = {
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "roundIndex": round_index,
        "attemptNumber": attempt_number,
        "validationStatus": "accepted",
        "acceptedAt": _utc_now_iso(),
        "strategyFoundationId": expected_strategy_foundation_id(strategy_foundation or {}),
        "creatorOutput": output,
        "methodologyVersion": str(output.get("methodologyVersion") or METHODOLOGY_VERSION),
    }
    creative_order = output.get("creativeOrderContract")
    if isinstance(creative_order, dict):
        snapshot["creativeOrderContract"] = deepcopy(creative_order)
    prototype_method = output.get("prototypeMethodContract")
    if isinstance(prototype_method, dict):
        snapshot["prototypeMethodContract"] = deepcopy(prototype_method)
    return snapshot


def persist_accepted_creator_candidate(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    prototype_id: str,
    round_index: int,
    attempt_number: int,
    creator_output: Dict[str, Any],
    strategy_foundation: Optional[Dict[str, Any]] = None,
) -> AcceptedCreatorCandidate:
    index = _ensure_index(state)
    if candidate_id in index:
        logger.info(
            "BUILDER2_ACCEPTED_CREATOR_PERSISTED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s reused=true",
            state.get("jobId"),
            state.get("tournamentId"),
            candidate_id,
            prototype_id,
        )
        return index[candidate_id]

    snapshot = build_accepted_creator_snapshot(
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        round_index=round_index,
        attempt_number=attempt_number,
        creator_output=creator_output,
        strategy_foundation=strategy_foundation,
    )
    index[candidate_id] = snapshot

    cand = state.setdefault("candidates", {}).setdefault(candidate_id, {})
    cand["candidateId"] = candidate_id
    cand["creatorAcceptanceStatus"] = "accepted"
    cand["creatorSnapshot"] = deepcopy(snapshot["creatorOutput"])
    cand["creatorOutput"] = deepcopy(snapshot["creatorOutput"])
    cand.setdefault("judgeStatus", "pending")
    cand.setdefault("judgmentSnapshot", None)
    cand.setdefault("judgeFailure", None)

    logger.info(
        "BUILDER2_ACCEPTED_CREATOR_PERSISTED jobId=%s tournamentId=%s candidateId=%s prototypeId=%s roundIndex=%s attempt=%s",
        state.get("jobId"),
        state.get("tournamentId"),
        candidate_id,
        prototype_id,
        round_index,
        attempt_number,
    )
    return snapshot


def _snapshot_from_candidate_record(candidate_id: str, record: Dict[str, Any]) -> Optional[AcceptedCreatorCandidate]:
    creator_output = record.get("creatorSnapshot") or record.get("creatorOutput")
    if not isinstance(creator_output, dict) or not creator_output:
        return None
    if record.get("creatorAcceptanceStatus") not in (None, "accepted") and record.get("validationStatus") != "accepted":
        if not record.get("creatorCandidateValid"):
            return None
    return {
        "candidateId": candidate_id,
        "prototypeId": str(record.get("prototypeId") or creator_output.get("prototypeId") or ""),
        "roundIndex": int(record.get("roundIndex") or 0),
        "attemptNumber": int(record.get("attemptNumber") or 0),
        "validationStatus": "accepted",
        "acceptedAt": str(record.get("completedAt") or record.get("acceptedAt") or ""),
        "strategyFoundationId": str(creator_output.get("strategyFoundationId") or ""),
        "creatorOutput": deepcopy(creator_output),
        "methodologyVersion": str(creator_output.get("methodologyVersion") or METHODOLOGY_VERSION),
    }


def backfill_accepted_creator_index(state: Dict[str, Any]) -> int:
    index = _ensure_index(state)
    added = 0
    for candidate_id, record in (state.get("candidates") or {}).items():
        if not isinstance(record, dict) or candidate_id in index:
            continue
        snapshot = _snapshot_from_candidate_record(str(candidate_id), record)
        if snapshot is None:
            continue
        if record.get("validationStatus") == "creator_rejected":
            continue
        if record.get("creatorAcceptanceStatus") == "accepted" or record.get("creatorCandidateValid") or record.get("validationStatus") in {
            "accepted",
            "judge_unavailable",
        }:
            index[candidate_id] = snapshot
            added += 1
    if added:
        logger.info(
            "BUILDER2_ACCEPTED_CREATOR_INDEX_BACKFILLED jobId=%s tournamentId=%s count=%s",
            state.get("jobId"),
            state.get("tournamentId"),
            added,
        )
    return added


def load_accepted_creator_candidate(
    *,
    job_id: str,
    candidate_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> AcceptedCreatorCandidate:
    requested = (candidate_id or "").strip()
    if not requested:
        raise Builder2TournamentError("builder2_accepted_creator_not_found:candidateId")

    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        logger.info("BUILDER2_ACCEPTED_CREATOR_NOT_FOUND jobId=%s candidateId=%s reason=job_not_found", job_id, requested)
        raise Builder2TournamentError("builder2_accepted_creator_not_found:job")

    backfill_accepted_creator_index(state)
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    snapshot = index.get(requested)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("creatorOutput"), dict):
        logger.info(
            "BUILDER2_ACCEPTED_CREATOR_NOT_FOUND jobId=%s candidateId=%s reason=missing_snapshot",
            job_id,
            requested,
        )
        raise Builder2TournamentError("builder2_accepted_creator_not_found:candidateId")

    if str(snapshot.get("candidateId") or "") != requested:
        raise Builder2TournamentError("builder2_accepted_creator_not_found:candidateId_mismatch")

    if snapshot.get("validationStatus") != "accepted":
        raise Builder2TournamentError("builder2_accepted_creator_not_found:validationStatus")

    logger.info(
        "BUILDER2_ACCEPTED_CREATOR_LOADED jobId=%s candidateId=%s prototypeId=%s",
        job_id,
        requested,
        snapshot.get("prototypeId"),
    )
    return deepcopy(snapshot)


def find_any_persisted_accepted_candidate(
    *,
    job_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Optional[AcceptedCreatorCandidate]:
    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        return None
    backfill_accepted_creator_index(state)
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    for candidate_id in sorted(index.keys()):
        try:
            return load_accepted_creator_candidate(
                job_id=job_id,
                candidate_id=str(candidate_id),
                tournament_state=state,
            )
        except Builder2TournamentError:
            continue
    return None


def list_accepted_creator_candidate_ids(
    *,
    job_id: str,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    state = tournament_state if tournament_state is not None else load_tournament_state(job_id)
    if state is None:
        return []
    backfill_accepted_creator_index(state)
    index = state.get(ACCEPTED_CREATOR_INDEX_KEY) or {}
    return sorted(str(key) for key in index.keys())


def update_candidate_judge_state(
    state: Dict[str, Any],
    *,
    candidate_id: str,
    judge_status: str,
    failure_reason: Optional[str] = None,
    judgment_id: Optional[str] = None,
    judgment_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    cand = state.get("candidates", {}).get(candidate_id)
    if not isinstance(cand, dict):
        return
    cand["judgeStatus"] = judge_status
    if failure_reason is not None:
        cand["judgeFailure"] = failure_reason
    if judgment_id is not None:
        cand["judgmentId"] = judgment_id
    if judgment_snapshot is not None:
        cand["judgmentSnapshot"] = deepcopy(judgment_snapshot)
    if judge_status == "unavailable":
        cand["status"] = "judge_unavailable"
        cand["validationStatus"] = "judge_unavailable"
        cand["creatorCandidateValid"] = True
    elif judge_status == "accepted":
        cand["status"] = "accepted"
        cand["validationStatus"] = "accepted"
        cand["judgeFailure"] = None
