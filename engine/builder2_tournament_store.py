"""
Builder2 tournament Redis persistence with optimistic locking.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.builder2_methodology_contract import METHODOLOGY_VERSION, PROCESS_FAILURE_TAGS
from engine.builder2_tournament_contracts import (
    TOURNAMENT_STATE_SCHEMA_VERSION,
    Builder2TournamentError,
)
from engine.video_jobs_redis import get_redis

logger = logging.getLogger(__name__)

TOURNAMENT_KEY_PREFIX = "ace:builder2:tournament:"
_TOURNAMENT_TTL_SECONDS = 7 * 24 * 3600
_MAX_LOCK_RETRIES = 8

_memory_states: Dict[str, Dict[str, Any]] = {}
_use_memory_store: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tournament_key(job_id: str) -> str:
    return f"{TOURNAMENT_KEY_PREFIX}{(job_id or '').strip()}"


def enable_memory_store() -> None:
    global _use_memory_store, _memory_states
    _use_memory_store = True
    _memory_states = {}


def disable_memory_store() -> None:
    global _use_memory_store, _memory_states
    _use_memory_store = False
    _memory_states = {}


def use_memory_store(enabled: bool = True) -> None:
    if enabled:
        enable_memory_store()
    else:
        disable_memory_store()


def _read_raw(job_id: str) -> Optional[Dict[str, Any]]:
    if _use_memory_store:
        stored = _memory_states.get(job_id)
        return deepcopy(stored) if stored else None
    data = get_redis().get(tournament_key(job_id))
    if not data:
        return None
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise Builder2TournamentError("builder2_tournament_state_error")
    return parsed


def _write_raw(job_id: str, state: Dict[str, Any]) -> None:
    from engine.builder2_read_only_inspection import read_only_inspection_active

    if read_only_inspection_active():
        logger.warning(
            "BUILDER2_READ_ONLY_INSPECTION_BLOCKED_WRITE jobId=%s method=_write_raw",
            job_id,
        )
        return
    state["updatedAt"] = _utc_now_iso()
    payload = json.dumps(state, ensure_ascii=False)
    if _use_memory_store:
        _memory_states[job_id] = deepcopy(state)
        return
    r = get_redis()
    key = tournament_key(job_id)
    r.set(key, payload, ex=_TOURNAMENT_TTL_SECONDS)


def load_tournament_state(job_id: str, *, read_only: bool = False) -> Optional[Dict[str, Any]]:
    state = _read_raw(job_id)
    if state is not None and not read_only:
        from engine.builder2_accepted_creator_store import backfill_accepted_creator_index

        backfill_accepted_creator_index(state)
    return state


def save_tournament_state(job_id: str, state: Dict[str, Any]) -> None:
    from engine.builder2_read_only_inspection import read_only_inspection_active

    if read_only_inspection_active():
        logger.warning(
            "BUILDER2_READ_ONLY_INSPECTION_BLOCKED_WRITE jobId=%s method=save_tournament_state",
            job_id,
        )
        return
    if state.get("schemaVersion") != TOURNAMENT_STATE_SCHEMA_VERSION:
        raise Builder2TournamentError("builder2_tournament_state_error")
    _write_raw(job_id, state)


def mutate_tournament_state(
    job_id: str,
    mutator: Callable[[Dict[str, Any]], None],
    *,
    create_if_missing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    for attempt in range(_MAX_LOCK_RETRIES):
        current = _read_raw(job_id)
        if current is None:
            if create_if_missing is None:
                raise Builder2TournamentError("builder2_tournament_state_error")
            current = deepcopy(create_if_missing)
        working = deepcopy(current)
        mutator(working)
        if _use_memory_store:
            _memory_states[job_id] = deepcopy(working)
            return working
        r = get_redis()
        key = tournament_key(job_id)
        pipe = r.pipeline()
        if current is not None:
            pipe.watch(key)
        pipe.multi()
        pipe.set(key, json.dumps(working, ensure_ascii=False), ex=_TOURNAMENT_TTL_SECONDS)
        try:
            pipe.execute()
            return working
        except Exception:
            if attempt + 1 >= _MAX_LOCK_RETRIES:
                raise Builder2TournamentError("builder2_tournament_state_error") from None
            time.sleep(0.01 * (attempt + 1))
    raise Builder2TournamentError("builder2_tournament_state_error")


def new_tournament_state(
    *,
    job_id: str,
    language: str,
    active_prototype_ids: List[str],
    random_seed: str,
) -> Dict[str, Any]:
    now = _utc_now_iso()
    tournament_id = str(uuid.uuid4())
    return {
        "schemaVersion": TOURNAMENT_STATE_SCHEMA_VERSION,
        "jobId": job_id,
        "tournamentId": tournament_id,
        "status": "created",
        "randomSeed": random_seed,
        "language": language,
        "strategyFoundation": None,
        "activePrototypeIds": list(active_prototype_ids),
        "eliminatedPrototypeIds": [],
        "currentRound": 0,
        "rounds": [],
        "candidates": {},
        "acceptedCreatorCandidates": {},
        "judgments": {},
        "bestCandidateByPrototype": {},
        "winnerCandidateId": None,
        "winnerDevelopmentPlan": None,
        "initialActivePrototypeIds": list(active_prototype_ids),
        "methodologyVersion": None,
        "methodologyCompatibilityMode": False,
        "processFailureTags": [],
        "completionReason": None,
        "metrics": None,
        "strategyDiagnostics": None,
        "runway": {
            "taskId": None,
            "submissionState": "none",
            "startImageCompleted": False,
        },
        "createdAt": now,
        "updatedAt": now,
        "lastCompletedStep": "created",
        "error": None,
        "copyContractVersion": "builder2_single_slogan_v1",
        "builder2NewFormatVersion": "builder2_complete_ad_v1",
    }


def patch_tournament_state(job_id: str, patcher: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    def mutator(state: Dict[str, Any]) -> None:
        patcher(state)

    current = _read_raw(job_id)
    if current is None:
        raise Builder2TournamentError("builder2_tournament_state_error")
    return mutate_tournament_state(job_id, mutator)


def register_candidate(state: Dict[str, Any], candidate_record: Dict[str, Any]) -> None:
    cid = candidate_record["candidateId"]
    existing = state["candidates"].get(cid)
    if existing and existing.get("validationStatus") == "accepted":
        return
    state["candidates"][cid] = candidate_record


def register_judgment(state: Dict[str, Any], judgment_record: Dict[str, Any]) -> None:
    jid = judgment_record["judgmentId"]
    if jid in state["judgments"]:
        return
    state["judgments"][jid] = judgment_record


def _has_persisted_pre_methodology_completed_output(state: Dict[str, Any]) -> bool:
    if state.get("methodologyVersion") == METHODOLOGY_VERSION:
        return False
    strategy = state.get("strategyFoundation")
    if isinstance(strategy, dict) and strategy and not strategy.get("methodologyVersion"):
        return True
    for candidate in (state.get("candidates") or {}).values():
        if candidate.get("validationStatus") != "accepted":
            continue
        output = candidate.get("creatorOutput")
        if isinstance(output, dict) and output and not output.get("methodologyVersion"):
            return True
    for judgment in (state.get("judgments") or {}).values():
        payload = judgment.get("judgment")
        if isinstance(payload, dict) and payload and not payload.get("methodologyVersion"):
            return True
    winner_plan = state.get("winnerDevelopmentPlan")
    if isinstance(winner_plan, dict) and winner_plan and not winner_plan.get("methodologyVersion"):
        return True
    return False


def ensure_methodology_compatibility_decided(state: Dict[str, Any], *, is_new_job: bool) -> bool:
    """
    Decide and persist job-local methodology compatibility mode once.
    New jobs never enter compatibility mode.
    """
    if is_new_job:
        state["methodologyCompatibilityMode"] = False
        state.pop("methodologyCompatibilityReason", None)
        return False
    if state.get("methodologyCompatibilityDecided"):
        return bool(state.get("methodologyCompatibilityMode"))
    if _has_persisted_pre_methodology_completed_output(state):
        state["methodologyCompatibilityMode"] = True
        state["methodologyCompatibilityReason"] = "persisted_pre_methodology_state"
        logger.info(
            "BUILDER2_METHODOLOGY_COMPATIBILITY_MODE jobId=%s tournamentId=%s reason=%s",
            state.get("jobId"),
            state.get("tournamentId"),
            state["methodologyCompatibilityReason"],
        )
    else:
        state["methodologyCompatibilityMode"] = False
        state.pop("methodologyCompatibilityReason", None)
    state["methodologyCompatibilityDecided"] = True
    return bool(state.get("methodologyCompatibilityMode"))


def record_process_failure_tag(state: Dict[str, Any], failure_reason: Optional[str]) -> None:
    from engine.builder2_methodology_validation import infer_process_failure_tag

    tag = infer_process_failure_tag(failure_reason)
    if not tag or tag not in PROCESS_FAILURE_TAGS:
        return
    tags = state.setdefault("processFailureTags", [])
    if not isinstance(tags, list):
        tags = []
        state["processFailureTags"] = tags
    if tag not in tags:
        tags.append(tag)


def update_best_candidate_if_stronger(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    candidate_id: str,
    total_score: int,
    tie_scores: Dict[str, int],
    completed_at: str,
) -> bool:
    from engine.builder2_tournament_contracts import compare_candidate_rankings

    current_id = state["bestCandidateByPrototype"].get(prototype_id)
    if not current_id:
        state["bestCandidateByPrototype"][prototype_id] = candidate_id
        return True
    current = state["candidates"].get(current_id)
    if not current:
        state["bestCandidateByPrototype"][prototype_id] = candidate_id
        return True
    new_record = {
        "candidateId": candidate_id,
        "totalScore": total_score,
        "tieScores": tie_scores,
        "completedAt": completed_at,
        "eligible": True,
    }
    old_record = {
        "candidateId": current_id,
        "totalScore": current.get("totalScore", -1),
        "tieScores": current.get("tieScores") or {},
        "completedAt": current.get("completedAt") or "",
        "eligible": current.get("eligible", False),
    }
    if compare_candidate_rankings(new_record, old_record) <= 0:
        return False
    state["bestCandidateByPrototype"][prototype_id] = candidate_id
    return True
