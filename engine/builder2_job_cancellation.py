"""
Builder2 job cancellation — idempotent cancel requests, checkpoints, resume protection.

Builder1 jobs are not affected. Cancellation is recorded in Redis immediately; workers
check cancelRequested before every paid stage.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_recovery import (
    TERMINAL_JOB_STATUSES,
    clear_job_queued,
    remove_recoverable_job,
)
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import (
    get_redis,
    job_key,
    video_job_get_raw,
    video_job_remove_from_queue,
)

logger = logging.getLogger(__name__)

CANCELLED_ERROR_CODE = "builder2_job_cancelled"
CANCEL_REASON_FRONTEND_REFRESH = "frontend_refresh"

_TERMINAL_CANCEL_OUTCOMES = frozenset({"cancelled", "already_cancelled", "already_completed", "already_terminal"})


class Builder2JobCancelledError(Builder2TournamentError):
    """Raised when a Builder2 job has a recorded cancellation and must stop work."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes"}


def _clean(raw: Any) -> str:
    return str(raw or "").strip()


def is_builder2_job_hash(job_hash: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(job_hash, dict):
        return False
    return _clean(job_hash.get("builder")) == "builder2" or bool(_clean(job_hash.get("builder2ResumeContractVersion")))


def is_builder2_job_cancelled(job_id: str) -> bool:
    raw = video_job_get_raw(_clean(job_id))
    if not raw:
        return False
    if _clean(raw.get("status")) == "cancelled":
        return True
    return _truthy(raw.get("cancelRequested"))


def raise_if_builder2_cancelled(job_id: str) -> None:
    jid = _clean(job_id)
    if jid and is_builder2_job_cancelled(jid):
        raise Builder2JobCancelledError(CANCELLED_ERROR_CODE)


def checkpoint_builder2_cancellation(job_id: str, *, stage: str = "") -> None:
    """Central checkpoint before paid or continuation stages."""
    jid = _clean(job_id)
    if not jid:
        return
    if is_builder2_job_cancelled(jid):
        if stage:
            logger.info("BUILDER2_JOB_CANCEL_CHECKPOINT jobId=%s stage=%s cancelled=true", jid, stage)
        raise Builder2JobCancelledError(CANCELLED_ERROR_CODE)


def _persist_tournament_cancelled(job_id: str) -> None:
    state = load_tournament_state(job_id)
    if not isinstance(state, dict):
        return
    state["status"] = "cancelled"
    state["canResume"] = False
    state["cancelReason"] = CANCEL_REASON_FRONTEND_REFRESH
    save_tournament_state(job_id, state)


def request_builder2_job_cancellation(
    job_id: str,
    *,
    reason: str = CANCEL_REASON_FRONTEND_REFRESH,
) -> Dict[str, Any]:
    """
    Idempotent Builder2 cancellation. Fast — does not wait for in-flight paid calls.

    Returns outcome: cancelled | already_cancelled | already_completed | already_terminal
    """
    jid = _clean(job_id)
    if not jid:
        return {"ok": False, "error": "missing_job_id", "jobId": jid}

    raw = video_job_get_raw(jid)
    if not raw:
        return {"ok": False, "error": "not_found", "jobId": jid}

    if not is_builder2_job_hash(raw):
        return {"ok": False, "error": "not_builder2_job", "jobId": jid}

    status = _clean(raw.get("status"))
    if status == "done":
        return {
            "ok": True,
            "outcome": "already_completed",
            "jobId": jid,
            "status": status,
        }
    if status == "cancelled" or _truthy(raw.get("cancelRequested")):
        return {
            "ok": True,
            "outcome": "already_cancelled",
            "jobId": jid,
            "status": "cancelled",
            "cancelRequestedAt": raw.get("cancelRequestedAt"),
            "cancelReason": raw.get("cancelReason"),
        }
    if status in TERMINAL_JOB_STATUSES:
        return {
            "ok": True,
            "outcome": "already_terminal",
            "jobId": jid,
            "status": status,
        }

    now_iso = _utc_now_iso()
    now_epoch = str(int(time.time()))
    mapping = {
        "cancelRequested": "1",
        "cancelRequestedAt": now_iso,
        "cancelReason": _clean(reason) or CANCEL_REASON_FRONTEND_REFRESH,
        "status": "cancelled",
        "cancelledAt": now_iso,
        "canResume": "0",
        "last_progress_ts": now_epoch,
        "progressStage": "cancelled",
        "error": "",
    }

    from engine.builder2_tournament_recovery import _use_memory_recovery, set_memory_job_hash

    if _use_memory_recovery:
        updated = dict(raw)
        updated.update({k: str(v) for k, v in mapping.items()})
        set_memory_job_hash(jid, updated)
    else:
        get_redis().hset(job_key(jid), mapping=mapping)

    try:
        removed = video_job_remove_from_queue(jid)
        if removed:
            logger.info("BUILDER2_JOB_CANCEL_QUEUE_REMOVED jobId=%s count=%s", jid, removed)
    except Exception as exc:
        logger.warning("BUILDER2_JOB_CANCEL_QUEUE_REMOVE_FAIL jobId=%s err=%s", jid, exc)

    try:
        clear_job_queued(jid)
    except Exception:
        pass
    try:
        remove_recoverable_job(jid)
    except Exception:
        pass

    try:
        _persist_tournament_cancelled(jid)
    except Exception as exc:
        logger.warning("BUILDER2_JOB_CANCEL_TOURNAMENT_PERSIST_FAIL jobId=%s err=%s", jid, exc)

    logger.info(
        "BUILDER2_JOB_CANCELLED jobId=%s reason=%s cancelRequestedAt=%s",
        jid,
        mapping["cancelReason"],
        now_iso,
    )
    return {
        "ok": True,
        "outcome": "cancelled",
        "jobId": jid,
        "status": "cancelled",
        "cancelRequestedAt": now_iso,
        "cancelReason": mapping["cancelReason"],
        "cancelledAt": now_iso,
    }


def video_job_mark_done_respecting_cancellation(
    job_id: str,
    video_url: str,
    marketing_text: str,
    overlay_headline: str = "",
) -> bool:
    """
    Mark job done only if cancellation was not recorded first.
    Returns True if done was written, False if job stays cancelled (completion race).
    """
    jid = _clean(job_id)
    if not jid:
        return False
    raw = video_job_get_raw(jid)
    if raw and (_clean(raw.get("status")) == "cancelled" or _truthy(raw.get("cancelRequested"))):
        logger.info("BUILDER2_JOB_CANCEL_COMPLETION_RACE jobId=%s winner=cancelled", jid)
        return False
    from engine.video_jobs_redis import video_job_mark_done

    video_job_mark_done(jid, video_url, marketing_text, overlay_headline)
    return True
