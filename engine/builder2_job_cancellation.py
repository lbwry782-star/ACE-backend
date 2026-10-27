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
CANCEL_REASON_INTENTIONAL_LOCAL_QUIT = "intentional_local_quit"

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
    if _truthy(raw.get("cancelRequested")):
        return True
    if _truthy(raw.get("haltRequested")):
        return True
    if _clean(raw.get("haltReason")) == CANCEL_REASON_INTENTIONAL_LOCAL_QUIT:
        return True
    return _clean(raw.get("localStopStatus")) in {"halt_requested", "uncertain", "local_stopped"}


def raise_if_builder2_cancelled(job_id: str) -> None:
    jid = _clean(job_id)
    if jid and is_builder2_job_cancelled(jid):
        raise Builder2JobCancelledError(CANCELLED_ERROR_CODE)


def checkpoint_builder2_cancellation(job_id: str, *, stage: str = "") -> None:
    """Central checkpoint before paid or continuation stages."""
    jid = _clean(job_id)
    if not jid:
        return
    from engine.ace_local_dev_config import is_local_dev_enabled

    if is_local_dev_enabled():
        from engine.ace_local_dev_builder2_binding import HaltStateUnavailable

        try:
            blocked = should_block_builder2_execution(jid)
        except HaltStateUnavailable:
            logger.warning("BUILDER2_JOB_CANCEL_CHECKPOINT jobId=%s stage=%s halt_state_unavailable", jid, stage)
            raise Builder2JobCancelledError(CANCELLED_ERROR_CODE)
    else:
        blocked = should_block_builder2_execution(jid)
    if blocked:
        if stage:
            logger.info("BUILDER2_JOB_CANCEL_CHECKPOINT jobId=%s stage=%s cancelled=true", jid, stage)
        raise Builder2JobCancelledError(CANCELLED_ERROR_CODE)


def _persist_tournament_cancelled(job_id: str, *, reason: str = CANCEL_REASON_FRONTEND_REFRESH) -> None:
    state = load_tournament_state(job_id)
    if not isinstance(state, dict):
        return
    state["status"] = "cancelled"
    state["canResume"] = False
    state["cancelReason"] = _clean(reason) or CANCEL_REASON_FRONTEND_REFRESH
    save_tournament_state(job_id, state)


def request_builder2_job_cancellation(
    job_id: str,
    *,
    reason: str = CANCEL_REASON_FRONTEND_REFRESH,
    expire_execution_lease: bool = True,
) -> Dict[str, Any]:
    """
    Idempotent Builder2 cancellation. Fast — does not wait for in-flight paid calls.

    expire_execution_lease defaults True for public/frontend cancel. Intentional LOCAL
    halt keeps the execution lease until writer stop is proven.

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
        remove_recoverable_job(jid, expire_lease=expire_execution_lease)
    except Exception:
        pass

    try:
        _persist_tournament_cancelled(jid, reason=_clean(reason) or CANCEL_REASON_FRONTEND_REFRESH)
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


def should_block_builder2_execution(job_id: str) -> bool:
    """Lasting halt of jobs included in a quit attempt. Does not freeze new admissions."""
    jid = _clean(job_id)
    if not jid:
        return True
    if is_builder2_job_cancelled(jid):
        return True
    from engine.ace_local_dev_config import is_local_dev_enabled

    if not is_local_dev_enabled():
        return False
    from engine.ace_local_dev_builder2_binding import (
        HaltStateUnavailable,
        LOCAL_DEV_BOUND_FIELD,
        LOCAL_DEV_INSTANCE_FIELD,
        local_builder2_worker_ace_root,
    )
    from engine.ace_local_dev_builder2_halt_registry import halted_job_index

    try:
        raw = video_job_get_raw(jid)
    except Exception as exc:
        raise HaltStateUnavailable("job_hash_unreadable") from exc
    instance_id = _clean((raw or {}).get(LOCAL_DEV_INSTANCE_FIELD))
    bound = _truthy((raw or {}).get(LOCAL_DEV_BOUND_FIELD))
    if not bound or not instance_id:
        return False
    ace_root = local_builder2_worker_ace_root()
    halted, _consistency = halted_job_index(instance_id=instance_id, ace_root=ace_root)
    return jid in halted


def accept_dequeued_builder2_job(job_id: str) -> Dict[str, Any]:
    """
    Gate after a queue pop. Does not call vendors. Refuses cancelled/halted jobs
    without re-enqueue. Halt-state read failure does not continue work.
    """
    from engine.ace_local_dev_config import is_local_dev_enabled

    jid = _clean(job_id)
    if not jid:
        return {"accepted": False, "reason": "missing_job_id", "requeued": False}
    if is_local_dev_enabled():
        from engine.ace_local_dev_builder2_binding import (
            HaltStateUnavailable,
            classify_builder2_local_affiliation,
            local_builder2_worker_ace_root,
        )

        local_root = local_builder2_worker_ace_root()
        if local_root is not None:
            raw = video_job_get_raw(jid) or {}
            affiliation = classify_builder2_local_affiliation(raw, local_root)
            if affiliation == "foreign_local":
                logger.info("BUILDER2_JOB_DEQUEUE_BLOCKED jobId=%s reason=foreign_local_instance", jid)
                return {"accepted": False, "reason": "foreign_local_instance", "requeued": False, "jobId": jid, "mutated": False}
            if affiliation == "unbound":
                logger.info("BUILDER2_JOB_DEQUEUE_BLOCKED jobId=%s reason=local_affiliation_unknown", jid)
                return {
                    "accepted": False,
                    "reason": "local_affiliation_unknown",
                    "requeued": False,
                    "jobId": jid,
                    "mutated": False,
                }
            if affiliation == "missing":
                return {"accepted": False, "reason": "missing_job", "requeued": False, "jobId": jid, "mutated": False}
        try:
            blocked = should_block_builder2_execution(jid)
        except HaltStateUnavailable:
            logger.warning("BUILDER2_JOB_DEQUEUE_BLOCKED jobId=%s reason=halt_state_unavailable", jid)
            return {"accepted": False, "reason": "halt_state_unavailable", "requeued": False, "jobId": jid}
    else:
        blocked = should_block_builder2_execution(jid)
    if blocked:
        logger.info("BUILDER2_JOB_DEQUEUE_BLOCKED jobId=%s reason=cancelled_or_halted", jid)
        return {"accepted": False, "reason": "cancelled_or_halted", "requeued": False, "jobId": jid}
    return {"accepted": True, "reason": "ok", "requeued": False, "jobId": jid}


def try_requeue_builder2_after_lease_fail(job_id: str) -> bool:
    """Worker lease-fail path: never auto-retry a cancelled/halted job."""
    from engine.ace_local_dev_config import is_local_dev_enabled

    jid = _clean(job_id)
    if not jid:
        return False
    if is_local_dev_enabled():
        from engine.ace_local_dev_builder2_binding import HaltStateUnavailable

        try:
            if should_block_builder2_execution(jid):
                return False
        except HaltStateUnavailable:
            return False
    elif should_block_builder2_execution(jid):
        return False
    from engine.builder2_tournament_recovery import mark_job_queued, register_recoverable_job

    if not mark_job_queued(jid):
        return False
    try:
        from engine.video_jobs_redis import video_job_lpush

        video_job_lpush(jid)
    except Exception:
        return False
    register_recoverable_job(jid)
    return True


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
    if raw and (
        _clean(raw.get("status")) == "cancelled"
        or _truthy(raw.get("cancelRequested"))
        or _truthy(raw.get("haltRequested"))
    ):
        logger.info("BUILDER2_JOB_CANCEL_COMPLETION_RACE jobId=%s winner=cancelled", jid)
        record_late_builder2_vendor_result(
            jid,
            video_url=video_url,
            marketing_text=marketing_text,
            overlay_headline=overlay_headline,
        )
        return False
    if raw and _clean(raw.get("status")) == "done":
        return True
    from engine.video_jobs_redis import video_job_mark_done

    video_job_mark_done(jid, video_url, marketing_text, overlay_headline)
    return True


def record_late_builder2_vendor_result(
    job_id: str,
    *,
    video_url: str = "",
    marketing_text: str = "",
    overlay_headline: str = "",
    provider_job_id: str = "",
) -> bool:
    """
    Keep vendor payload for inquiry after halt. Does not mark the job done,
    does not start a new stage, and does not clear halt/cancel fields.
    """
    jid = _clean(job_id)
    if not jid:
        return False
    raw = video_job_get_raw(jid) or {}
    if _clean(raw.get("status")) == "done":
        return False
    mapping = {
        "lateVendorReceivedAt": _utc_now_iso(),
        "lateVendorVideoUrl": video_url or "",
        "lateVendorMarketingText": marketing_text or "",
        "lateVendorOverlayHeadline": overlay_headline or "",
    }
    if provider_job_id:
        mapping["lateVendorProviderJobId"] = provider_job_id
    if video_url and not _clean(raw.get("video_url")):
        mapping["inquiryVideoUrl"] = video_url
    from engine.builder2_tournament_recovery import _use_memory_recovery, set_memory_job_hash
    from engine.video_jobs_redis import _use_memory_jobs, job_key, get_redis

    if _use_memory_recovery or _use_memory_jobs:
        updated = dict(raw)
        updated.update(mapping)
        set_memory_job_hash(jid, updated)
        return True
    get_redis().hset(job_key(jid), mapping=mapping)
    return True
