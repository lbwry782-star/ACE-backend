"""
Builder1 cooperative cancellation — job + campaign durable state.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CANCELLED_ERROR_CODE = "builder1_job_cancelled"
CANCEL_REASON_FRONTEND_REFRESH = "frontend_refresh"
CAMPAIGN_CANCELLED_ERROR_CODE = "builder1_campaign_cancelled"


class Builder1JobCancelledError(Exception):
    """Raised when a Builder1 job or campaign is cancelled and work must stop."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes"}


def _clean(raw: Any) -> str:
    return str(raw or "").strip()


def is_builder1_job_record(job: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(job, dict):
        return False
    return _clean(job.get("builder")) == "builder1" or bool(_clean(job.get("builder1ContractVersion")))


def is_builder1_job_cancelled(job_id: str) -> bool:
    from engine.builder1_jobs_store import get_builder1_job

    raw = get_builder1_job(_clean(job_id))
    if not raw:
        return False
    if _clean(raw.get("status")) == "cancelled":
        return True
    return _truthy(raw.get("cancelRequested"))


def is_builder1_campaign_cancelled(campaign_id: str) -> bool:
    from engine.builder1_campaign_store import get_campaign_session_raw

    raw = get_campaign_session_raw(_clean(campaign_id))
    if not raw:
        return False
    if _truthy(raw.get("cancelRequested")):
        return True
    if _clean(raw.get("campaignLifecycleStatus")) == "cancelled":
        return True
    return False


def raise_if_builder1_cancelled(
    *,
    job_id: str = "",
    campaign_id: str = "",
    stage: str = "",
) -> None:
    jid = _clean(job_id)
    cid = _clean(campaign_id)
    if cid and is_builder1_campaign_cancelled(cid):
        if stage:
            logger.info(
                "BUILDER1_CAMPAIGN_CANCEL_CHECKPOINT campaignId=%s stage=%s cancelled=true",
                cid,
                stage,
            )
        raise Builder1JobCancelledError(CAMPAIGN_CANCELLED_ERROR_CODE)
    if jid and is_builder1_job_cancelled(jid):
        if stage:
            logger.info(
                "BUILDER1_JOB_CANCEL_CHECKPOINT jobId=%s stage=%s cancelled=true",
                jid,
                stage,
            )
        raise Builder1JobCancelledError(CANCELLED_ERROR_CODE)


def checkpoint_builder1_cancellation(
    job_id: str,
    *,
    campaign_id: str = "",
    stage: str = "",
) -> None:
    """Central checkpoint before paid or continuation stages."""
    raise_if_builder1_cancelled(job_id=job_id, campaign_id=campaign_id, stage=stage)
    jid = _clean(job_id)
    if jid:
        from engine.builder1_jobs_store import touch_builder1_job_heartbeat

        touch_builder1_job_heartbeat(jid)


def _mark_campaign_cancelled(campaign_id: str, *, reason: str) -> None:
    from engine.builder1_campaign_store import mark_campaign_cancelled

    mark_campaign_cancelled(campaign_id, reason=reason)


def _release_campaign_lock_for_job(campaign_id: str, job_id: str) -> None:
    from engine.builder1_campaign_store import release_generation_lock_for_cancelled_job

    release_generation_lock_for_cancelled_job(campaign_id, job_id=job_id)


def request_builder1_job_cancellation(
    job_id: str,
    *,
    reason: str = CANCEL_REASON_FRONTEND_REFRESH,
) -> Dict[str, Any]:
    """
    Idempotent Builder1 cancellation. Fast — does not wait for in-flight paid calls.
    """
    from engine.builder1_jobs_store import get_builder1_job, update_builder1_job

    jid = _clean(job_id)
    if not jid:
        return {"ok": False, "error": "missing_job_id", "jobId": jid}

    raw = get_builder1_job(jid)
    if not raw:
        return {"ok": False, "error": "not_found", "jobId": jid}

    if not is_builder1_job_record(raw):
        return {"ok": False, "error": "not_builder1_job", "jobId": jid}

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

    now_iso = _utc_now_iso()
    now_epoch = time.time()
    update_builder1_job(
        jid,
        cancelRequested=True,
        cancelRequestedAt=now_iso,
        cancelReason=_clean(reason) or CANCEL_REASON_FRONTEND_REFRESH,
        cancelledAt=now_iso,
        status="cancelled",
        stage="cancelled",
        lastHeartbeatAt=now_epoch,
    )

    campaign_id = _clean(raw.get("campaignId"))
    if campaign_id:
        _mark_campaign_cancelled(campaign_id, reason=_clean(reason) or CANCEL_REASON_FRONTEND_REFRESH)
        _release_campaign_lock_for_job(campaign_id, jid)

    logger.info(
        "BUILDER1_JOB_CANCELLED jobId=%s campaignId=%s reason=%s",
        jid,
        campaign_id or None,
        reason,
    )
    return {
        "ok": True,
        "outcome": "cancelled",
        "jobId": jid,
        "status": "cancelled",
        "cancelRequestedAt": now_iso,
        "cancelReason": _clean(reason) or CANCEL_REASON_FRONTEND_REFRESH,
        "cancelledAt": now_iso,
        "campaignId": campaign_id or None,
    }


def finalize_builder1_job_respecting_cancellation(
    job_id: str,
    result: dict[str, Any],
    *,
    target_ad_count: int,
) -> bool:
    """
    Finalize only if cancellation was not recorded first.
    Returns True if finalized, False if job stays cancelled (completion race).
    """
    jid = _clean(job_id)
    if not jid:
        return False
    if is_builder1_job_cancelled(jid):
        logger.info("BUILDER1_JOB_CANCEL_COMPLETION_RACE jobId=%s winner=cancelled", jid)
        return False
    campaign_id = _clean(result.get("campaignId"))
    if campaign_id and is_builder1_campaign_cancelled(campaign_id):
        logger.info(
            "BUILDER1_CAMPAIGN_CANCEL_COMPLETION_RACE jobId=%s campaignId=%s winner=cancelled",
            jid,
            campaign_id,
        )
        return False
    from engine.builder1_jobs_store import finalize_builder1_job

    finalize_builder1_job(jid, result, target_ad_count=target_ad_count)
    return True
