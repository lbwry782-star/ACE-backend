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
CANCEL_REASON_INTENTIONAL_LOCAL_QUIT = "intentional_local_quit"
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
    if _truthy(raw.get("cancelRequested")):
        return True
    if _truthy(raw.get("haltRequested")):
        return True
    if _clean(raw.get("haltReason")) == CANCEL_REASON_INTENTIONAL_LOCAL_QUIT:
        return True
    return _clean(raw.get("localStopStatus")) in {"halt_requested", "uncertain", "local_stopped"}


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


def should_block_builder1_execution(job_id: str) -> bool:
    """Lasting halt of jobs included in a quit attempt. Does not freeze new admissions."""
    from engine.builder1_jobs_store import get_builder1_job

    jid = _clean(job_id)
    if not jid:
        return True
    if is_builder1_job_cancelled(jid):
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
        raw = get_builder1_job(jid)
    except Exception as exc:
        raise HaltStateUnavailable("job_hash_unreadable") from exc
    instance_id = _clean((raw or {}).get(LOCAL_DEV_INSTANCE_FIELD))
    bound = _truthy((raw or {}).get(LOCAL_DEV_BOUND_FIELD))
    if not bound or not instance_id:
        return False
    ace_root = local_builder2_worker_ace_root()
    if ace_root is None:
        from engine.ace_local_dev_coordination import resolve_product_coordination_ace_root

        ace_root = resolve_product_coordination_ace_root()
    halted, _consistency = halted_job_index(instance_id=instance_id, ace_root=ace_root)
    return jid in halted


def checkpoint_builder1_cancellation(
    job_id: str,
    *,
    campaign_id: str = "",
    stage: str = "",
) -> None:
    """Central checkpoint before paid or continuation stages. Not atomic with the vendor call."""
    from engine.ace_local_dev_config import is_local_dev_enabled

    jid = _clean(job_id)
    if is_local_dev_enabled():
        from engine.ace_local_dev_builder2_binding import HaltStateUnavailable

        try:
            blocked = should_block_builder1_execution(jid) if jid else False
        except HaltStateUnavailable:
            logger.warning("BUILDER1_JOB_HALT_STATE_UNAVAILABLE jobId=%s stage=%s", jid, stage)
            raise Builder1JobCancelledError(CANCELLED_ERROR_CODE)
        if blocked:
            raise_if_builder1_cancelled(job_id=jid, campaign_id=campaign_id, stage=stage)
            raise Builder1JobCancelledError(CANCELLED_ERROR_CODE)
    elif jid and should_block_builder1_execution(jid):
        raise_if_builder1_cancelled(job_id=jid, campaign_id=campaign_id, stage=stage)
        raise Builder1JobCancelledError(CANCELLED_ERROR_CODE)
    raise_if_builder1_cancelled(job_id=job_id, campaign_id=campaign_id, stage=stage)
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
    release_generation_lock: bool = True,
) -> Dict[str, Any]:
    """
    Idempotent Builder1 cancellation. Fast — does not wait for in-flight paid calls.

    release_generation_lock defaults True for public/frontend cancel. Intentional LOCAL
    halt keeps the lock until writer stop is proven.
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
        if release_generation_lock:
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
        record_late_builder1_vendor_result(jid, result)
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


def record_late_builder1_vendor_result(job_id: str, result: Optional[Dict[str, Any]] = None) -> None:
    """Inquiry-only. Does not mark success, retry, or clear halt."""
    from engine.builder1_jobs_store import get_builder1_job, update_builder1_job

    jid = _clean(job_id)
    if not jid or not get_builder1_job(jid):
        return
    payload = result if isinstance(result, dict) else {}
    fields: Dict[str, Any] = {
        "lateVendorReceivedAt": _utc_now_iso(),
        "lateVendorInquiryOnly": True,
    }
    image_url = _clean(payload.get("imageUrl") or payload.get("image_url") or payload.get("url"))
    if image_url:
        fields["lateVendorImageUrl"] = image_url
    ad_index = payload.get("adIndex") or payload.get("ad_index")
    if ad_index is not None:
        fields["lateVendorAdIndex"] = ad_index
    update_builder1_job(jid, **fields)
