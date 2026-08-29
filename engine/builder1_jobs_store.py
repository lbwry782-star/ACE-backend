"""
Builder1 job status storage (Redis with in-memory fallback).

Job records are keyed only by jobId and must never overwrite another job or campaign.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

JOB_KEY_PREFIX = "builder1:job:"
JOB_TTL_SECONDS = 24 * 3600
BUILDER1_JOB_STALE_SECONDS = int(
    (os.environ.get("BUILDER1_JOB_STALE_SECONDS") or "900").strip() or "900"
)

_memory_lock = threading.Lock()
_memory_jobs: Dict[str, Dict[str, Any]] = {}


def _redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def get_builder1_job_store_backend() -> str:
    return "redis" if _redis_configured() else "memory"


def _job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def create_builder1_job(
    *,
    job_id: str,
    campaign_id: str,
    target_ad_count: int,
    stage: str = "planning",
    ownership_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    jid = (job_id or "").strip()
    cid = (campaign_id or "").strip()
    if not jid or not cid:
        raise ValueError("missing_job_or_campaign_id")
    now = time.time()
    entry: Dict[str, Any] = {
        "status": "running",
        "stage": stage,
        "completedAds": 0,
        "totalAds": int(target_ad_count),
        "targetAdCount": int(target_ad_count),
        "campaignId": cid,
        "createdAt": now,
        "lastHeartbeatAt": now,
        "builder": "builder1",
        "builder1ContractVersion": "builder1_production_v1",
    }
    if ownership_fields:
        entry.update(dict(ownership_fields))
    if _redis_configured():
        try:
            _get_redis().set(_job_key(jid), json.dumps(entry, ensure_ascii=False), ex=JOB_TTL_SECONDS)
        except Exception as exc:
            logger.error("BUILDER1_JOB_REDIS_CREATE_ERR jobId=%s err=%s", jid, exc)
            raise
    else:
        with _memory_lock:
            _memory_jobs[jid] = dict(entry)
    logger.info(
        "BUILDER1_JOB_CREATED jobId=%s campaignId=%s targetAdCount=%s backend=%s",
        jid,
        cid,
        target_ad_count,
        get_builder1_job_store_backend(),
    )
    return entry


def get_builder1_job(job_id: str) -> Optional[Dict[str, Any]]:
    jid = (job_id or "").strip()
    if not jid:
        return None
    if _redis_configured():
        try:
            raw = _get_redis().get(_job_key(jid))
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.error("BUILDER1_JOB_REDIS_LOAD_ERR jobId=%s err=%s", jid, exc)
            return None
    with _memory_lock:
        stored = _memory_jobs.get(jid)
        return dict(stored) if stored is not None else None


def update_builder1_job(job_id: str, **fields: Any) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _redis_configured():
        try:
            r = _get_redis()
            key = _job_key(jid)
            raw = r.get(key)
            if not raw:
                return
            entry = json.loads(raw)
            if not isinstance(entry, dict):
                return
            entry.update(fields)
            r.set(key, json.dumps(entry, ensure_ascii=False), ex=JOB_TTL_SECONDS)
            return
        except Exception as exc:
            logger.error("BUILDER1_JOB_REDIS_UPDATE_ERR jobId=%s err=%s", jid, exc)
            return
    with _memory_lock:
        if jid in _memory_jobs:
            _memory_jobs[jid].update(fields)


def touch_builder1_job_heartbeat(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    now = time.time()
    update_builder1_job(jid, lastHeartbeatAt=now)
    job = get_builder1_job(jid) or {}
    campaign_id = str(job.get("campaignId") or "").strip()
    if campaign_id:
        from engine.builder1_campaign_store import touch_campaign_lock_heartbeat

        touch_campaign_lock_heartbeat(campaign_id, job_id=jid, heartbeat_at=now)


def is_builder1_job_stale(job: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(job, dict):
        return True
    status = str(job.get("status") or "").strip()
    if status != "running":
        return False
    try:
        last = float(job.get("lastHeartbeatAt") or job.get("createdAt") or 0)
    except (TypeError, ValueError):
        return True
    return (time.time() - last) > BUILDER1_JOB_STALE_SECONDS


def try_finalize_stale_builder1_job(job_id: str) -> bool:
    """
    Mark a stale running job as abandoned. Returns True if transitioned.
    Does not mark done; releases campaign lock when owned by this job.
    """
    jid = (job_id or "").strip()
    job = get_builder1_job(jid)
    if not job or not is_builder1_job_stale(job):
        return False
    if str(job.get("status") or "").strip() != "running":
        return False
    campaign_id = str(job.get("campaignId") or "").strip()
    update_builder1_job(
        jid,
        status="error",
        error="stale_job_abandoned",
        retryable=False,
        staleAbandoned=True,
    )
    logger.info("BUILDER1_JOB_STALE_ABANDONED jobId=%s campaignId=%s", jid, campaign_id or None)
    return True


def finalize_builder1_job(job_id: str, result: dict[str, Any], *, target_ad_count: int) -> None:
    jid = (job_id or "").strip()
    existing = get_builder1_job(jid)
    if existing is None:
        return
    if str(existing.get("status") or "").strip() == "cancelled" or existing.get("cancelRequested"):
        logger.info("BUILDER1_JOB_FINALIZE_SKIPPED_CANCELLED jobId=%s", jid)
        return
    campaign_id = (existing.get("campaignId") or "").strip()
    result_campaign = (result.get("campaignId") or "").strip()
    if campaign_id and result_campaign and campaign_id != result_campaign:
        logger.error(
            "BUILDER1_JOB_CAMPAIGN_MISMATCH jobId=%s jobCampaignId=%s resultCampaignId=%s",
            jid,
            campaign_id,
            result_campaign,
        )
        update_builder1_job(
            jid,
            status="error",
            error="job_campaign_mismatch",
            result=result,
        )
        return

    if result.get("ok") is True:
        generated = int(result.get("generatedCount") or 0)
        update_builder1_job(
            jid,
            status="done",
            stage="done",
            completedAds=generated,
            totalAds=int(target_ad_count),
            targetAdCount=int(target_ad_count),
            result=result,
            lastPaidStageStatus="succeeded",
        )
        logger.info(
            "BUILDER1_JOB_DONE jobId=%s campaignId=%s generatedCount=%s targetAdCount=%s",
            jid,
            campaign_id,
            generated,
            target_ad_count,
        )
        return

    err = result.get("error") or "builder1_generation_failed"
    entry: Dict[str, Any] = {
        "status": "error",
        "error": err,
        "result": result,
        "totalAds": int(target_ad_count),
        "targetAdCount": int(target_ad_count),
    }
    if result.get("retryable"):
        entry["retryable"] = True
    update_builder1_job(jid, **entry)
    logger.error("BUILDER1_JOB_ERROR jobId=%s campaignId=%s err=%s", jid, campaign_id, err)


def clear_memory_jobs_for_tests() -> None:
    with _memory_lock:
        _memory_jobs.clear()
