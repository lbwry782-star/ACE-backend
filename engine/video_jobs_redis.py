"""
Redis-backed queue + job state for ACE async video generation (web enqueues, worker consumes).

Used by app.py and worker_video.py only — does not touch image engine.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

QUEUE_KEY = "ace:video:queue"
JOB_KEY_PREFIX = "ace:video:job:"
_JOB_TTL_SECONDS = 7 * 24 * 3600
# No heartbeat update for this long while status=running → poll finalizes as terminal error (SIGKILL / lost worker).
_STALE_RUNNING_SECONDS = int((os.environ.get("VIDEO_JOB_STALE_SECONDS") or "900").strip() or "900")

_redis = None


def redis_url() -> str:
    return (os.environ.get("REDIS_URL") or "").strip()


def redis_configured() -> bool:
    return bool(redis_url())


def get_redis():
    """Singleton Redis client (decode_responses=True)."""
    global _redis
    if _redis is not None:
        return _redis
    url = redis_url()
    if not url:
        raise RuntimeError("REDIS_URL is not set")
    import redis as redis_lib

    # Avoid indefinite hangs if Redis is unreachable (worker would stop after VIDEO_JOB_STARTED).
    _t = float((os.environ.get("REDIS_SOCKET_TIMEOUT_SECONDS") or "60").strip() or "60")
    _redis = redis_lib.Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=_t,
        socket_timeout=_t,
    )
    return _redis


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def video_job_set_resolved_product_name(job_id: str, resolved: str, source: str) -> None:
    """Worker/video engine: canonical product name for this job (user or auto)."""
    jid = (job_id or "").strip()
    if not jid:
        return
    get_redis().hset(
        job_key(jid),
        mapping={
            "resolved_product_name": (resolved or "").strip(),
            "product_name_source": (source or "").strip(),
        },
    )


def video_job_create(
    job_id: str,
    product_name: str,
    product_description: str,
    public_base_url: str,
) -> None:
    """Persist job hash and push job_id onto the queue."""
    r = get_redis()
    key = job_key(job_id)
    now = int(time.time())
    pipe = r.pipeline()
    pipe.hset(
        key,
        mapping={
            "status": "running",
            "product_name": product_name or "",
            "product_description": product_description or "",
            "public_base_url": public_base_url or "",
            "video_url": "",
            "marketing_text": "",
            "overlay_headline": "",
            "postprocess_ran": "0",
            "error": "",
            "last_progress_ts": str(now),
        },
    )
    pipe.expire(key, _JOB_TTL_SECONDS)
    pipe.lpush(QUEUE_KEY, job_id)
    pipe.execute()
    logger.info("VIDEO_JOB_REDIS_ENQUEUE jobId=%s", job_id)


def video_job_get(job_id: str) -> Optional[Dict[str, Any]]:
    """Return job dict for /api/video-status or None if missing/expired."""
    r = get_redis()
    data = r.hgetall(job_key(job_id))
    if not data:
        return None
    return {
        "status": (data.get("status") or "running").strip(),
        "videoUrl": data.get("video_url") or "",
        "marketingText": data.get("marketing_text") or "",
        "overlayHeadline": data.get("overlay_headline") or "",
        "publicBaseUrl": data.get("public_base_url") or "",
        "postprocessRan": (data.get("postprocess_ran") or "").strip(),
        "error": data.get("error") or "",
        "resolvedProductName": (data.get("resolved_product_name") or "").strip(),
        "productNameSource": (data.get("product_name_source") or "").strip(),
    }


def video_job_touch_progress(job_id: str) -> None:
    """Worker heartbeat: refresh last_progress_ts while job is in progress."""
    get_redis().hset(job_key(job_id), "last_progress_ts", str(int(time.time())))


def video_job_try_finalize_stale_running(job_id: str) -> bool:
    """
    If job is still running and last_progress_ts is older than VIDEO_JOB_STALE_SECONDS, set terminal error.
    Returns True if this call transitioned the job to error (caller should re-read the job).
    """
    r = get_redis()
    key = job_key(job_id)
    data = r.hgetall(key)
    if not data:
        return False
    if (data.get("status") or "").strip() != "running":
        return False
    raw_ts = (data.get("last_progress_ts") or "").strip()
    now = int(time.time())
    if not raw_ts:
        # Legacy hashes without heartbeat field: start grace window from first observation.
        r.hset(key, "last_progress_ts", str(now))
        logger.info("VIDEO_JOB_PROGRESS_BOOTSTRAP jobId=%s", job_id)
        return False
    try:
        last = int(raw_ts)
    except ValueError:
        last = 0
    age = now - last
    if age <= _STALE_RUNNING_SECONDS:
        return False
    r.hset(
        key,
        mapping={
            "status": "error",
            "error": "stale_job_no_worker_progress",
        },
    )
    logger.info(
        "VIDEO_JOB_STALE_DETECTED jobId=%s age_s=%s threshold_s=%s",
        job_id,
        age,
        _STALE_RUNNING_SECONDS,
    )
    return True


def video_job_mark_done(
    job_id: str,
    video_url: str,
    marketing_text: str,
    overlay_headline: str = "",
) -> None:
    r = get_redis()
    r.hset(
        job_key(job_id),
        mapping={
            "status": "done",
            "video_url": video_url or "",
            "marketing_text": marketing_text or "",
            "overlay_headline": overlay_headline or "",
            "postprocess_ran": "0",
            "error": "",
            "last_progress_ts": str(int(time.time())),
        },
    )


def video_job_set_postprocess_result(job_id: str, final_video_url: str) -> None:
    """Web service: after local ffmpeg postprocess, store final URL and mark postprocess complete."""
    r = get_redis()
    r.hset(
        job_key(job_id),
        mapping={
            "video_url": final_video_url or "",
            "postprocess_ran": "1",
        },
    )


def video_job_set_postprocess_ran_only(job_id: str) -> None:
    """Mark postprocess as finished without changing video_url (edge cases)."""
    get_redis().hset(job_key(job_id), "postprocess_ran", "1")


def video_job_mark_error(job_id: str, error_code: str = "video_generation_failed") -> None:
    r = get_redis()
    r.hset(
        job_key(job_id),
        mapping={
            "status": "error",
            "error": error_code or "video_generation_failed",
            "last_progress_ts": str(int(time.time())),
        },
    )


def video_job_brpop(timeout_seconds: int = 30) -> Optional[str]:
    """Blocking pop of next job id from queue (worker loop)."""
    r = get_redis()
    item = r.brpop(QUEUE_KEY, timeout=timeout_seconds)
    if not item:
        return None
    # (key, value)
    return item[1]
