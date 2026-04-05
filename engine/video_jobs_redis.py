"""
Redis-backed queue + job state for ACE async video generation (web enqueues, worker consumes).

Used by app.py and worker_video.py only — does not touch image engine.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

QUEUE_KEY = "ace:video:queue"
JOB_KEY_PREFIX = "ace:video:job:"
_JOB_TTL_SECONDS = 7 * 24 * 3600

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

    _redis = redis_lib.Redis.from_url(url, decode_responses=True)
    return _redis


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def video_job_create(
    job_id: str,
    product_name: str,
    product_description: str,
    public_base_url: str,
) -> None:
    """Persist job hash and push job_id onto the queue."""
    r = get_redis()
    key = job_key(job_id)
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
            "error": "",
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
        "error": data.get("error") or "",
    }


def video_job_mark_done(job_id: str, video_url: str, marketing_text: str) -> None:
    r = get_redis()
    r.hset(
        job_key(job_id),
        mapping={
            "status": "done",
            "video_url": video_url or "",
            "marketing_text": marketing_text or "",
            "error": "",
        },
    )


def video_job_mark_error(job_id: str, error_code: str = "video_generation_failed") -> None:
    r = get_redis()
    r.hset(
        job_key(job_id),
        mapping={
            "status": "error",
            "error": error_code or "video_generation_failed",
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
