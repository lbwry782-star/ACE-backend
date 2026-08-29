"""
Resolve Builder2 productDescription for packaging — state first, Redis job fallback.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_builder2_product_description_for_packaging(
    *,
    job_id: str = "",
    state: Optional[Dict[str, Any]] = None,
    explicit: str = "",
) -> str:
    """
    Return authoritative product description for packaging marketing copy.

    Tournament state may omit productDescription; the Redis video job hash retains it.
    """
    token = str(explicit or "").strip()
    if token:
        return token

    bucket = state if isinstance(state, dict) else {}
    for key in ("productDescription", "product_description"):
        token = str(bucket.get(key) or "").strip()
        if token:
            return token

    jid = str(job_id or bucket.get("jobId") or "").strip()
    if not jid:
        return ""

    try:
        from engine.video_jobs_redis import redis_configured, video_job_get, video_job_get_raw

        raw = video_job_get_raw(jid)
        if isinstance(raw, dict):
            for key in ("product_description", "productDescription"):
                token = str(raw.get(key) or "").strip()
                if token:
                    return token
        if not redis_configured():
            return ""
        job = video_job_get(jid)
        if isinstance(job, dict):
            for key in ("productDescription", "product_description"):
                token = str(job.get(key) or "").strip()
                if token:
                    return token
    except Exception:
        return ""
    return ""
