"""
Redis-backed queue + job state for ACE async video generation (web enqueues, worker consumes).

Used by app.py and worker_video.py only — does not touch image engine.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

QUEUE_KEY = "ace:video:queue"
JOB_KEY_PREFIX = "ace:video:job:"
_JOB_TTL_SECONDS = 7 * 24 * 3600
# No heartbeat update for this long while status=running → poll finalizes as terminal error (SIGKILL / lost worker).
_STALE_RUNNING_SECONDS = int((os.environ.get("VIDEO_JOB_STALE_SECONDS") or "900").strip() or "900")

_redis = None
_redis_bound_url = ""


def redis_url() -> str:
    return (os.environ.get("REDIS_URL") or "").strip()


def redis_configured() -> bool:
    return bool(redis_url())


def reset_redis_client_for_tests() -> None:
    """Drop the process singleton so the next get_redis() uses current REDIS_URL."""
    global _redis, _redis_bound_url
    _redis = None
    _redis_bound_url = ""


def get_redis():
    """Singleton Redis client (decode_responses=True)."""
    global _redis, _redis_bound_url
    url = redis_url()
    if _redis is not None and _redis_bound_url == url:
        return _redis
    _redis = None
    if not url:
        raise RuntimeError("REDIS_URL is not set")
    from engine.ace_local_dev_config import assert_local_dev_redis_url_allowed

    assert_local_dev_redis_url_allowed(url)
    import redis as redis_lib

    # Avoid indefinite hangs if Redis is unreachable (worker would stop after VIDEO_JOB_STARTED).
    _t = float((os.environ.get("REDIS_SOCKET_TIMEOUT_SECONDS") or "60").strip() or "60")
    _redis = redis_lib.Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=_t,
        socket_timeout=_t,
    )
    _redis_bound_url = url
    return _redis


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def video_job_set_resolved_product_name(job_id: str, resolved: str, source: str) -> None:
    """Worker/video engine: canonical product name for this job (user or auto)."""
    jid = (job_id or "").strip()
    if not jid:
        return
    val = (resolved or "").strip()
    get_redis().hset(
        job_key(jid),
        mapping={
            "resolved_product_name": val,
            "product_name_source": (source or "").strip(),
        },
    )
    logger.info(
        "VIDEO_PRODUCT_NAME_RESOLVED_STORED value=%s",
        json.dumps(val, ensure_ascii=False),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_memory_job_hashes: Dict[str, Dict[str, str]] = {}
_memory_queue: list[str] = []
_use_memory_jobs = False
_after_record_persist: Optional[Callable[[str], None]] = None
_force_enqueue_fail = False


def enable_memory_jobs() -> None:
    global _use_memory_jobs, _memory_job_hashes, _memory_queue
    global _after_record_persist, _force_enqueue_fail
    _use_memory_jobs = True
    _memory_job_hashes = {}
    _memory_queue = []
    _after_record_persist = None
    _force_enqueue_fail = False


def disable_memory_jobs() -> None:
    global _use_memory_jobs, _memory_job_hashes, _memory_queue
    global _after_record_persist, _force_enqueue_fail
    _use_memory_jobs = False
    _memory_job_hashes = {}
    _memory_queue = []
    _after_record_persist = None
    _force_enqueue_fail = False


def set_after_video_job_record_persist_hook(hook: Optional[Callable[[str], None]]) -> None:
    """Test-only. Runs after the job hash exists and before bind/enqueue."""
    global _after_record_persist
    _after_record_persist = hook


def simulate_video_job_enqueue_fail(enabled: bool = True) -> None:
    global _force_enqueue_fail
    _force_enqueue_fail = bool(enabled)


def set_memory_job_hash(job_id: str, data: Dict[str, str]) -> None:
    _memory_job_hashes[job_id] = dict(data)


def list_known_video_job_ids() -> list[str]:
    if _use_memory_jobs:
        return list(_memory_job_hashes.keys())
    if not redis_configured():
        return []
    ids: list[str] = []
    try:
        for key in get_redis().scan_iter(match=f"{JOB_KEY_PREFIX}*", count=64):
            token = str(key)
            if token.startswith(JOB_KEY_PREFIX):
                ids.append(token[len(JOB_KEY_PREFIX) :])
    except Exception:
        return []
    return ids


def video_job_lpush(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_jobs:
        _memory_queue.insert(0, jid)
        return
    get_redis().lpush(QUEUE_KEY, jid)


def video_job_is_in_execution_queue(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid:
        return False
    if _use_memory_jobs:
        return jid in _memory_queue
    r = get_redis()
    try:
        return r.lpos(QUEUE_KEY, jid) is not None
    except Exception:
        return jid in list(r.lrange(QUEUE_KEY, 0, -1) or [])


def video_job_enqueue_if_absent(job_id: str) -> bool:
    """Publish jobId for workers once. Does not create the hash."""
    jid = (job_id or "").strip()
    if not jid:
        return False
    if video_job_is_in_execution_queue(jid):
        return False
    if _force_enqueue_fail:
        raise RuntimeError("video_job_enqueue_failed")
    video_job_lpush(jid)
    return True


def video_job_get_raw(job_id: str) -> Optional[Dict[str, str]]:
    """Return raw Redis hash for a job or None if missing."""
    if _use_memory_jobs:
        data = _memory_job_hashes.get(job_id)
        return dict(data) if data else None
    if not redis_configured():
        return None
    data = get_redis().hgetall(job_key(job_id))
    return data if data else None


def _persist_video_job_hash(job_id: str, mapping: Dict[str, str]) -> None:
    stored = {k: str(v) for k, v in mapping.items()}
    if _use_memory_jobs:
        _memory_job_hashes[job_id] = stored
        return
    r = get_redis()
    pipe = r.pipeline()
    pipe.hset(job_key(job_id), mapping=stored)
    pipe.expire(job_key(job_id), _JOB_TTL_SECONDS)
    pipe.execute()


def _new_video_job_mapping(
    *,
    product_name: str,
    product_description: str,
    public_base_url: str,
    extra_fields: Optional[Dict[str, str]],
    now: int,
    now_iso: str,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {
        "status": "queued",
        "product_name": product_name or "",
        "product_description": product_description or "",
        "public_base_url": public_base_url or "",
        "video_url": "",
        "marketing_text": "",
        "overlay_headline": "",
        "postprocess_ran": "0",
        "error": "",
        "last_progress_ts": str(now),
        "enqueued_ts": str(now),
        "progressStage": "queued",
        "progressStartedAt": now_iso,
        "canResume": "1",
    }
    if isinstance(extra_fields, dict):
        from engine.builder_client_binding import strip_client_binding_fields

        for field_key, field_value in strip_client_binding_fields(extra_fields).items():
            mapping[str(field_key)] = str(field_value)
    return mapping


def video_job_create(
    job_id: str,
    product_name: str,
    product_description: str,
    public_base_url: str,
    *,
    extra_fields: Optional[Dict[str, str]] = None,
    authorized_ace_root: Optional[Path] = None,
) -> None:
    """
    Persist the job hash, apply authorized LOCAL binding when given, then publish to the queue.
    Client extra_fields cannot supply binding. Publication is the queue push, not the hash write.
    """
    now = int(time.time())
    now_iso = _utc_now_iso()
    existing = video_job_get_raw(job_id)
    created = not isinstance(existing, dict) or not existing
    mapping = _new_video_job_mapping(
        product_name=product_name,
        product_description=product_description,
        public_base_url=public_base_url,
        extra_fields=extra_fields,
        now=now,
        now_iso=now_iso,
    )
    from engine.ace_local_dev_config import is_local_dev_enabled

    if authorized_ace_root is not None and not is_local_dev_enabled():
        authorized_ace_root = None
    if authorized_ace_root is None:
        if created:
            if _use_memory_jobs:
                _memory_job_hashes[job_id] = {k: str(v) for k, v in mapping.items()}
                _memory_queue.insert(0, job_id)
            else:
                r = get_redis()
                pipe = r.pipeline()
                pipe.hset(job_key(job_id), mapping=mapping)
                pipe.expire(job_key(job_id), _JOB_TTL_SECONDS)
                pipe.lpush(QUEUE_KEY, job_id)
                pipe.execute()
            logger.info("VIDEO_JOB_REDIS_ENQUEUE jobId=%s", job_id)
            logger.info("VIDEO_TIMING_STAGE_END stage=enqueue jobId=%s enqueued_ts=%s", job_id, now)
            return
        video_job_enqueue_if_absent(job_id)
        return

    if created:
        _persist_video_job_hash(job_id, mapping)
        hook = _after_record_persist
        if hook is not None:
            hook(job_id)
    from engine.ace_local_dev_builder2_binding import (
        bind_builder2_job_to_local_instance,
        job_bound_to_local_instance,
    )

    bind_builder2_job_to_local_instance(job_id, authorized_ace_root)
    raw = video_job_get_raw(job_id) or {}
    if not job_bound_to_local_instance(raw, authorized_ace_root):
        raise RuntimeError("local_binding_failed")
    if video_job_enqueue_if_absent(job_id):
        logger.info("VIDEO_JOB_REDIS_ENQUEUE jobId=%s", job_id)
        logger.info("VIDEO_TIMING_STAGE_END stage=enqueue jobId=%s enqueued_ts=%s", job_id, now)


def video_job_get(job_id: str) -> Optional[Dict[str, Any]]:
    """Return job dict for /api/video-status or None if missing/expired."""
    if _use_memory_jobs:
        data = _memory_job_hashes.get(job_id)
        if not data:
            return None
    else:
        r = get_redis()
        data = r.hgetall(job_key(job_id))
        if not data:
            return None
    rp = (data.get("resolved_product_name") or "").strip()
    st = (data.get("status") or "running").strip()
    infra = (data.get("infrastructure_failure") or "").strip() == "1"
    icode = (data.get("interrupt_code") or "").strip()
    return {
        "status": st,
        "videoUrl": data.get("video_url") or "",
        "marketingText": data.get("marketing_text") or "",
        "overlayHeadline": data.get("overlay_headline") or "",
        "publicBaseUrl": data.get("public_base_url") or "",
        "postprocessRan": (data.get("postprocess_ran") or "").strip(),
        "error": data.get("error") or "",
        "interruptCode": icode,
        "infrastructureFailure": infra,
        "resolvedProductName": rp,
        "productNameResolved": rp,
        "productNameSource": (data.get("product_name_source") or "").strip(),
        "productDescription": (data.get("product_description") or "").strip(),
        "progressStage": (data.get("progressStage") or data.get("progress_stage") or "").strip(),
        "progressStartedAt": (data.get("progressStartedAt") or "").strip(),
        "builder": (data.get("builder") or "").strip(),
        "builder2ResumeContractVersion": (data.get("builder2ResumeContractVersion") or "").strip(),
        "ownerContextPresent": (data.get("ownerContextPresent") or "").strip() in {"1", "true", "True"},
        "cancelRequested": str(data.get("cancelRequested") or "").strip().lower() in {"1", "true", "yes"},
        "cancelRequestedAt": (data.get("cancelRequestedAt") or "").strip(),
        "cancelReason": (data.get("cancelReason") or "").strip(),
        "cancelledAt": (data.get("cancelledAt") or "").strip(),
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


def video_job_remove_from_queue(job_id: str) -> int:
    """Remove all queue entries for job_id. Returns count removed (0 if none)."""
    jid = (job_id or "").strip()
    if not jid:
        return 0
    if _use_memory_jobs:
        removed = 0
        while jid in _memory_queue:
            _memory_queue.remove(jid)
            removed += 1
        from engine.builder2_tournament_recovery import _use_memory_recovery, _memory_queued, clear_job_queued

        if _use_memory_recovery and jid in _memory_queued:
            clear_job_queued(jid)
            removed = max(removed, 1)
        return removed
    return int(get_redis().lrem(QUEUE_KEY, 0, jid) or 0)


def video_job_mark_done(
    job_id: str,
    video_url: str,
    marketing_text: str,
    overlay_headline: str = "",
) -> None:
    t0 = time.monotonic()
    mapping = {
        "status": "done",
        "video_url": video_url or "",
        "marketing_text": marketing_text or "",
        "overlay_headline": overlay_headline or "",
        "postprocess_ran": "0",
        "error": "",
        "last_progress_ts": str(int(time.time())),
    }
    if _use_memory_jobs:
        existing = _memory_job_hashes.get(job_id) or {}
        existing.update({k: str(v) for k, v in mapping.items()})
        _memory_job_hashes[job_id] = existing
    else:
        r = get_redis()
        r.hset(job_key(job_id), mapping=mapping)
    logger.info(
        "VIDEO_TIMING_STAGE_END stage=redis_mark_done jobId=%s elapsed_ms=%.1f",
        job_id,
        (time.monotonic() - t0) * 1000.0,
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


def video_job_mark_interrupted(
    job_id: str,
    *,
    error_truth: str = "worker_shutdown_during_job",
    interrupt_code: str = "interrupted_worker_shutdown",
) -> None:
    """
    Worker infra shutdown (SIGTERM/SIGINT) while a job is active.
    Preserves Redis hash fields (product_name, product_description, etc.) for inspection or manual retry.
    """
    r = get_redis()
    r.hset(
        job_key(job_id),
        mapping={
            "status": "interrupted",
            "error": error_truth or "worker_shutdown_during_job",
            "interrupt_code": interrupt_code or "interrupted_worker_shutdown",
            "infrastructure_failure": "1",
            "last_progress_ts": str(int(time.time())),
        },
    )


def video_job_brpop(timeout_seconds: int = 30) -> Optional[str]:
    """Blocking pop of next job id from queue (worker loop)."""
    if _use_memory_jobs:
        if not _memory_queue:
            return None
        return _memory_queue.pop()
    r = get_redis()
    item = r.brpop(QUEUE_KEY, timeout=timeout_seconds)
    if not item:
        return None
    # (key, value)
    return item[1]
