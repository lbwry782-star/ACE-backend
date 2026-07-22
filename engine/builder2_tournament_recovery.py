"""
Builder2 tournament worker recovery — leases, deduplicated requeue, recoverable registry.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from engine.builder2_tournament_config import resolve_builder2_tournament_enabled
from engine.builder2_tournament_store import load_tournament_state, tournament_key
from engine.video_jobs_redis import QUEUE_KEY, get_redis, job_key

logger = logging.getLogger(__name__)

RECOVERABLE_JOBS_KEY = "ace:builder2:recoverable_jobs"
QUEUED_KEY_PREFIX = "ace:builder2:queued:"
LEASE_KEY_PREFIX = "ace:builder2:lease:"

_memory_recoverable: set[str] = set()
_memory_queued: set[str] = set()
_memory_leases: Dict[str, Dict[str, Any]] = {}
_use_memory_recovery = False


def enable_memory_recovery() -> None:
    global _use_memory_recovery, _memory_recoverable, _memory_queued, _memory_leases
    _use_memory_recovery = True
    _memory_recoverable = set()
    _memory_queued = set()
    _memory_leases = {}


def disable_memory_recovery() -> None:
    global _use_memory_recovery, _memory_recoverable, _memory_queued, _memory_leases
    _use_memory_recovery = False
    _memory_recoverable = set()
    _memory_queued = set()
    _memory_leases = {}


def _lease_seconds() -> int:
    raw = (os.environ.get("BUILDER2_TOURNAMENT_LEASE_SECONDS") or "").strip()
    if raw:
        try:
            return max(30, int(raw))
        except ValueError:
            pass
    stale = (os.environ.get("VIDEO_JOB_STALE_SECONDS") or "900").strip() or "900"
    try:
        return max(30, int(stale))
    except ValueError:
        return 900


def _queued_key(job_id: str) -> str:
    return f"{QUEUED_KEY_PREFIX}{(job_id or '').strip()}"


def _lease_key(job_id: str) -> str:
    return f"{LEASE_KEY_PREFIX}{(job_id or '').strip()}"


def new_worker_token() -> str:
    return f"worker-{uuid.uuid4().hex}"


def register_recoverable_job(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_recoverable.add(jid)
    else:
        get_redis().sadd(RECOVERABLE_JOBS_KEY, jid)
    logger.info("BUILDER2_TOURNAMENT_RECOVERY_REGISTERED jobId=%s", jid)


def remove_recoverable_job(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_recoverable.discard(jid)
    else:
        get_redis().srem(RECOVERABLE_JOBS_KEY, jid)


def is_job_queued(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid:
        return False
    if _use_memory_recovery:
        return jid in _memory_queued
    return bool(get_redis().exists(_queued_key(jid)))


def mark_job_queued(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid:
        return False
    if _use_memory_recovery:
        if jid in _memory_queued:
            return False
        _memory_queued.add(jid)
        return True
    return bool(get_redis().set(_queued_key(jid), "1", nx=True, ex=_lease_seconds()))


def clear_job_queued(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_queued.discard(jid)
    else:
        get_redis().delete(_queued_key(jid))


def acquire_job_lease(job_id: str, worker_token: str) -> bool:
    jid = (job_id or "").strip()
    token = (worker_token or "").strip()
    if not jid or not token:
        return False
    payload = json.dumps({"owner": token, "acquiredAt": int(time.time())})
    if _use_memory_recovery:
        current = _memory_leases.get(jid)
        now = int(time.time())
        if current and int(current.get("expiresAt") or 0) > now:
            return current.get("owner") == token
        _memory_leases[jid] = {"owner": token, "expiresAt": now + _lease_seconds()}
        logger.info("BUILDER2_TOURNAMENT_LEASE_ACQUIRED jobId=%s", jid)
        return True
    ok = bool(
        get_redis().set(_lease_key(jid), payload, nx=True, ex=_lease_seconds())
    )
    if ok:
        logger.info("BUILDER2_TOURNAMENT_LEASE_ACQUIRED jobId=%s", jid)
    return ok


def release_job_lease(job_id: str, worker_token: str) -> None:
    jid = (job_id or "").strip()
    token = (worker_token or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        current = _memory_leases.get(jid)
        if current and current.get("owner") == token:
            _memory_leases.pop(jid, None)
            logger.info("BUILDER2_TOURNAMENT_LEASE_RELEASED jobId=%s", jid)
        return
    key = _lease_key(jid)
    r = get_redis()
    raw = r.get(key)
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if data.get("owner") == token:
        r.delete(key)
        logger.info("BUILDER2_TOURNAMENT_LEASE_RELEASED jobId=%s", jid)


def has_active_lease(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid:
        return False
    if _use_memory_recovery:
        current = _memory_leases.get(jid)
        return bool(current and int(current.get("expiresAt") or 0) > int(time.time()))
    return bool(get_redis().exists(_lease_key(jid)))


def expire_job_lease(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_leases.pop(jid, None)
    else:
        get_redis().delete(_lease_key(jid))


def _job_is_recoverable(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid or not resolve_builder2_tournament_enabled():
        return False
    if _use_memory_recovery:
        if jid not in _memory_recoverable:
            return False
    elif not get_redis().sismember(RECOVERABLE_JOBS_KEY, jid):
        return False
    if not load_tournament_state(jid):
        return False
    data = _read_job_hash(jid)
    if not data:
        return False
    status = (data.get("status") or "").strip()
    if status in {"done", "error"}:
        return False
    if status == "error" and (data.get("error") or "").strip() not in {"", "worker_shutdown_during_job"}:
        return False
    state = load_tournament_state(jid) or {}
    if state.get("status") == "failed":
        return False
    if state.get("lastCompletedStep") in {"winner_plan_complete", "runway_complete", "done"}:
        return False
    return True


def _read_job_hash(job_id: str) -> Dict[str, Any]:
    if _use_memory_recovery:
        return _memory_job_hashes.get(job_id, {})
    return get_redis().hgetall(job_key(job_id)) or {}


_memory_job_hashes: Dict[str, Dict[str, Any]] = {}


def set_memory_job_hash(job_id: str, data: Dict[str, Any]) -> None:
    _memory_job_hashes[job_id] = dict(data)


def requeue_recoverable_job(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not _job_is_recoverable(jid):
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED jobId=%s reason=not_recoverable", jid)
        return False
    if has_active_lease(jid):
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED jobId=%s reason=active_lease", jid)
        return False
    if is_job_queued(jid):
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED jobId=%s reason=already_queued", jid)
        return False
    if not mark_job_queued(jid):
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED jobId=%s reason=queue_dedupe", jid)
        return False
    if _use_memory_recovery:
        pass
    else:
        get_redis().lpush(QUEUE_KEY, jid)
    logger.info("BUILDER2_TOURNAMENT_RECOVERY_REQUEUED jobId=%s", jid)
    return True


def scan_and_requeue_recoverable_jobs() -> List[str]:
    if not resolve_builder2_tournament_enabled():
        return []
    if _use_memory_recovery:
        candidates = list(_memory_recoverable)
    else:
        candidates = list(get_redis().smembers(RECOVERABLE_JOBS_KEY) or [])
    requeued: List[str] = []
    for jid in candidates:
        if requeue_recoverable_job(str(jid)):
            requeued.append(str(jid))
    return requeued


def tournament_exists(job_id: str) -> bool:
    if _use_memory_recovery:
        from engine.builder2_tournament_store import _memory_states

        return job_id in _memory_states
    return bool(get_redis().exists(tournament_key(job_id)))
