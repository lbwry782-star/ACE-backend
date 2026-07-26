"""
Builder2 tournament worker recovery — leases, deduplicated requeue, recoverable registry.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.builder2_tournament_config import resolve_builder2_tournament_enabled
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state, tournament_key
from engine.video_jobs_redis import QUEUE_KEY, get_redis, job_key

logger = logging.getLogger(__name__)

RECOVERABLE_JOBS_KEY = "ace:builder2:recoverable_jobs"
RECOVERY_META_KEY_PREFIX = "ace:builder2:recovery_meta:"
QUEUED_KEY_PREFIX = "ace:builder2:queued:"
LEASE_KEY_PREFIX = "ace:builder2:lease:"

TERMINAL_JOB_STATUSES = frozenset({"done", "error", "failed", "cancelled", "recovery_exhausted"})
TERMINAL_TOURNAMENT_STATUSES = frozenset({"failed", "cancelled", "recovery_exhausted"})

def _recovery_max_attempts() -> int:
    raw = (os.environ.get("BUILDER2_RECOVERY_MAX_AUTOMATIC_ATTEMPTS") or "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _recovery_ttl_seconds() -> int:
    raw = (os.environ.get("BUILDER2_RECOVERY_TTL_SECONDS") or "604800").strip()
    try:
        return max(3600, int(raw))
    except ValueError:
        return 604800


def _recovery_meta_key(job_id: str) -> str:
    return f"{RECOVERY_META_KEY_PREFIX}{(job_id or '').strip()}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_recovery_meta(job_id: str) -> Dict[str, Any]:
    jid = (job_id or "").strip()
    if not jid:
        return {}
    if _use_memory_recovery:
        return dict(_memory_recovery_meta.get(jid) or {})
    raw = get_redis().get(_recovery_meta_key(jid))
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_recovery_meta(job_id: str, meta: Dict[str, Any]) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_recovery_meta[jid] = dict(meta)
        return
    get_redis().set(_recovery_meta_key(jid), json.dumps(meta), ex=_recovery_ttl_seconds() * 2)


def _clear_recovery_meta(job_id: str) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_recovery_meta.pop(jid, None)
    else:
        get_redis().delete(_recovery_meta_key(jid))


_memory_recoverable: set[str] = set()
_memory_queued: set[str] = set()
_memory_leases: Dict[str, Dict[str, Any]] = {}
_memory_recovery_meta: Dict[str, Dict[str, Any]] = {}
_use_memory_recovery = False


def enable_memory_recovery() -> None:
    global _use_memory_recovery, _memory_recoverable, _memory_queued, _memory_leases, _memory_recovery_meta
    _use_memory_recovery = True
    _memory_recoverable = set()
    _memory_queued = set()
    _memory_leases = {}
    _memory_recovery_meta = {}


def disable_memory_recovery() -> None:
    global _use_memory_recovery, _memory_recoverable, _memory_queued, _memory_leases, _memory_recovery_meta
    _use_memory_recovery = False
    _memory_recoverable = set()
    _memory_queued = set()
    _memory_leases = {}
    _memory_recovery_meta = {}


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
    meta = _migrate_legacy_recovery_meta(jid)
    if _use_memory_recovery:
        already_registered = jid in _memory_recoverable
    else:
        already_registered = bool(get_redis().sismember(RECOVERABLE_JOBS_KEY, jid))
    if already_registered:
        meta["recoveryInterruptCount"] = int(meta.get("recoveryInterruptCount") or 0) + 1
        meta["recoveryAttemptCount"] = max(
            int(meta.get("recoveryAttemptCount") or 0),
            int(meta.get("recoveryInterruptCount") or 0),
        )
        _save_recovery_meta(jid, meta)
    if not meta.get("recoveryFirstRegisteredAt"):
        meta["recoveryFirstRegisteredAt"] = _utc_now_iso()
    if "recoveryAttemptCount" not in meta:
        meta["recoveryAttemptCount"] = 0
    _save_recovery_meta(jid, meta)
    if _use_memory_recovery:
        _memory_recoverable.add(jid)
    else:
        get_redis().sadd(RECOVERABLE_JOBS_KEY, jid)
    logger.info("BUILDER2_TOURNAMENT_RECOVERY_REGISTERED jobId=%s", jid)


def remove_recoverable_job(job_id: str, *, clear_meta: bool = True) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    if _use_memory_recovery:
        _memory_recoverable.discard(jid)
    else:
        get_redis().srem(RECOVERABLE_JOBS_KEY, jid)
    if clear_meta:
        _clear_recovery_meta(jid)
    expire_job_lease(jid)


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


def _migrate_legacy_recovery_meta(job_id: str) -> Dict[str, Any]:
    jid = (job_id or "").strip()
    meta = _load_recovery_meta(jid)
    if meta.get("legacyMigrated"):
        return meta
    in_registry = False
    if _use_memory_recovery:
        in_registry = jid in _memory_recoverable
    else:
        in_registry = bool(get_redis().sismember(RECOVERABLE_JOBS_KEY, jid))
    if not meta and in_registry:
        meta = {
            "recoveryFirstRegisteredAt": _utc_now_iso(),
            "recoveryAttemptCount": _recovery_max_attempts(),
            "legacyMigrated": True,
            "legacyMigrationReason": "missing_recovery_meta",
        }
        _save_recovery_meta(jid, meta)
        logger.info(
            "BUILDER2_TOURNAMENT_RECOVERY_LEGACY_MIGRATED jobId=%s attempts=%s reason=missing_recovery_meta",
            jid,
            meta["recoveryAttemptCount"],
        )
        _mark_recovery_exhausted(jid, reason="recovery_exhausted")
        return meta
    if meta and "recoveryAttemptCount" not in meta:
        meta["recoveryAttemptCount"] = max(1, int(meta.get("recoveryInterruptCount") or 0))
        meta["legacyMigrated"] = True
        meta["legacyMigrationReason"] = "missing_attempt_count"
        _save_recovery_meta(jid, meta)
        logger.info(
            "BUILDER2_TOURNAMENT_RECOVERY_LEGACY_MIGRATED jobId=%s attempts=%s reason=missing_attempt_count",
            jid,
            meta["recoveryAttemptCount"],
        )
    return meta


def _recovery_is_stale(job_id: str, meta: Dict[str, Any]) -> bool:
    first = meta.get("recoveryFirstRegisteredAt") or meta.get("lastRecoveryAttemptAt")
    if not first:
        return False
    try:
        first_dt = datetime.fromisoformat(str(first).replace("Z", "+00:00"))
    except ValueError:
        return False
    age = datetime.now(timezone.utc) - first_dt.astimezone(timezone.utc)
    return age.total_seconds() > _recovery_ttl_seconds()


def _mark_recovery_exhausted(job_id: str, *, reason: str) -> None:
    jid = (job_id or "").strip()
    meta = _load_recovery_meta(jid)
    meta["recoveryTerminalReason"] = reason
    _save_recovery_meta(jid, meta)
    remove_recoverable_job(jid, clear_meta=False)
    state = load_tournament_state(jid)
    if state is not None:
        state["status"] = "recovery_exhausted"
        state["error"] = reason
        save_tournament_state(jid, state)
    data = _read_job_hash(jid)
    if data:
        data["status"] = "recovery_exhausted"
        if _use_memory_recovery:
            _memory_job_hashes[jid] = data
        else:
            get_redis().hset(job_key(jid), mapping=data)


def _job_is_recoverable(job_id: str) -> bool:
    jid = (job_id or "").strip()
    if not jid or not resolve_builder2_tournament_enabled():
        return False
    if _use_memory_recovery:
        if jid not in _memory_recoverable:
            return False
    elif not get_redis().sismember(RECOVERABLE_JOBS_KEY, jid):
        return False
    meta = _migrate_legacy_recovery_meta(jid)
    if _recovery_is_stale(jid, meta):
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_STALE_REMOVED jobId=%s", jid)
        _mark_recovery_exhausted(jid, reason="recovery_stale_ttl")
        return False
    if meta.get("recoveryTerminalReason"):
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED_TERMINAL jobId=%s reason=%s", jid, meta["recoveryTerminalReason"])
        return False
    if not load_tournament_state(jid):
        return False
    data = _read_job_hash(jid)
    if not data:
        return False
    status = (data.get("status") or "").strip()
    if status in TERMINAL_JOB_STATUSES:
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED_TERMINAL jobId=%s status=%s", jid, status)
        return False
    state = load_tournament_state(jid) or {}
    if state.get("status") in TERMINAL_TOURNAMENT_STATUSES:
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_SKIPPED_TERMINAL jobId=%s tournamentStatus=%s", jid, state.get("status"))
        return False
    if state.get("lastCompletedStep") in {"winner_plan_complete", "runway_complete", "done"}:
        return False
    attempts = int(meta.get("recoveryAttemptCount") or 0)
    if attempts >= _recovery_max_attempts():
        logger.info("BUILDER2_TOURNAMENT_RECOVERY_EXHAUSTED jobId=%s attempts=%s", jid, attempts)
        _mark_recovery_exhausted(jid, reason="recovery_exhausted")
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
    from engine.builder2_creator_preflight import creator_preflight_only_enabled

    if creator_preflight_only_enabled():
        logger.info("BUILDER2_CREATOR_PREFLIGHT_SKIP_REQUEUE jobId=%s", (job_id or "").strip())
        return False
    jid = (job_id or "").strip()
    _migrate_legacy_recovery_meta(jid)
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
    meta = _load_recovery_meta(jid)
    meta["recoveryAttemptCount"] = int(meta.get("recoveryAttemptCount") or 0) + 1
    meta["lastRecoveryAttemptAt"] = _utc_now_iso()
    _save_recovery_meta(jid, meta)
    if _use_memory_recovery:
        pass
    else:
        get_redis().lpush(QUEUE_KEY, jid)
    logger.info("BUILDER2_TOURNAMENT_RECOVERY_REQUEUED jobId=%s", jid)
    return True


def scan_and_requeue_recoverable_jobs() -> List[str]:
    from engine.builder2_creator_preflight import creator_preflight_only_enabled

    if creator_preflight_only_enabled():
        logger.info("BUILDER2_CREATOR_PREFLIGHT_SKIP_RECOVERY reason=preflight_mode")
        return []
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
