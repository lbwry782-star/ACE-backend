"""
Builder2 initial /api/generate-video idempotency — owner-scoped, Redis-atomic.

Prevents duplicate initial job + allowance + queue entry for the same
X-ACE-Request-Id + owner + request fingerprint.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-ACE-Request-Id"
_IDEMPOTENCY_KEY_PREFIX = "ace:builder2:initial-generate-idempotency:"
_IDEMPOTENCY_TTL_SECONDS = 7 * 24 * 3600
_CREATION_CLAIM_LEASE_SECONDS = 120

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

STATE_RESERVED = "reserved"
STATE_CREATING = "creating"
STATE_ENQUEUED = "enqueued"

BeginKind = Literal["new", "replay", "conflict", "recover"]

_memory_lock = threading.Lock()
_memory_idempotency: Dict[str, Dict[str, Any]] = {}
_use_memory_store = False

_RESERVE_OR_GET_LUA = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local payload = ARGV[2]
local existing = redis.call('GET', key)
if existing then
  return existing
end
local ok = redis.call('SET', key, payload, 'NX', 'EX', ttl)
if ok then
  return payload
end
return redis.call('GET', key)
"""

_CLAIM_ENQUEUE_LUA = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local lease_seconds = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local claim_token = ARGV[4]
local raw = redis.call('GET', key)
if not raw then
  return cjson.encode({ok=false, reason='missing'})
end
local data = cjson.decode(raw)
if data.state == 'enqueued' then
  return cjson.encode({ok=true, replay=true, record=raw})
end
if data.creationClaimedAt then
  local age = now - tonumber(data.creationClaimedAt)
  if age < lease_seconds then
    return cjson.encode({ok=true, replay=true, in_progress=true, record=raw})
  end
end
data.creationClaimedAt = now
data.creationClaimToken = claim_token
data.state = 'creating'
local encoded = cjson.encode(data)
redis.call('SET', key, encoded, 'EX', ttl)
return cjson.encode({ok=true, claimed=true, record=encoded})
"""


def _redis_configured() -> bool:
    if _use_memory_store:
        return False
    return bool((os.environ.get("REDIS_URL") or "").strip())


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def extract_request_id(request: Any) -> str:
    headers = getattr(request, "headers", {}) or {}
    return _clean(headers.get(REQUEST_ID_HEADER) or headers.get("X-Ace-Request-Id"))


def validate_request_id_format(request_id: str) -> bool:
    token = _clean(request_id)
    return bool(token and _UUID_RE.match(token))


def enable_memory_idempotency_store() -> None:
    global _use_memory_store, _memory_idempotency
    _use_memory_store = True
    _memory_idempotency = {}


def disable_memory_idempotency_store() -> None:
    global _use_memory_store, _memory_idempotency
    _use_memory_store = False
    _memory_idempotency = {}


def clear_memory_idempotency_for_tests() -> None:
    with _memory_lock:
        _memory_idempotency.clear()


def idempotency_ttl_seconds() -> int:
    return _IDEMPOTENCY_TTL_SECONDS


def _idempotency_key(*, owner_context_ref: str, request_id: str) -> str:
    owner = _clean(owner_context_ref)
    rid = _clean(request_id)
    return f"{_IDEMPOTENCY_KEY_PREFIX}{owner}:{rid}"


def fingerprint_initial_generate_video(payload: Mapping[str, Any], *, target_video_count: int) -> str:
    from engine.builder2_job_ownership import _hash_token

    canonical = {
        "productName": _clean(payload.get("productName")),
        "productDescription": _clean(payload.get("productDescription")),
        "targetVideoCount": int(target_video_count),
    }
    return _hash_token(json.dumps(canonical, sort_keys=True, ensure_ascii=False))


def _load_record(key: str) -> Optional[Dict[str, Any]]:
    if _redis_configured():
        try:
            raw = _get_redis().get(key)
            if not raw:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.error("BUILDER2_IDEMPOTENCY_LOAD_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder2_idempotency_unavailable") from exc
    with _memory_lock:
        rec = _memory_idempotency.get(key)
        return dict(rec) if rec else None


def _save_record(key: str, record: Dict[str, Any]) -> None:
    payload = json.dumps(record, ensure_ascii=False)
    if _redis_configured():
        try:
            _get_redis().set(key, payload, ex=_IDEMPOTENCY_TTL_SECONDS)
            return
        except Exception as exc:
            logger.error("BUILDER2_IDEMPOTENCY_SAVE_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder2_idempotency_unavailable") from exc
    with _memory_lock:
        _memory_idempotency[key] = dict(record)


def _reserve_or_get(key: str, new_record: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps(new_record, ensure_ascii=False)
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_RESERVE_OR_GET_LUA)
            raw = script(keys=[key], args=[str(_IDEMPOTENCY_TTL_SECONDS), payload])
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else new_record
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("BUILDER2_IDEMPOTENCY_RESERVE_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder2_idempotency_unavailable") from exc
    with _memory_lock:
        existing = _memory_idempotency.get(key)
        if existing:
            return dict(existing)
        _memory_idempotency[key] = dict(new_record)
        return dict(new_record)


def _creation_claim_is_live(record: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False
    if _clean(record.get("state")) == STATE_ENQUEUED:
        return True
    claimed_at = record.get("creationClaimedAt")
    if claimed_at is None:
        return False
    try:
        age = time.time() - float(claimed_at)
    except (TypeError, ValueError):
        return False
    return age < _CREATION_CLAIM_LEASE_SECONDS


@dataclass(frozen=True)
class InitialGenerateIdempotencyBeginResult:
    kind: BeginKind
    record: Dict[str, Any]
    key: str


def begin_initial_generate_idempotency(
    *,
    request_id: str,
    owner_context_ref: str,
    request_fingerprint: str,
    job_id: str,
    video_allowance_id: str,
    target_video_count: int,
) -> InitialGenerateIdempotencyBeginResult:
    key = _idempotency_key(owner_context_ref=owner_context_ref, request_id=request_id)
    now = time.time()
    reserved = {
        "requestId": _clean(request_id),
        "ownerContextRef": _clean(owner_context_ref),
        "requestFingerprint": _clean(request_fingerprint),
        "state": STATE_RESERVED,
        "jobId": _clean(job_id),
        "videoAllowanceId": _clean(video_allowance_id),
        "targetVideoCount": int(target_video_count),
        "videoIndex": 1,
        "createdAt": now,
        "response": {},
    }
    stored = _reserve_or_get(key, reserved)
    if stored.get("requestFingerprint") != request_fingerprint:
        return InitialGenerateIdempotencyBeginResult(kind="conflict", record=stored, key=key)

    if _clean(stored.get("state")) == STATE_ENQUEUED:
        return InitialGenerateIdempotencyBeginResult(kind="replay", record=stored, key=key)
    if _creation_claim_is_live(stored):
        return InitialGenerateIdempotencyBeginResult(kind="replay", record=stored, key=key)

    authoritative_job = _clean(stored.get("jobId") or job_id)
    if authoritative_job != _clean(job_id):
        return InitialGenerateIdempotencyBeginResult(kind="recover", record=stored, key=key)
    return InitialGenerateIdempotencyBeginResult(kind="new", record=stored, key=key)


def _claim_enqueue_memory(key: str, *, claim_token: str) -> Tuple[bool, bool, Dict[str, Any]]:
    with _memory_lock:
        raw = _memory_idempotency.get(key)
        if not raw:
            return False, False, {}
        data = dict(raw)
        if _clean(data.get("state")) == STATE_ENQUEUED:
            return False, True, data
        if data.get("creationClaimedAt"):
            try:
                age = time.time() - float(data["creationClaimedAt"])
            except (TypeError, ValueError):
                age = _CREATION_CLAIM_LEASE_SECONDS
            if age < _CREATION_CLAIM_LEASE_SECONDS:
                return False, True, data
        now = time.time()
        data["creationClaimedAt"] = now
        data["creationClaimToken"] = claim_token
        data["state"] = STATE_CREATING
        _memory_idempotency[key] = data
        return True, False, data


def claim_initial_generate_enqueue(key: str) -> Tuple[bool, bool, Dict[str, Any]]:
    """
    Claim exclusive right to enqueue the initial job for this idempotency key.
    Returns (claimed, replay, record).
    """
    token = uuid.uuid4().hex
    now = time.time()
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_CLAIM_ENQUEUE_LUA)
            raw = script(
                keys=[key],
                args=[str(_IDEMPOTENCY_TTL_SECONDS), str(_CREATION_CLAIM_LEASE_SECONDS), str(now), token],
            )
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            parsed = json.loads(raw)
            if not parsed.get("ok"):
                return False, False, _load_record(key) or {}
            record = json.loads(parsed.get("record") or "{}")
            if parsed.get("replay"):
                return False, True, record if isinstance(record, dict) else {}
            return bool(parsed.get("claimed")), False, record if isinstance(record, dict) else {}
        except Exception as exc:
            logger.error("BUILDER2_IDEMPOTENCY_CLAIM_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder2_idempotency_unavailable") from exc
    return _claim_enqueue_memory(key, claim_token=token)


def mark_initial_generate_enqueued(key: str, *, response: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(_load_record(key) or {})
    current["state"] = STATE_ENQUEUED
    current["enqueuedAt"] = time.time()
    current["response"] = dict(response)
    _save_record(key, current)
    logger.info(
        "BUILDER2_INITIAL_GENERATE_ENQUEUED jobId=%s videoAllowanceId=%s requestId=%s",
        current.get("jobId"),
        current.get("videoAllowanceId"),
        current.get("requestId"),
    )
    return current


def replay_response_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    response = dict(record.get("response") or {})
    if not response.get("jobId"):
        response["jobId"] = record.get("jobId")
    if not response.get("videoAllowanceId") and record.get("videoAllowanceId"):
        response["videoAllowanceId"] = record.get("videoAllowanceId")
    if record.get("targetVideoCount") is not None and response.get("targetVideoCount") is None:
        response["targetVideoCount"] = record.get("targetVideoCount")
    if record.get("videoIndex") is not None and response.get("videoIndex") is None:
        response["videoIndex"] = record.get("videoIndex")
    response["ok"] = response.get("ok", True)
    response["status"] = response.get("status") or "queued"
    response["idempotentReplay"] = True
    return response


def build_in_progress_response(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "queued",
        "jobId": _clean(record.get("jobId")),
        "videoAllowanceId": _clean(record.get("videoAllowanceId")) or None,
        "targetVideoCount": int(record.get("targetVideoCount") or 1),
        "videoIndex": int(record.get("videoIndex") or 1),
        "idempotentReplay": True,
        "creationInProgress": True,
    }
