"""
Builder2 video purchase allowance — Redis-backed durable state.

One allowance authorizes 1 or 2 independent normal Builder2 productions for the
same immutable product snapshot. Not a campaign; Builder2-specific only.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine.video_jobs_redis import get_redis

logger = logging.getLogger(__name__)

ALLOWANCE_KEY_PREFIX = "ace:builder2:video-allowance:"
ALLOWANCE_TTL_SECONDS = 7 * 24 * 3600
_MAX_LOCK_RETRIES = 8

_memory_lock = threading.Lock()
_memory_allowances: Dict[str, Dict[str, Any]] = {}
_use_memory_store = False

_LUA_IS_NULL = """
local function is_null(v)
  return v == nil or v == cjson.null
end
"""

_RESERVE_VIDEO_TWO_LUA = (
    _LUA_IS_NULL
    + """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='allowance_not_found'})
end
local data = cjson.decode(raw)
local owner_ref = ARGV[1] or ''
local job_id = ARGV[2] or ''
local ttl = tonumber(ARGV[3])
local created_at = ARGV[4] or ''

if owner_ref == '' or data.ownerContextRef ~= owner_ref then
  return cjson.encode({ok=false, code='ownership_mismatch'})
end
if tonumber(data.targetVideoCount) ~= 2 then
  return cjson.encode({ok=false, code='target_video_count_not_two'})
end

local videos = data.videos or {}
local has_video_one = false
local video_two_job = nil
for _, entry in ipairs(videos) do
  local idx = tonumber(entry.videoIndex)
  if idx == 1 then
    has_video_one = true
  end
  if idx == 2 then
    video_two_job = entry.jobId
  end
end

if not has_video_one then
  return cjson.encode({ok=false, code='video_one_missing'})
end

if video_two_job ~= nil and video_two_job ~= '' then
  return cjson.encode({
    ok=true,
    idempotent=true,
    jobId=video_two_job,
    videoIndex=2
  })
end

if job_id == '' then
  return cjson.encode({ok=false, code='invalid_job_id'})
end

table.insert(videos, {
  videoIndex=2,
  jobId=job_id,
  createdAt=created_at
})
data.videos = videos
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', ttl)
return cjson.encode({
  ok=true,
  idempotent=false,
  jobId=job_id,
  videoIndex=2
})
"""
)


class Builder2VideoAllowanceStoreError(Exception):
    def __init__(self, code: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class ReserveVideoTwoResult:
    ok: bool
    code: str = ""
    job_id: str = ""
    video_index: int = 0
    idempotent: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def allowance_key(video_allowance_id: str) -> str:
    return f"{ALLOWANCE_KEY_PREFIX}{(video_allowance_id or '').strip()}"


def enable_memory_allowance_store() -> None:
    global _use_memory_store, _memory_allowances
    _use_memory_store = True
    _memory_allowances = {}


def disable_memory_allowance_store() -> None:
    global _use_memory_store, _memory_allowances
    _use_memory_store = False
    _memory_allowances = {}


def clear_memory_allowance_store_for_tests() -> None:
    with _memory_lock:
        _memory_allowances.clear()


def _read_raw(video_allowance_id: str) -> Optional[Dict[str, Any]]:
    aid = (video_allowance_id or "").strip()
    if not aid:
        return None
    if _use_memory_store:
        stored = _memory_allowances.get(aid)
        return deepcopy(stored) if stored else None
    data = get_redis().get(allowance_key(aid))
    if not data:
        return None
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise Builder2VideoAllowanceStoreError("allowance_state_error")
    return parsed


def _write_raw(video_allowance_id: str, state: Dict[str, Any]) -> None:
    aid = (video_allowance_id or "").strip()
    if not aid:
        raise Builder2VideoAllowanceStoreError("invalid_allowance_id")
    state["updatedAt"] = _utc_now_iso()
    payload = json.dumps(state, ensure_ascii=False)
    if _use_memory_store:
        with _memory_lock:
            _memory_allowances[aid] = deepcopy(state)
        return
    get_redis().set(allowance_key(aid), payload, ex=ALLOWANCE_TTL_SECONDS)


def get_video_allowance(video_allowance_id: str) -> Optional[Dict[str, Any]]:
    return _read_raw(video_allowance_id)


def create_video_allowance(
    *,
    video_allowance_id: str,
    owner_context_ref: str,
    target_video_count: int,
    product_name: str,
    product_description: str,
    first_job_id: str,
) -> Dict[str, Any]:
    aid = (video_allowance_id or "").strip()
    owner_ref = (owner_context_ref or "").strip()
    first_jid = (first_job_id or "").strip()
    if not aid or not owner_ref or not first_jid:
        raise Builder2VideoAllowanceStoreError("invalid_allowance_create_args")
    if target_video_count not in {1, 2}:
        raise Builder2VideoAllowanceStoreError("invalid_target_video_count")

    now = _utc_now_iso()
    state: Dict[str, Any] = {
        "videoAllowanceId": aid,
        "ownerContextRef": owner_ref,
        "targetVideoCount": int(target_video_count),
        "productName": product_name or "",
        "productDescription": product_description or "",
        "createdAt": now,
        "videos": [
            {
                "videoIndex": 1,
                "jobId": first_jid,
                "createdAt": now,
            }
        ],
    }
    _write_raw(aid, state)
    logger.info(
        "BUILDER2_VIDEO_ALLOWANCE_CREATED videoAllowanceId=%s targetVideoCount=%s jobId=%s",
        aid,
        target_video_count,
        first_jid,
    )
    return deepcopy(state)


def _reserve_video_two_memory(
    video_allowance_id: str,
    *,
    owner_context_ref: str,
    job_id: str,
) -> ReserveVideoTwoResult:
    with _memory_lock:
        raw = _memory_allowances.get(video_allowance_id)
        if raw is None:
            return ReserveVideoTwoResult(ok=False, code="allowance_not_found")
        data = deepcopy(raw)
        if data.get("ownerContextRef") != owner_context_ref:
            return ReserveVideoTwoResult(ok=False, code="ownership_mismatch")
        if int(data.get("targetVideoCount") or 0) != 2:
            return ReserveVideoTwoResult(ok=False, code="target_video_count_not_two")
        videos = list(data.get("videos") or [])
        has_one = any(int(v.get("videoIndex") or 0) == 1 for v in videos)
        existing_two = next((v for v in videos if int(v.get("videoIndex") or 0) == 2), None)
        if not has_one:
            return ReserveVideoTwoResult(ok=False, code="video_one_missing")
        if existing_two:
            return ReserveVideoTwoResult(
                ok=True,
                job_id=str(existing_two.get("jobId") or ""),
                video_index=2,
                idempotent=True,
            )
        jid = (job_id or "").strip()
        if not jid:
            return ReserveVideoTwoResult(ok=False, code="invalid_job_id")
        videos.append({"videoIndex": 2, "jobId": jid, "createdAt": _utc_now_iso()})
        data["videos"] = videos
        _memory_allowances[video_allowance_id] = data
    logger.info(
        "BUILDER2_VIDEO_TWO_RESERVED videoAllowanceId=%s jobId=%s idempotent=false",
        video_allowance_id,
        job_id,
    )
    return ReserveVideoTwoResult(ok=True, job_id=job_id, video_index=2, idempotent=False)


def reserve_video_two_slot(
    video_allowance_id: str,
    *,
    owner_context_ref: str,
    job_id: str,
) -> ReserveVideoTwoResult:
    aid = (video_allowance_id or "").strip()
    owner_ref = (owner_context_ref or "").strip()
    jid = (job_id or "").strip()
    if not aid:
        return ReserveVideoTwoResult(ok=False, code="invalid_allowance_id")
    if _use_memory_store:
        return _reserve_video_two_memory(aid, owner_context_ref=owner_ref, job_id=jid)

    created_at = _utc_now_iso()
    for attempt in range(_MAX_LOCK_RETRIES):
        try:
            raw = get_redis().eval(
                _RESERVE_VIDEO_TWO_LUA,
                1,
                allowance_key(aid),
                owner_ref,
                jid,
                str(ALLOWANCE_TTL_SECONDS),
                created_at,
            )
            parsed = json.loads(raw)
            if not parsed.get("ok"):
                return ReserveVideoTwoResult(ok=False, code=str(parsed.get("code") or "reserve_failed"))
            return ReserveVideoTwoResult(
                ok=True,
                job_id=str(parsed.get("jobId") or ""),
                video_index=int(parsed.get("videoIndex") or 2),
                idempotent=bool(parsed.get("idempotent")),
            )
        except Exception:
            if attempt + 1 >= _MAX_LOCK_RETRIES:
                raise Builder2VideoAllowanceStoreError("allowance_reserve_failed") from None
            time.sleep(0.01 * (attempt + 1))
    return ReserveVideoTwoResult(ok=False, code="reserve_failed")
