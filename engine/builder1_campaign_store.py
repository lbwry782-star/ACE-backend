"""
Builder1 active campaign-session storage (Redis with in-memory fallback).

Each campaign is keyed by campaignId. targetAdCount is immutable after creation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_from_store_dict, series_plan_to_store_dict
from engine.builder1_image_retry import parse_image_attempt_history, union_violations_for_ad
from engine.builder1_retry_state import (
    RETRY_MODE_IMAGE_ONLY,
    RETRY_MODE_NONE,
    RETRY_MODE_REPAIR_FROM_PHYSICAL,
    normalize_retry_mode,
    resolve_authoritative_retry_mode,
)

logger = logging.getLogger(__name__)

CAMPAIGN_KEY_PREFIX = "builder1:campaign:"
CAMPAIGN_TTL_SECONDS = 24 * 3600

_memory_lock = threading.Lock()
_memory_campaigns: Dict[str, Dict[str, Any]] = {}

_LUA_IS_NULL = """
local function is_null(v)
  return v == nil or v == cjson.null
end
"""

_RESERVE_AD_INDEX_LUA = (
    _LUA_IS_NULL
    + """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='campaign_not_found'})
end
local data = cjson.decode(raw)
local expected = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local job_id = ARGV[3] or ''
local lock_token = ARGV[4] or ''
if data.complete then
  return cjson.encode({ok=false, code='campaign_complete'})
end
if not is_null(data.generatingIndex) then
  local reserved = tonumber(data.generatingIndex)
  local owner = data.generatingLockOwnerJobId
  if job_id ~= '' and owner == job_id and reserved == expected then
    return cjson.encode({
      ok=true,
      continued=true,
      adIndex=expected,
      lockToken=data.generatingLockToken,
      targetAdCount=data.targetAdCount,
      generatedCount=#(data.generatedIndexes or {})
    })
  end
  return cjson.encode({
    ok=false,
    code='campaign_generation_in_progress',
    reservedAdIndex=reserved,
    lockOwnerJobId=owner
  })
end
local next_idx = data.nextAdIndex
if is_null(next_idx) then
  return cjson.encode({ok=false, code='campaign_complete'})
end
if tonumber(next_idx) ~= expected then
  return cjson.encode({ok=false, code='campaign_index_conflict'})
end
for _, idx in ipairs(data.generatedIndexes or {}) do
  if idx == expected then
    return cjson.encode({ok=false, code='campaign_index_conflict'})
  end
end
data.generatingIndex = expected
if job_id ~= '' then
  data.generatingLockOwnerJobId = job_id
  if lock_token == '' then
    lock_token = job_id .. ':' .. tostring(expected) .. ':' .. tostring(redis.call('TIME')[1])
  end
  data.generatingLockToken = lock_token
  data.generatingLockAcquiredAt = tonumber(redis.call('TIME')[1])
end
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', ttl)
return cjson.encode({
  ok=true,
  continued=false,
  adIndex=expected,
  lockToken=lock_token,
  targetAdCount=data.targetAdCount,
  generatedCount=#(data.generatedIndexes or {})
})
"""
)

_MARK_AD_GENERATED_LUA = (
    _LUA_IS_NULL
    + """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='campaign_not_found'})
end
local data = cjson.decode(raw)
local ad_index = tonumber(ARGV[1])
if is_null(data.generatingIndex) or tonumber(data.generatingIndex) ~= ad_index then
  return cjson.encode({ok=false, code='campaign_index_conflict'})
end
local generated = data.generatedIndexes or {}
local found = false
for _, idx in ipairs(generated) do
  if idx == ad_index then
    found = true
    break
  end
end
if not found then
  table.insert(generated, ad_index)
  table.sort(generated)
end
data.generatedIndexes = generated
local target = tonumber(data.targetAdCount)
local generated_count = #generated
local complete = generated_count >= target
if complete then
  data.nextAdIndex = cjson.null
else
  data.nextAdIndex = ad_index + 1
end
data.complete = complete
data.generatedCount = generated_count
data.status = "active"
data.retryMode = "none"
data.failedAdIndex = cjson.null
data.lastImageViolations = cjson.null
data.preservedThroughStage = cjson.null
data.repairInProgress = false
data.generatingIndex = cjson.null
data.generatingLockOwnerJobId = cjson.null
data.generatingLockToken = cjson.null
data.generatingLockAcquiredAt = cjson.null
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', tonumber(ARGV[2]))
return cjson.encode({
  ok=true,
  generatedCount=generated_count,
  targetAdCount=target,
  complete=complete
})
"""
)

_RELEASE_RESERVATION_LUA = (
    _LUA_IS_NULL
    + """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='campaign_not_found'})
end
local data = cjson.decode(raw)
local req_job = ARGV[2] or ''
local req_token = ARGV[3] or ''
if not is_null(data.generatingIndex) then
  local owner = data.generatingLockOwnerJobId
  local token = data.generatingLockToken
  if req_job ~= '' and not is_null(owner) and owner ~= req_job then
    return cjson.encode({ok=false, code='campaign_lock_owner_mismatch'})
  end
  if req_token ~= '' and not is_null(token) and token ~= req_token then
    return cjson.encode({ok=false, code='campaign_lock_token_mismatch'})
  end
end
data.generatingIndex = cjson.null
data.generatingLockOwnerJobId = cjson.null
data.generatingLockToken = cjson.null
data.generatingLockAcquiredAt = cjson.null
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', tonumber(ARGV[1]))
return cjson.encode({ok=true})
"""
)


class CampaignStoreError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message or code
        super().__init__(self.message)


@dataclass
class Builder1CampaignSession:
    campaign_id: str
    target_ad_count: int
    ad_count: int
    format: str
    created_at: float
    next_ad_index: Optional[int]
    generated_indexes: List[int]
    generated_count: int
    complete: bool
    plan: Builder1SeriesPlan
    generating_index: Optional[int] = None
    generating_lock_owner_job_id: Optional[str] = None
    generating_lock_token: Optional[str] = None
    generating_lock_acquired_at: Optional[float] = None
    status: str = "active"
    planning_complete: bool = True
    failed_ad_index: Optional[int] = None
    last_image_violations: Optional[List[str]] = None
    preserved_through_stage: Optional[str] = None
    retry_mode: str = RETRY_MODE_NONE
    plan_revision: int = 1
    repair_in_progress: bool = False
    image_attempt_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


def get_campaign_store_backend() -> str:
    return "redis" if _redis_configured() else "memory"


def _redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def _campaign_key(campaign_id: str) -> str:
    return f"{CAMPAIGN_KEY_PREFIX}{campaign_id}"


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def _target_from_raw(raw: Dict[str, Any]) -> int:
    plan_data = raw.get("plan") or {}
    return int(raw.get("targetAdCount") or raw.get("adCount") or plan_data.get("adCount") or 2)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_plan_revision_from_raw(raw: Dict[str, Any], *, campaign_id: str = "") -> int:
    if "planRevision" not in raw:
        logger.warning(
            "BUILDER1_PLAN_REVISION_LEGACY_MISSING campaignId=%s fallback=1",
            campaign_id or raw.get("campaignId") or "",
        )
    try:
        revision = int(raw.get("planRevision") or 1)
    except (TypeError, ValueError):
        revision = 1
    return max(1, revision)


def _session_from_raw(campaign_id: str, raw: Dict[str, Any]) -> Builder1CampaignSession:
    plan_data = raw.get("plan") or {}
    plan = series_plan_from_store_dict(plan_data)
    generated = sorted(int(x) for x in (raw.get("generatedIndexes") or []))
    next_idx = _optional_int(raw.get("nextAdIndex"))
    target = _target_from_raw(raw)
    generated_count = int(raw.get("generatedCount") if raw.get("generatedCount") is not None else len(generated))
    return Builder1CampaignSession(
        campaign_id=campaign_id,
        target_ad_count=target,
        ad_count=target,
        format=str(raw.get("format") or plan.format),
        created_at=float(raw.get("createdAt") or time.time()),
        next_ad_index=next_idx,
        generated_indexes=generated,
        generated_count=generated_count,
        complete=bool(raw.get("complete")),
        plan=plan,
        generating_index=_optional_int(raw.get("generatingIndex")),
        generating_lock_owner_job_id=str(raw.get("generatingLockOwnerJobId") or "").strip() or None,
        generating_lock_token=str(raw.get("generatingLockToken") or "").strip() or None,
        generating_lock_acquired_at=(
            float(raw["generatingLockAcquiredAt"])
            if raw.get("generatingLockAcquiredAt") is not None
            else None
        ),
        status=str(raw.get("status") or "active"),
        planning_complete=bool(raw.get("planningComplete", True)),
        failed_ad_index=_optional_int(raw.get("failedAdIndex")),
        last_image_violations=list(raw.get("lastImageViolations") or []) or None,
        preserved_through_stage=str(raw.get("preservedThroughStage") or "").strip() or None,
        retry_mode=resolve_authoritative_retry_mode(
            status=str(raw.get("status") or "active"),
            retry_mode=str(raw.get("retryMode") or RETRY_MODE_NONE),
        ),
        plan_revision=_resolve_plan_revision_from_raw(raw, campaign_id=campaign_id),
        repair_in_progress=bool(raw.get("repairInProgress")),
        image_attempt_history=parse_image_attempt_history(raw.get("imageAttemptHistory")),
    )


def _session_to_raw(session: Builder1CampaignSession) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "campaignId": session.campaign_id,
        "targetAdCount": session.target_ad_count,
        "adCount": session.target_ad_count,
        "generatedCount": session.generated_count,
        "format": session.format,
        "createdAt": session.created_at,
        "generatedIndexes": list(session.generated_indexes),
        "complete": session.complete,
        "plan": series_plan_to_store_dict(session.plan),
        "planningComplete": session.planning_complete,
        "status": session.status,
        "retryMode": session.retry_mode,
        "planRevision": session.plan_revision,
        "repairInProgress": session.repair_in_progress,
    }
    if session.image_attempt_history:
        raw["imageAttemptHistory"] = session.image_attempt_history
    if session.failed_ad_index is not None:
        raw["failedAdIndex"] = session.failed_ad_index
    if session.last_image_violations:
        raw["lastImageViolations"] = list(session.last_image_violations)
    if session.preserved_through_stage:
        raw["preservedThroughStage"] = session.preserved_through_stage
    if session.next_ad_index is not None:
        raw["nextAdIndex"] = session.next_ad_index
    if session.generating_index is not None:
        raw["generatingIndex"] = session.generating_index
        if session.generating_lock_owner_job_id:
            raw["generatingLockOwnerJobId"] = session.generating_lock_owner_job_id
        if session.generating_lock_token:
            raw["generatingLockToken"] = session.generating_lock_token
        if session.generating_lock_acquired_at is not None:
            raw["generatingLockAcquiredAt"] = session.generating_lock_acquired_at
    return raw


def _load_raw(campaign_id: str) -> Optional[Dict[str, Any]]:
    cid = (campaign_id or "").strip()
    if not cid:
        return None
    if _redis_configured():
        try:
            raw = _get_redis().get(_campaign_key(cid))
            if not raw:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_REDIS_LOAD_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    with _memory_lock:
        stored = _memory_campaigns.get(cid)
        return dict(stored) if stored is not None else None


def _save_raw(campaign_id: str, data: Dict[str, Any], *, create: bool = False) -> None:
    cid = (campaign_id or "").strip()
    if not create:
        existing = _load_raw(cid)
        if existing is not None:
            data["targetAdCount"] = _target_from_raw(existing)
            data["adCount"] = data["targetAdCount"]
            data["createdAt"] = existing.get("createdAt", data.get("createdAt"))
    if _redis_configured():
        try:
            _get_redis().set(
                _campaign_key(cid),
                json.dumps(data, ensure_ascii=False),
                ex=CAMPAIGN_TTL_SECONDS,
            )
            return
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_REDIS_SAVE_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    with _memory_lock:
        _memory_campaigns[cid] = data


def _decode_lua_result(raw: object) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    raise CampaignStoreError("campaign_store_error", "invalid_lua_result")


def _lock_age_ms(acquired_at: Optional[float]) -> Optional[int]:
    if acquired_at is None:
        return None
    return max(0, int((time.time() - acquired_at) * 1000))


def _raise_from_lua(
    code: str,
    *,
    campaign_id: str,
    requesting_job_id: str = "",
    reserved_ad_index: Optional[int] = None,
    lock_owner_job_id: Optional[str] = None,
    lock_age_ms: Optional[int] = None,
) -> None:
    if code == "campaign_generation_in_progress":
        same_owner = bool(
            requesting_job_id
            and lock_owner_job_id
            and requesting_job_id == lock_owner_job_id
        )
        logger.info(
            "BUILDER1_CONCURRENT_REQUEST_BLOCKED campaignId=%s requestingJobId=%s "
            "lockOwnerJobId=%s reservedAdIndex=%s lockAgeMs=%s sameOwner=%s",
            campaign_id,
            requesting_job_id or None,
            lock_owner_job_id or None,
            reserved_ad_index,
            lock_age_ms,
            same_owner,
        )
    raise CampaignStoreError(code)


def _new_lock_token(*, job_id: str, ad_index: int) -> str:
    return f"{job_id}:{ad_index}:{uuid.uuid4().hex}"


def _reserve_ad_index_memory(
    campaign_id: str,
    expected_index: int,
    *,
    job_id: str = "",
    lock_token: str = "",
) -> Builder1CampaignSession:
    with _memory_lock:
        raw = _memory_campaigns.get(campaign_id)
        if raw is None:
            raise CampaignStoreError("campaign_not_found")
        raw = dict(raw)
        session = _session_from_raw(campaign_id, raw)
        if session.generating_index is not None:
            if job_id and session.generating_lock_owner_job_id == job_id and session.generating_index == expected_index:
                logger.info(
                    "BUILDER1_GENERATION_LOCK_CONTINUED campaignId=%s jobId=%s reservedAdIndex=%s",
                    campaign_id,
                    job_id,
                    expected_index,
                )
                return session
            logger.info(
                "BUILDER1_CONCURRENT_REQUEST_BLOCKED campaignId=%s requestingJobId=%s "
                "lockOwnerJobId=%s reservedAdIndex=%s lockAgeMs=%s sameOwner=%s",
                campaign_id,
                job_id or None,
                session.generating_lock_owner_job_id or None,
                session.generating_index,
                _lock_age_ms(session.generating_lock_acquired_at),
                bool(job_id and session.generating_lock_owner_job_id == job_id),
            )
            raise CampaignStoreError("campaign_generation_in_progress")
        _validate_next_index(session, expected_index)
        token = lock_token or (_new_lock_token(job_id=job_id, ad_index=expected_index) if job_id else "")
        raw["generatingIndex"] = expected_index
        if job_id:
            raw["generatingLockOwnerJobId"] = job_id
            raw["generatingLockToken"] = token
            raw["generatingLockAcquiredAt"] = time.time()
        _memory_campaigns[campaign_id] = raw
        session = _session_from_raw(campaign_id, raw)
    logger.info(
        "BUILDER1_NEXT_AD_RESERVED campaignId=%s adIndex=%s targetAdCount=%s generatedCount=%s jobId=%s planRevision=%s",
        campaign_id,
        expected_index,
        session.target_ad_count,
        session.generated_count,
        job_id or None,
        session.plan_revision,
    )
    return session


def _mark_ad_generated_memory(campaign_id: str, ad_index: int) -> Builder1CampaignSession:
    with _memory_lock:
        raw = _memory_campaigns.get(campaign_id)
        if raw is None:
            raise CampaignStoreError("campaign_not_found")
        raw = dict(raw)
        session = _session_from_raw(campaign_id, raw)
        if session.generating_index != ad_index:
            raise CampaignStoreError("campaign_index_conflict")
        if ad_index in session.generated_indexes:
            raise CampaignStoreError("campaign_index_conflict")
        generated = sorted(set(session.generated_indexes + [ad_index]))
        generated_count = len(generated)
        target = session.target_ad_count
        complete = generated_count >= target
        next_idx: Optional[int] = ad_index + 1 if ad_index < target else None
        raw["generatedIndexes"] = generated
        raw["generatedCount"] = generated_count
        if next_idx is not None:
            raw["nextAdIndex"] = next_idx
        else:
            raw.pop("nextAdIndex", None)
        raw["complete"] = complete
        raw["status"] = "active"
        raw["retryMode"] = RETRY_MODE_NONE
        raw.pop("failedAdIndex", None)
        raw.pop("lastImageViolations", None)
        raw.pop("preservedThroughStage", None)
        raw["repairInProgress"] = False
        history = dict(raw.get("imageAttemptHistory") or {})
        history.pop(str(ad_index), None)
        if history:
            raw["imageAttemptHistory"] = history
        else:
            raw.pop("imageAttemptHistory", None)
        raw.pop("generatingIndex", None)
        raw.pop("generatingLockOwnerJobId", None)
        raw.pop("generatingLockToken", None)
        raw.pop("generatingLockAcquiredAt", None)
        _memory_campaigns[campaign_id] = raw
        session = _session_from_raw(campaign_id, raw)
    _log_campaign_progress(session, ad_index=ad_index)
    return session


def _release_reservation_memory(
    campaign_id: str,
    *,
    job_id: str = "",
    lock_token: str = "",
) -> None:
    with _memory_lock:
        raw = _memory_campaigns.get(campaign_id)
        if raw is None:
            return
        raw = dict(raw)
        session = _session_from_raw(campaign_id, raw)
        if session.generating_index is not None:
            if job_id and session.generating_lock_owner_job_id and session.generating_lock_owner_job_id != job_id:
                raise CampaignStoreError("campaign_lock_owner_mismatch")
            if lock_token and session.generating_lock_token and session.generating_lock_token != lock_token:
                raise CampaignStoreError("campaign_lock_token_mismatch")
        raw.pop("generatingIndex", None)
        raw.pop("generatingLockOwnerJobId", None)
        raw.pop("generatingLockToken", None)
        raw.pop("generatingLockAcquiredAt", None)
        _memory_campaigns[campaign_id] = raw


def _log_campaign_progress(session: Builder1CampaignSession, *, ad_index: int) -> None:
    logger.info(
        "BUILDER1_CAMPAIGN_PROGRESS campaignId=%s generatedCount=%s targetAdCount=%s complete=%s",
        session.campaign_id,
        session.generated_count,
        session.target_ad_count,
        session.complete,
    )
    logger.info(
        "BUILDER1_AD_GENERATED campaignId=%s adIndex=%s generatedCount=%s complete=%s planRevision=%s",
        session.campaign_id,
        ad_index,
        session.generated_count,
        session.complete,
        session.plan_revision,
    )
    if session.complete:
        logger.info("BUILDER1_CAMPAIGN_COMPLETE campaignId=%s", session.campaign_id)


def create_campaign_session(
    *,
    campaign_id: str,
    plan: Builder1SeriesPlan,
    target_ad_count: Optional[int] = None,
) -> Builder1CampaignSession:
    target = int(target_ad_count if target_ad_count is not None else plan.ad_count)
    session = Builder1CampaignSession(
        campaign_id=campaign_id,
        target_ad_count=target,
        ad_count=target,
        format=plan.format,
        created_at=time.time(),
        next_ad_index=1,
        generated_indexes=[],
        generated_count=0,
        complete=False,
        plan=plan,
    )
    _save_raw(campaign_id, _session_to_raw(session), create=True)
    logger.info(
        "BUILDER1_CAMPAIGN_CREATED campaignId=%s targetAdCount=%s backend=%s planRevision=%s",
        campaign_id,
        target,
        get_campaign_store_backend(),
        session.plan_revision,
    )
    logger.info(
        "BUILDER1_CAMPAIGN_SESSION_CREATED campaignId=%s targetAdCount=%s",
        campaign_id,
        target,
    )
    return session


def get_campaign_session(campaign_id: str) -> Builder1CampaignSession:
    raw = _load_raw(campaign_id)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    return _session_from_raw(campaign_id, raw)


def reserve_next_ad_index(
    campaign_id: str,
    expected_index: int,
    *,
    job_id: str = "",
    lock_token: str = "",
) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    token = lock_token or (_new_lock_token(job_id=job_id, ad_index=expected_index) if job_id else "")
    pre_session = get_campaign_session(cid)
    _validate_next_index(pre_session, expected_index)
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_RESERVE_AD_INDEX_LUA)
            result = _decode_lua_result(
                script(keys=[_campaign_key(cid)], args=[expected_index, CAMPAIGN_TTL_SECONDS, job_id, token])
            )
            if not result.get("ok"):
                owner = result.get("lockOwnerJobId")
                lock_owner_job_id = None if owner in (None, "") else str(owner)
                _raise_from_lua(
                    str(result.get("code") or "campaign_store_error"),
                    campaign_id=cid,
                    requesting_job_id=job_id,
                    reserved_ad_index=_optional_int(result.get("reservedAdIndex")),
                    lock_owner_job_id=lock_owner_job_id,
                )
            session = get_campaign_session(cid)
            if result.get("continued"):
                logger.info(
                    "BUILDER1_GENERATION_LOCK_CONTINUED campaignId=%s jobId=%s reservedAdIndex=%s",
                    cid,
                    job_id,
                    expected_index,
                )
            else:
                logger.info(
                    "BUILDER1_NEXT_AD_RESERVED campaignId=%s adIndex=%s targetAdCount=%s generatedCount=%s jobId=%s planRevision=%s",
                    cid,
                    expected_index,
                    session.target_ad_count,
                    session.generated_count,
                    job_id or None,
                    session.plan_revision,
                )
            return session
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_RESERVE_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    return _reserve_ad_index_memory(cid, expected_index, job_id=job_id, lock_token=token)


def try_acquire_generation_lock(
    campaign_id: str,
    ad_index: int,
    *,
    job_id: str = "",
    lock_token: str = "",
) -> Builder1CampaignSession:
    """Backward-compatible alias for atomic reservation."""
    return reserve_next_ad_index(
        campaign_id,
        ad_index,
        job_id=job_id,
        lock_token=lock_token,
    )


def release_generation_lock(
    campaign_id: str,
    *,
    job_id: str = "",
    lock_token: str = "",
) -> None:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_RELEASE_RESERVATION_LUA)
            result = _decode_lua_result(
                script(keys=[_campaign_key(cid)], args=[CAMPAIGN_TTL_SECONDS, job_id, lock_token])
            )
            if not result.get("ok"):
                code = str(result.get("code") or "campaign_store_error")
                if code != "campaign_not_found":
                    raise CampaignStoreError(code)
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_UNLOCK_ERR campaignId=%s err=%s", cid, exc)
        return
    _release_reservation_memory(cid, job_id=job_id, lock_token=lock_token)


def mark_ad_generated(campaign_id: str, ad_index: int) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_MARK_AD_GENERATED_LUA)
            result = _decode_lua_result(
                script(keys=[_campaign_key(cid)], args=[ad_index, CAMPAIGN_TTL_SECONDS])
            )
            if not result.get("ok"):
                _raise_from_lua(str(result.get("code") or "campaign_store_error"), campaign_id=cid)
            session = get_campaign_session(cid)
            _clear_image_attempt_history_for_ad(cid, ad_index)
            session = get_campaign_session(cid)
            _log_campaign_progress(session, ad_index=ad_index)
            return session
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_MARK_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    return _mark_ad_generated_memory(cid, ad_index)


def _validate_image_generation_allowed(session: Builder1CampaignSession) -> None:
    if session.repair_in_progress:
        raise CampaignStoreError("physical_repair_not_completed")
    mode = resolve_authoritative_retry_mode(status=session.status, retry_mode=session.retry_mode)
    if mode == RETRY_MODE_REPAIR_FROM_PHYSICAL:
        raise CampaignStoreError("physical_repair_not_completed")


def _validate_next_index(session: Builder1CampaignSession, expected_index: int) -> None:
    if session.complete:
        raise CampaignStoreError("campaign_complete")
    if expected_index in session.generated_indexes:
        raise CampaignStoreError("campaign_index_conflict")
    if session.next_ad_index != expected_index:
        raise CampaignStoreError("campaign_index_conflict")
    if expected_index < 1 or expected_index > session.target_ad_count:
        raise CampaignStoreError("campaign_index_conflict")
    if session.generated_count >= session.target_ad_count:
        raise CampaignStoreError("campaign_complete")
    _validate_image_generation_allowed(session)


def _clear_image_attempt_history_for_ad(campaign_id: str, ad_index: int) -> None:
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        return
    raw = dict(raw)
    history = dict(raw.get("imageAttemptHistory") or {})
    history.pop(str(ad_index), None)
    if history:
        raw["imageAttemptHistory"] = history
    else:
        raw.pop("imageAttemptHistory", None)
    _save_raw(cid, raw)


def cumulative_violations_for_ad(session: Builder1CampaignSession, ad_index: int) -> List[str]:
    return union_violations_for_ad(session.image_attempt_history or {}, ad_index)


def record_image_attempt_violations(
    campaign_id: str,
    *,
    ad_index: int,
    attempt: int,
    violations: List[str],
) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    raw = dict(raw)
    history = parse_image_attempt_history(raw.get("imageAttemptHistory"))
    key = str(ad_index)
    entries = list(history.get(key) or [])
    normalized = list(dict.fromkeys(str(v).strip() for v in violations if str(v).strip()))
    if not normalized:
        return _session_from_raw(cid, raw)
    if entries and int(entries[-1].get("attempt") or 0) == int(attempt):
        prior = list(entries[-1].get("violations") or [])
        entries[-1] = {
            "attempt": int(attempt),
            "violations": list(dict.fromkeys(prior + normalized)),
        }
    else:
        entries.append({"attempt": int(attempt), "violations": normalized})
    history[key] = entries
    raw["imageAttemptHistory"] = history
    _save_raw(cid, raw)
    return _session_from_raw(cid, raw)


def mark_physical_repair_required(
    campaign_id: str,
    *,
    failed_ad_index: int,
    violations: List[str],
    preserved_through_stage: str = "conceptual_stage",
) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    raw = dict(raw)
    raw["status"] = "physical_repair_required"
    raw["planningComplete"] = True
    raw["failedAdIndex"] = int(failed_ad_index)
    raw["lastImageViolations"] = list(violations)
    raw["preservedThroughStage"] = preserved_through_stage
    raw["retryMode"] = RETRY_MODE_REPAIR_FROM_PHYSICAL
    raw["repairInProgress"] = False
    raw["planRevision"] = max(1, int(raw.get("planRevision") or 1))
    _save_raw(cid, raw)
    session = _session_from_raw(cid, raw)
    logger.info(
        "BUILDER1_PHYSICAL_REPAIR_REQUIRED campaignId=%s failedAdIndex=%s preservedThroughStage=%s generatedCount=%s planRevision=%s violations=%s",
        cid,
        failed_ad_index,
        preserved_through_stage,
        session.generated_count,
        session.plan_revision,
        violations,
    )
    return session


def apply_repaired_campaign_plan(campaign_id: str, plan: Builder1SeriesPlan) -> Builder1CampaignSession:
    """Atomically persist a repaired downstream plan and advance plan revision."""
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    raw = dict(raw)
    previous_revision = max(1, int(raw.get("planRevision") or 1))
    raw["plan"] = series_plan_to_store_dict(plan)
    raw["planRevision"] = previous_revision + 1
    raw["status"] = "active"
    raw["retryMode"] = RETRY_MODE_IMAGE_ONLY
    raw["repairInProgress"] = False
    _save_raw(cid, raw)
    session = _session_from_raw(cid, raw)
    logger.info(
        "BUILDER1_CAMPAIGN_PLAN_REPAIRED campaignId=%s planRevision=%s failedAdIndex=%s generatedCount=%s",
        cid,
        session.plan_revision,
        session.failed_ad_index,
        session.generated_count,
    )
    return session


def update_campaign_plan(campaign_id: str, plan: Builder1SeriesPlan) -> Builder1CampaignSession:
    return apply_repaired_campaign_plan(campaign_id, plan)


def begin_physical_repair(campaign_id: str, *, job_id: str = "") -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    raw = dict(raw)
    mode = resolve_authoritative_retry_mode(
        status=str(raw.get("status") or "active"),
        retry_mode=str(raw.get("retryMode") or RETRY_MODE_NONE),
    )
    if mode != RETRY_MODE_REPAIR_FROM_PHYSICAL:
        raise CampaignStoreError("physical_repair_not_required")
    raw["repairInProgress"] = True
    _save_raw(cid, raw)
    session = _session_from_raw(cid, raw)
    logger.info(
        "BUILDER1_PHYSICAL_REPAIR_STARTED campaignId=%s jobId=%s failedAdIndex=%s planRevision=%s",
        cid,
        job_id or None,
        session.failed_ad_index,
        session.plan_revision,
    )
    return session


def cancel_physical_repair_in_progress(campaign_id: str) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    raw = dict(raw)
    raw["repairInProgress"] = False
    _save_raw(cid, raw)
    return _session_from_raw(cid, raw)


def mark_image_retry_required(
    campaign_id: str,
    *,
    failed_ad_index: int,
    violations: List[str],
) -> Builder1CampaignSession:
    """Persist image failure state without discarding the planned campaign."""
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    raw = dict(raw)
    raw["status"] = "image_retry_required"
    raw["planningComplete"] = True
    raw["failedAdIndex"] = int(failed_ad_index)
    raw["lastImageViolations"] = list(violations)
    raw["retryMode"] = RETRY_MODE_IMAGE_ONLY
    raw["repairInProgress"] = False
    raw["planRevision"] = max(1, int(raw.get("planRevision") or 1))
    _save_raw(cid, raw)
    session = _session_from_raw(cid, raw)
    logger.info(
        "BUILDER1_IMAGE_RETRY_REQUIRED campaignId=%s failedAdIndex=%s generatedCount=%s targetAdCount=%s planRevision=%s violations=%s",
        cid,
        failed_ad_index,
        session.generated_count,
        session.target_ad_count,
        session.plan_revision,
        violations,
    )
    return session


def validate_next_ad_request(campaign_id: str, expected_next_index: int) -> Builder1CampaignSession:
    try:
        session = get_campaign_session(campaign_id)
    except CampaignStoreError:
        raise
    except Exception as exc:
        logger.error("BUILDER1_CAMPAIGN_LOAD_ERR campaignId=%s err=%s", campaign_id, exc)
        raise CampaignStoreError("campaign_expired") from exc
    if session.complete:
        raise CampaignStoreError("campaign_complete")
    if session.generating_index is not None:
        raise CampaignStoreError("campaign_generation_in_progress")
    if session.repair_in_progress:
        raise CampaignStoreError("physical_repair_not_completed")
    if expected_next_index != session.next_ad_index:
        raise CampaignStoreError("campaign_index_conflict")
    if expected_next_index in session.generated_indexes:
        raise CampaignStoreError("campaign_index_conflict")
    return session


def clear_memory_store_for_tests() -> None:
    with _memory_lock:
        _memory_campaigns.clear()
