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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_from_store_dict, series_plan_to_store_dict

logger = logging.getLogger(__name__)

CAMPAIGN_KEY_PREFIX = "builder1:campaign:"
CAMPAIGN_TTL_SECONDS = 24 * 3600

_memory_lock = threading.Lock()
_memory_campaigns: Dict[str, Dict[str, Any]] = {}

_RESERVE_AD_INDEX_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='campaign_not_found'})
end
local data = cjson.decode(raw)
local expected = tonumber(ARGV[1])
if data.complete then
  return cjson.encode({ok=false, code='campaign_complete'})
end
if data.generatingIndex ~= nil then
  return cjson.encode({ok=false, code='campaign_generation_in_progress', reservedAdIndex=data.generatingIndex})
end
local next_idx = data.nextAdIndex
if next_idx == nil then
  return cjson.encode({ok=false, code='campaign_complete'})
end
if next_idx ~= expected then
  return cjson.encode({ok=false, code='campaign_index_conflict'})
end
for _, idx in ipairs(data.generatedIndexes or {}) do
  if idx == expected then
    return cjson.encode({ok=false, code='campaign_index_conflict'})
  end
end
data.generatingIndex = expected
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', tonumber(ARGV[2]))
return cjson.encode({
  ok=true,
  adIndex=expected,
  targetAdCount=data.targetAdCount,
  generatedCount=#(data.generatedIndexes or {})
})
"""

_MARK_AD_GENERATED_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='campaign_not_found'})
end
local data = cjson.decode(raw)
local ad_index = tonumber(ARGV[1])
if data.generatingIndex ~= ad_index then
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
data.generatingIndex = cjson.null
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', tonumber(ARGV[2]))
return cjson.encode({
  ok=true,
  generatedCount=generated_count,
  targetAdCount=target,
  complete=complete
})
"""

_RELEASE_RESERVATION_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return cjson.encode({ok=false, code='campaign_not_found'})
end
local data = cjson.decode(raw)
data.generatingIndex = cjson.null
redis.call('SET', KEYS[1], cjson.encode(data), 'EX', tonumber(ARGV[1]))
return cjson.encode({ok=true})
"""


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


def _session_from_raw(campaign_id: str, raw: Dict[str, Any]) -> Builder1CampaignSession:
    plan_data = raw.get("plan") or {}
    plan = series_plan_from_store_dict(plan_data)
    generated = sorted(int(x) for x in (raw.get("generatedIndexes") or []))
    next_idx = raw.get("nextAdIndex")
    target = _target_from_raw(raw)
    generated_count = int(raw.get("generatedCount") if raw.get("generatedCount") is not None else len(generated))
    return Builder1CampaignSession(
        campaign_id=campaign_id,
        target_ad_count=target,
        ad_count=target,
        format=str(raw.get("format") or plan.format),
        created_at=float(raw.get("createdAt") or time.time()),
        next_ad_index=None if next_idx is None else int(next_idx),
        generated_indexes=generated,
        generated_count=generated_count,
        complete=bool(raw.get("complete")),
        plan=plan,
        generating_index=(
            int(raw["generatingIndex"]) if raw.get("generatingIndex") is not None else None
        ),
    )


def _session_to_raw(session: Builder1CampaignSession) -> Dict[str, Any]:
    return {
        "campaignId": session.campaign_id,
        "targetAdCount": session.target_ad_count,
        "adCount": session.target_ad_count,
        "generatedCount": session.generated_count,
        "format": session.format,
        "createdAt": session.created_at,
        "nextAdIndex": session.next_ad_index,
        "generatedIndexes": list(session.generated_indexes),
        "complete": session.complete,
        "generatingIndex": session.generating_index,
        "plan": series_plan_to_store_dict(session.plan),
    }


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


def _raise_from_lua(code: str, *, campaign_id: str, reserved_ad_index: Optional[int] = None) -> None:
    if code == "campaign_generation_in_progress":
        logger.info(
            "BUILDER1_CONCURRENT_REQUEST_BLOCKED campaignId=%s reservedAdIndex=%s",
            campaign_id,
            reserved_ad_index,
        )
    raise CampaignStoreError(code)


def _reserve_ad_index_memory(campaign_id: str, expected_index: int) -> Builder1CampaignSession:
    with _memory_lock:
        raw = _memory_campaigns.get(campaign_id)
        if raw is None:
            raise CampaignStoreError("campaign_not_found")
        raw = dict(raw)
        session = _session_from_raw(campaign_id, raw)
        _validate_next_index(session, expected_index)
        if session.generating_index is not None:
            logger.info(
                "BUILDER1_CONCURRENT_REQUEST_BLOCKED campaignId=%s reservedAdIndex=%s",
                campaign_id,
                session.generating_index,
            )
            raise CampaignStoreError("campaign_generation_in_progress")
        raw["generatingIndex"] = expected_index
        _memory_campaigns[campaign_id] = raw
        session = _session_from_raw(campaign_id, raw)
    logger.info(
        "BUILDER1_NEXT_AD_RESERVED campaignId=%s adIndex=%s targetAdCount=%s generatedCount=%s",
        campaign_id,
        expected_index,
        session.target_ad_count,
        session.generated_count,
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
        raw["nextAdIndex"] = None if complete else next_idx
        raw["complete"] = complete
        raw["generatingIndex"] = None
        _memory_campaigns[campaign_id] = raw
        session = _session_from_raw(campaign_id, raw)
    _log_campaign_progress(session, ad_index=ad_index)
    return session


def _release_reservation_memory(campaign_id: str) -> None:
    with _memory_lock:
        raw = _memory_campaigns.get(campaign_id)
        if raw is None:
            return
        raw = dict(raw)
        raw["generatingIndex"] = None
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
        "BUILDER1_AD_GENERATED campaignId=%s adIndex=%s generatedCount=%s complete=%s",
        session.campaign_id,
        ad_index,
        session.generated_count,
        session.complete,
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
        generating_index=None,
    )
    _save_raw(campaign_id, _session_to_raw(session), create=True)
    logger.info(
        "BUILDER1_CAMPAIGN_CREATED campaignId=%s targetAdCount=%s backend=%s",
        campaign_id,
        target,
        get_campaign_store_backend(),
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


def reserve_next_ad_index(campaign_id: str, expected_index: int) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_RESERVE_AD_INDEX_LUA)
            result = _decode_lua_result(script(keys=[_campaign_key(cid)], args=[expected_index, CAMPAIGN_TTL_SECONDS]))
            if not result.get("ok"):
                _raise_from_lua(
                    str(result.get("code") or "campaign_store_error"),
                    campaign_id=cid,
                    reserved_ad_index=result.get("reservedAdIndex"),
                )
            session = get_campaign_session(cid)
            logger.info(
                "BUILDER1_NEXT_AD_RESERVED campaignId=%s adIndex=%s targetAdCount=%s generatedCount=%s",
                cid,
                expected_index,
                session.target_ad_count,
                session.generated_count,
            )
            return session
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_RESERVE_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    return _reserve_ad_index_memory(cid, expected_index)


def try_acquire_generation_lock(campaign_id: str, ad_index: int) -> Builder1CampaignSession:
    """Backward-compatible alias for atomic reservation."""
    return reserve_next_ad_index(campaign_id, ad_index)


def release_generation_lock(campaign_id: str) -> None:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_RELEASE_RESERVATION_LUA)
            result = _decode_lua_result(script(keys=[_campaign_key(cid)], args=[CAMPAIGN_TTL_SECONDS]))
            if not result.get("ok"):
                code = str(result.get("code") or "campaign_store_error")
                if code != "campaign_not_found":
                    raise CampaignStoreError(code)
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_UNLOCK_ERR campaignId=%s err=%s", cid, exc)
        return
    _release_reservation_memory(cid)


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
            _log_campaign_progress(session, ad_index=ad_index)
            return session
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_MARK_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    return _mark_ad_generated_memory(cid, ad_index)


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
    if expected_next_index != session.next_ad_index:
        raise CampaignStoreError("campaign_index_conflict")
    if expected_next_index in session.generated_indexes:
        raise CampaignStoreError("campaign_index_conflict")
    return session


def clear_memory_store_for_tests() -> None:
    with _memory_lock:
        _memory_campaigns.clear()
