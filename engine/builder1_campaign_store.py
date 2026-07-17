"""
Builder1 active campaign-session storage (Redis with in-memory fallback).
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
LOCK_KEY_SUFFIX = ":lock"
CAMPAIGN_TTL_SECONDS = 24 * 3600
LOCK_TTL_SECONDS = 300

_memory_lock = threading.Lock()
_memory_campaigns: Dict[str, Dict[str, Any]] = {}
_memory_campaign_locks: Dict[str, threading.Lock] = {}


class CampaignStoreError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message or code
        super().__init__(self.message)


def _redis_configured() -> bool:
    return bool((os.environ.get("REDIS_URL") or "").strip())


def _campaign_key(campaign_id: str) -> str:
    return f"{CAMPAIGN_KEY_PREFIX}{campaign_id}"


def _lock_key(campaign_id: str) -> str:
    return f"{CAMPAIGN_KEY_PREFIX}{campaign_id}{LOCK_KEY_SUFFIX}"


def _get_redis():
    from engine.video_jobs_redis import get_redis

    return get_redis()


def _memory_campaign_lock(campaign_id: str) -> threading.Lock:
    with _memory_lock:
        if campaign_id not in _memory_campaign_locks:
            _memory_campaign_locks[campaign_id] = threading.Lock()
        return _memory_campaign_locks[campaign_id]


@dataclass
class Builder1CampaignSession:
    campaign_id: str
    ad_count: int
    format: str
    created_at: float
    next_ad_index: Optional[int]
    generated_indexes: List[int]
    complete: bool
    plan: Builder1SeriesPlan
    generating_index: Optional[int] = None


def _session_from_raw(campaign_id: str, raw: Dict[str, Any]) -> Builder1CampaignSession:
    plan_data = raw.get("plan") or {}
    plan = series_plan_from_store_dict(plan_data)
    generated = [int(x) for x in (raw.get("generatedIndexes") or [])]
    next_idx = raw.get("nextAdIndex")
    return Builder1CampaignSession(
        campaign_id=campaign_id,
        ad_count=int(raw.get("adCount") or plan.ad_count),
        format=str(raw.get("format") or plan.format),
        created_at=float(raw.get("createdAt") or time.time()),
        next_ad_index=None if next_idx is None else int(next_idx),
        generated_indexes=sorted(generated),
        complete=bool(raw.get("complete")),
        plan=plan,
        generating_index=(
            int(raw["generatingIndex"]) if raw.get("generatingIndex") is not None else None
        ),
    )


def _session_to_raw(session: Builder1CampaignSession) -> Dict[str, Any]:
    return {
        "campaignId": session.campaign_id,
        "adCount": session.ad_count,
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
        return _memory_campaigns.get(cid)


def _save_raw(campaign_id: str, data: Dict[str, Any]) -> None:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            _get_redis().set(_campaign_key(cid), json.dumps(data, ensure_ascii=False), ex=CAMPAIGN_TTL_SECONDS)
            return
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_REDIS_SAVE_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc
    with _memory_lock:
        _memory_campaigns[cid] = data


def create_campaign_session(
    *,
    campaign_id: str,
    plan: Builder1SeriesPlan,
) -> Builder1CampaignSession:
    session = Builder1CampaignSession(
        campaign_id=campaign_id,
        ad_count=plan.ad_count,
        format=plan.format,
        created_at=time.time(),
        next_ad_index=1,
        generated_indexes=[],
        complete=False,
        plan=plan,
        generating_index=None,
    )
    _save_raw(campaign_id, _session_to_raw(session))
    logger.info(
        "BUILDER1_CAMPAIGN_SESSION_CREATED campaignId=%s adCount=%s",
        campaign_id,
        plan.ad_count,
    )
    return session


def get_campaign_session(campaign_id: str) -> Builder1CampaignSession:
    raw = _load_raw(campaign_id)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    return _session_from_raw(campaign_id, raw)


def try_acquire_generation_lock(campaign_id: str, ad_index: int) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            r = _get_redis()
            lock_key = _lock_key(cid)
            acquired = r.set(lock_key, str(ad_index), nx=True, ex=LOCK_TTL_SECONDS)
            if not acquired:
                raise CampaignStoreError("campaign_generation_in_progress")
            raw = _load_raw(cid)
            if raw is None:
                r.delete(lock_key)
                raise CampaignStoreError("campaign_not_found")
            session = _session_from_raw(cid, raw)
            _validate_next_index(session, ad_index)
            raw["generatingIndex"] = ad_index
            _save_raw(cid, raw)
            return session
        except CampaignStoreError:
            raise
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_LOCK_ERR campaignId=%s err=%s", cid, exc)
            raise CampaignStoreError("campaign_store_error", str(exc)) from exc

    lock = _memory_campaign_lock(cid)
    if not lock.acquire(blocking=False):
        raise CampaignStoreError("campaign_generation_in_progress")
    try:
        raw = _load_raw(cid)
        if raw is None:
            raise CampaignStoreError("campaign_not_found")
        session = _session_from_raw(cid, raw)
        _validate_next_index(session, ad_index)
        raw["generatingIndex"] = ad_index
        _save_raw(cid, raw)
        return session
    except Exception:
        lock.release()
        raise


def release_generation_lock(campaign_id: str) -> None:
    cid = (campaign_id or "").strip()
    if _redis_configured():
        try:
            r = _get_redis()
            r.delete(_lock_key(cid))
            raw = _load_raw(cid)
            if raw is not None:
                raw["generatingIndex"] = None
                _save_raw(cid, raw)
        except Exception as exc:
            logger.error("BUILDER1_CAMPAIGN_UNLOCK_ERR campaignId=%s err=%s", cid, exc)
        return
    raw = _load_raw(cid)
    if raw is not None:
        raw["generatingIndex"] = None
        _save_raw(cid, raw)
    try:
        _memory_campaign_lock(cid).release()
    except RuntimeError:
        pass


def mark_ad_generated(campaign_id: str, ad_index: int) -> Builder1CampaignSession:
    cid = (campaign_id or "").strip()
    raw = _load_raw(cid)
    if raw is None:
        raise CampaignStoreError("campaign_not_found")
    session = _session_from_raw(cid, raw)
    if ad_index in session.generated_indexes:
        raise CampaignStoreError("campaign_index_conflict")
    generated = sorted(set(session.generated_indexes + [ad_index]))
    next_idx: Optional[int] = ad_index + 1 if ad_index < session.ad_count else None
    complete = len(generated) >= session.ad_count
    raw["generatedIndexes"] = generated
    raw["nextAdIndex"] = None if complete else next_idx
    raw["complete"] = complete
    raw["generatingIndex"] = None
    _save_raw(cid, raw)
    release_generation_lock(cid)
    logger.info(
        "BUILDER1_AD_GENERATED campaignId=%s adIndex=%s generatedCount=%s complete=%s",
        cid,
        ad_index,
        len(generated),
        complete,
    )
    if complete:
        logger.info("BUILDER1_CAMPAIGN_COMPLETE campaignId=%s", cid)
    return _session_from_raw(cid, raw)


def _validate_next_index(session: Builder1CampaignSession, expected_index: int) -> None:
    if session.complete:
        raise CampaignStoreError("campaign_complete")
    if expected_index in session.generated_indexes:
        raise CampaignStoreError("campaign_index_conflict")
    if session.next_ad_index != expected_index:
        raise CampaignStoreError("campaign_index_conflict")


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
        _memory_campaign_locks.clear()
