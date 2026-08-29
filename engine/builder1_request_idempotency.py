"""
Builder1 request idempotency — owner-scoped, Redis-atomic in production.
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
_IDEMPOTENCY_KEY_PREFIX = "builder1:idempotency:"
_IDEMPOTENCY_TTL_SECONDS = 24 * 3600
SUBMISSION_CLAIM_LEASE_SECONDS = int(
    (os.environ.get("BUILDER1_SUBMISSION_CLAIM_LEASE_SECONDS") or "120").strip() or "120"
)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

OPERATION_INITIAL_GENERATE = "initial_generate"
OPERATION_GENERATE_NEXT = "generate_next"
OPERATION_RETRY_IMAGE = "retry_image"
OPERATION_REPAIR_PHYSICAL = "repair_physical"

STATE_RESERVED = "reserved"
STATE_JOB_CREATED = "job_created"
STATE_SUBMISSION_CLAIMED = "submission_claimed"
STATE_EXECUTOR_ENQUEUED = "executor_enqueued"
STATE_WORKER_STARTED = "worker_started"
STATE_SUBMITTED = "submitted"

_BLOCKING_PAID_STAGE_STATUSES = frozenset({"submitted", "in_flight", "outcome_unknown"})

BeginKind = Literal["new", "replay", "conflict", "recover"]

_memory_lock = threading.Lock()
_memory_idempotency: Dict[str, Dict[str, Any]] = {}

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

_CLAIM_SUBMISSION_LEASE_LUA = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local response_json = ARGV[2]
local lease_seconds = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local claim_token = ARGV[5]
local raw = redis.call('GET', key)
if not raw then
  return cjson.encode({ok=false, reason='missing'})
end
local data = cjson.decode(raw)
if data.workerStartedAt then
  return cjson.encode({ok=false, reason='worker_started', record=raw})
end
if data.submissionClaimedAt then
  local age = now - tonumber(data.submissionClaimedAt)
  if age < lease_seconds then
    return cjson.encode({ok=false, reason='live_claim', record=raw})
  end
end
data.submissionClaimedAt = now
data.submissionClaimToken = claim_token
data.state = 'submission_claimed'
data.workerSubmitted = false
if response_json ~= '' then
  data.response = cjson.decode(response_json)
end
local encoded = cjson.encode(data)
redis.call('SET', key, encoded, 'EX', ttl)
return cjson.encode({ok=true, record=encoded})
"""


def _redis_configured() -> bool:
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


def builder1_request_id_required() -> bool:
    from engine.builder1_production_config import builder1_production_mode_enabled

    if builder1_production_mode_enabled():
        return True
    raw = (os.environ.get("BUILDER1_REQUEST_ID_REQUIRED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _idempotency_key(*, owner_context_ref: str, operation: str, request_id: str) -> str:
    owner = _clean(owner_context_ref)
    op = _clean(operation)
    rid = _clean(request_id)
    return f"{_IDEMPOTENCY_KEY_PREFIX}{owner}:{op}:{rid}"


def _canonical_brand_guidelines(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k): raw[k] for k in sorted(raw.keys(), key=str)}
    return str(raw)


def fingerprint_initial_generate(payload: Mapping[str, Any], *, ad_count: int) -> str:
    from engine.builder1_job_ownership import _hash_token

    canonical = {
        "productName": _clean(payload.get("productName")),
        "productDescription": _clean(payload.get("productDescription")),
        "format": _clean(payload.get("format")) or "portrait",
        "adCount": int(ad_count),
        "brandGuidelines": _canonical_brand_guidelines(payload.get("brandGuidelines")),
    }
    return _hash_token(json.dumps(canonical, sort_keys=True, ensure_ascii=False))


def fingerprint_generate_next(*, campaign_id: str, expected_next_index: int) -> str:
    from engine.builder1_job_ownership import _hash_token

    canonical = {
        "campaignId": _clean(campaign_id),
        "expectedNextIndex": int(expected_next_index),
    }
    return _hash_token(json.dumps(canonical, sort_keys=True, ensure_ascii=False))


def fingerprint_retry_image(*, campaign_id: str, retry_ad_index: int) -> str:
    from engine.builder1_job_ownership import _hash_token

    canonical = {
        "campaignId": _clean(campaign_id),
        "retryAdIndex": int(retry_ad_index),
    }
    return _hash_token(json.dumps(canonical, sort_keys=True, ensure_ascii=False))


def fingerprint_repair_physical(*, campaign_id: str, retry_ad_index: int) -> str:
    from engine.builder1_job_ownership import _hash_token

    canonical = {
        "campaignId": _clean(campaign_id),
        "retryAdIndex": int(retry_ad_index),
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
            logger.error("BUILDER1_IDEMPOTENCY_LOAD_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder1_idempotency_unavailable") from exc
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
            logger.error("BUILDER1_IDEMPOTENCY_SAVE_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder1_idempotency_unavailable") from exc
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
            logger.error("BUILDER1_IDEMPOTENCY_RESERVE_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder1_idempotency_unavailable") from exc
    with _memory_lock:
        existing = _memory_idempotency.get(key)
        if existing:
            return dict(existing)
        _memory_idempotency[key] = dict(new_record)
        return dict(new_record)


def _job_record(job_id: str) -> Optional[Dict[str, Any]]:
    from engine.builder1_jobs_store import get_builder1_job

    jid = _clean(job_id)
    if not jid:
        return None
    return get_builder1_job(jid)


def job_blocks_automatic_worker_resubmit(job: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(job, dict):
        return False
    return _clean(job.get("lastPaidStageStatus")) in _BLOCKING_PAID_STAGE_STATUSES


def execution_is_proven(record: Optional[Dict[str, Any]], job_id: str) -> bool:
    if isinstance(record, dict) and record.get("workerStartedAt"):
        return True
    job = _job_record(job_id)
    if not job:
        return False
    if job.get("workerStartedAt"):
        return True
    status = _clean(job.get("status"))
    return status in {"done", "cancelled", "error"}


def submission_claim_is_live(record: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("workerStartedAt"):
        return True
    claimed_at = record.get("submissionClaimedAt")
    if claimed_at is None:
        return False
    try:
        age = time.time() - float(claimed_at)
    except (TypeError, ValueError):
        return False
    return age < SUBMISSION_CLAIM_LEASE_SECONDS


def should_replay_without_worker_resubmit(record: Optional[Dict[str, Any]], job_id: str) -> bool:
    if execution_is_proven(record, job_id):
        return True
    if submission_claim_is_live(record):
        return True
    job = _job_record(job_id)
    if job_blocks_automatic_worker_resubmit(job):
        return True
    return False


def should_recover_stale_submission(record: Optional[Dict[str, Any]], job_id: str) -> bool:
    if not isinstance(record, dict):
        return False
    if execution_is_proven(record, job_id):
        return False
    if submission_claim_is_live(record):
        return False
    job = _job_record(job_id)
    if job_blocks_automatic_worker_resubmit(job):
        return False
    return bool(record.get("submissionClaimedAt") or record.get("executorEnqueuedAt"))


def _begin_kind_for_record(record: Dict[str, Any], *, job_id: str) -> BeginKind:
    authoritative_job = _clean(record.get("jobId") or job_id)
    if should_replay_without_worker_resubmit(record, authoritative_job):
        return "replay"
    if should_recover_stale_submission(record, authoritative_job):
        return "recover"
    if _clean(record.get("jobId")) != _clean(job_id):
        return "recover"
    return "new"


@dataclass(frozen=True)
class IdempotencyBeginResult:
    kind: BeginKind
    record: Dict[str, Any]
    key: str


def begin_builder1_idempotent_request(
    *,
    operation: str,
    request_id: str,
    owner_context_ref: str,
    request_fingerprint: str,
    job_id: str,
    campaign_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> IdempotencyBeginResult:
    """
    Atomically reserve or load an idempotency record BEFORE paid work starts.
    Pre-assign jobId/campaignId so crash recovery never mints duplicate identities.
    """
    from engine.builder1_production_config import builder1_production_mode_enabled

    if builder1_production_mode_enabled() and not _redis_configured():
        raise RuntimeError("builder1_production_requires_redis")

    key = _idempotency_key(
        owner_context_ref=owner_context_ref,
        operation=operation,
        request_id=request_id,
    )
    now = time.time()
    reserved = {
        "requestId": _clean(request_id),
        "operation": _clean(operation),
        "ownerContextRef": _clean(owner_context_ref),
        "requestFingerprint": _clean(request_fingerprint),
        "state": STATE_RESERVED,
        "jobId": _clean(job_id),
        "campaignId": _clean(campaign_id),
        "createdAt": now,
        "workerSubmitted": False,
        "response": {},
    }
    if extra:
        reserved.update(dict(extra))

    stored = _reserve_or_get(key, reserved)
    if stored.get("requestFingerprint") != request_fingerprint:
        return IdempotencyBeginResult(kind="conflict", record=stored, key=key)

    kind = _begin_kind_for_record(stored, job_id=job_id)
    return IdempotencyBeginResult(kind=kind, record=stored, key=key)


def mark_idempotent_job_created(key: str, *, record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    current = dict(_load_record(key) or record or {})
    if _clean(current.get("state")) in {
        STATE_SUBMISSION_CLAIMED,
        STATE_EXECUTOR_ENQUEUED,
        STATE_WORKER_STARTED,
        STATE_SUBMITTED,
    }:
        return current
    if _clean(current.get("state")) != STATE_RESERVED:
        return current
    current["state"] = STATE_JOB_CREATED
    _save_record(key, current)
    return current


def claim_submission_lease(
    key: str,
    *,
    response: Optional[Dict[str, Any]] = None,
    claim_token: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Claim (or stale-reclaim) a submission lease. Does NOT prove worker execution.
    Returns (claimed, record).
    """
    token = _clean(claim_token) or uuid.uuid4().hex
    response_json = json.dumps(response or {}, ensure_ascii=False)
    now = time.time()
    if _redis_configured():
        try:
            r = _get_redis()
            script = r.register_script(_CLAIM_SUBMISSION_LEASE_LUA)
            raw = script(
                keys=[key],
                args=[
                    str(_IDEMPOTENCY_TTL_SECONDS),
                    response_json,
                    str(SUBMISSION_CLAIM_LEASE_SECONDS),
                    str(now),
                    token,
                ],
            )
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            parsed = json.loads(raw)
            if parsed.get("ok"):
                rec = json.loads(parsed["record"])
                return True, rec
            rec = json.loads(parsed.get("record") or "{}")
            return False, rec if isinstance(rec, dict) else (_load_record(key) or {})
        except Exception as exc:
            logger.error("BUILDER1_IDEMPOTENCY_CLAIM_ERR key=%s err=%s", key, exc)
            raise RuntimeError("builder1_idempotency_unavailable") from exc

    with _memory_lock:
        rec = _memory_idempotency.get(key)
        if not rec:
            return False, {}
        rec = dict(rec)
        if rec.get("workerStartedAt"):
            return False, rec
        claimed_at = rec.get("submissionClaimedAt")
        if claimed_at is not None:
            try:
                age = now - float(claimed_at)
            except (TypeError, ValueError):
                age = SUBMISSION_CLAIM_LEASE_SECONDS
            if age < SUBMISSION_CLAIM_LEASE_SECONDS:
                return False, rec
        rec["submissionClaimedAt"] = now
        rec["submissionClaimToken"] = token
        rec["state"] = STATE_SUBMISSION_CLAIMED
        rec["workerSubmitted"] = False
        if response is not None:
            rec["response"] = dict(response)
        _memory_idempotency[key] = rec
        return True, dict(rec)


def mark_executor_enqueued(key: str, *, record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    updated = dict(_load_record(key) or record or {})
    updated["executorEnqueuedAt"] = time.time()
    updated["state"] = STATE_EXECUTOR_ENQUEUED
    _save_record(key, updated)
    return updated


def mark_builder1_worker_started(job_id: str, *, idempotency_key: str = "") -> None:
    """Durable proof that the in-process worker thread actually started."""
    now = time.time()
    from engine.builder1_jobs_store import update_builder1_job

    update_builder1_job(
        job_id,
        workerStartedAt=now,
        lastHeartbeatAt=now,
    )
    key = _clean(idempotency_key)
    if not key:
        job = _job_record(job_id)
        key = _clean((job or {}).get("idempotencyKey"))
    if not key:
        return
    record = _load_record(key)
    if not record:
        return
    updated = dict(record)
    updated["workerStartedAt"] = now
    updated["state"] = STATE_WORKER_STARTED
    updated["workerSubmitted"] = True
    _save_record(key, updated)
    logger.info(
        "BUILDER1_WORKER_STARTED jobId=%s idempotencyKey=%s",
        job_id,
        key,
    )


def finalize_idempotent_worker_dispatch(
    *,
    idem_key: Optional[str],
    idem_record: Optional[Dict[str, Any]],
    job_id: str,
    response_body: Dict[str, Any],
    submit_fn: Any,
) -> Tuple[Dict[str, Any], bool]:
    """
    Resolve submission lease, enqueue worker once, return (response_dict, is_replay).
    Order: claim lease → executor.submit → mark_executor_enqueued.
    workerStartedAt is recorded separately when the worker thread starts.
    """
    if not idem_key or idem_record is None:
        submit_fn()
        return dict(response_body), False

    mark_idempotent_job_created(idem_key)
    record = _load_record(idem_key) or dict(idem_record)
    jid = _clean(job_id or response_body.get("jobId"))

    if should_replay_without_worker_resubmit(record, jid):
        return replay_response_from_record(record), True

    claim_token = uuid.uuid4().hex
    claimed, record = claim_submission_lease(
        idem_key,
        response=response_body,
        claim_token=claim_token,
    )
    if not claimed:
        return replay_response_from_record(record), True

    from engine.builder1_jobs_store import update_builder1_job

    update_builder1_job(
        jid,
        idempotencyKey=idem_key,
        submissionClaimToken=claim_token,
    )
    submit_fn()
    mark_executor_enqueued(idem_key)
    return dict(response_body), False


def claim_idempotent_worker_submission(
    key: str,
    *,
    response: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Backward-compatible alias — prefer claim_submission_lease + finalize dispatch."""
    return claim_submission_lease(key, response=response)


def store_idempotent_response(key: str, *, record: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(record)
    updated["response"] = dict(response)
    updated["state"] = STATE_SUBMITTED
    _save_record(key, updated)
    return updated


def replay_response_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    response = dict(record.get("response") or {})
    if not response.get("jobId"):
        response["jobId"] = record.get("jobId")
    if not response.get("campaignId"):
        response["campaignId"] = record.get("campaignId")
    if record.get("pollUrl") and not response.get("pollUrl"):
        response["pollUrl"] = record.get("pollUrl")
    response["idempotentReplay"] = True
    response["ok"] = response.get("ok", True)
    return response


def clear_memory_idempotency_for_tests() -> None:
    with _memory_lock:
        _memory_idempotency.clear()
