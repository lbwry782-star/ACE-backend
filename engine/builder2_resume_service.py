"""
Builder2 durable resume service — idempotent resume requests and status enrichment.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from engine.builder2_execution_lease import execution_lease_public_fields, has_active_lease
from engine.builder2_job_ownership import (
    is_historical_job_without_ownership,
    owner_context_present_in_job,
    public_owner_fields,
    verify_owner_context,
)
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION, normalize_builder2_stage
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_config import resolve_builder2_tournament_enabled
from engine.builder2_tournament_recovery import is_job_queued, mark_job_queued, register_recoverable_job
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import QUEUE_KEY, get_redis, job_key, video_job_get_raw, video_job_touch_progress

_ESTIMATED_TOTAL_SECONDS = 1200


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_from_epoch(raw_value: Any) -> Optional[str]:
    token = _clean(raw_value)
    if not token:
        return None
    if token.isdigit():
        try:
            return datetime.fromtimestamp(int(token), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return token
    return token


def _elapsed_seconds(job_hash: Mapping[str, Any]) -> int:
    started = _clean(job_hash.get("progressStartedAt") or job_hash.get("progress_started_at"))
    if started:
        try:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            return max(0, int((datetime.now(timezone.utc) - start_dt.astimezone(timezone.utc)).total_seconds()))
        except ValueError:
            pass
    raw = _clean(job_hash.get("enqueued_ts"))
    if raw.isdigit():
        return max(0, int(time.time()) - int(raw))
    return 0


def build_builder2_status_payload(
    job_id: str,
    job_hash: Optional[Mapping[str, Any]] = None,
    *,
    tournament_state: Optional[Dict[str, Any]] = None,
    request: Any = None,
) -> Dict[str, Any]:
    jid = _clean(job_id)
    raw = dict(job_hash) if isinstance(job_hash, Mapping) else (video_job_get_raw(jid) or {})
    tournament = tournament_state if tournament_state is not None else load_tournament_state(jid)
    resolver = resolve_builder2_resume_stage(raw, tournament)

    status = _clean(raw.get("status")) or "running"
    if status == "running" and (has_active_lease(jid) or is_job_queued(jid)):
        status = "processing" if has_active_lease(jid) else "queued"
    if resolver.get("jobAlreadyCompleted"):
        status = "done"

    media = (tournament or {}).get("mediaResume") if isinstance(tournament, dict) else {}
    media = media if isinstance(media, dict) else {}
    progress_stage = normalize_builder2_stage(
        media.get("progressStage")
        or raw.get("progressStage")
        or raw.get("progress_stage")
        or resolver.get("resumeFromStage")
        or "queued"
    )

    elapsed = _elapsed_seconds(raw)
    estimated_remaining = max(0, _ESTIMATED_TOTAL_SECONDS - elapsed)

    payload: Dict[str, Any] = {
        "jobId": jid,
        "status": status,
        "progressStage": progress_stage,
        "progressStartedAt": raw.get("progressStartedAt") or _timestamp_from_epoch(raw.get("enqueued_ts")),
        "mediaResumeStartedAt": media.get("mediaResumeStartedAt"),
        "lastResumeRequestedAt": raw.get("lastResumeRequestedAt"),
        "currentStageStartedAt": raw.get("currentStageStartedAt") or media.get("currentStageStartedAt"),
        "estimatedTotalSeconds": _ESTIMATED_TOTAL_SECONDS,
        "elapsedSeconds": elapsed,
        "estimatedRemainingSeconds": estimated_remaining,
        "canResume": bool(resolver.get("canResume")),
        "resumeFromStage": resolver.get("resumeFromStage"),
        "resumeAlreadyInProgress": bool(has_active_lease(jid) or is_job_queued(jid)),
        "advertisingClosureStatus": (tournament or {}).get("advertisingClosureStatus") if isinstance(tournament, dict) else None,
        "videoUrl": _clean(raw.get("video_url") or raw.get("videoUrl")) or None,
        "marketingText": _clean(raw.get("marketing_text") or raw.get("marketingText"))
        or (_clean(media.get("marketingText")) if isinstance(media, dict) else "")
        or None,
        "finalVideoUrl": _clean(media.get("finalPublicUrl")) or None,
        "finalVideoWithClosureUrl": _clean(media.get("finalVideoWithClosureUrl")) or None,
        "failureStage": _clean(raw.get("failureStage")) or None,
        "failureReason": _clean(raw.get("failureReason") or raw.get("error")) or None,
        "builder2ResumeContractVersion": raw.get("builder2ResumeContractVersion") or BUILDER2_RESUME_CONTRACT_VERSION,
        "completedStages": resolver.get("completedStages") or [],
        "reusableArtifacts": resolver.get("reusableArtifacts") or [],
        "jobAlreadyCompleted": bool(resolver.get("jobAlreadyCompleted")),
        "blockedReason": resolver.get("blockedReason"),
    }
    payload.update(public_owner_fields(raw))
    payload.update(execution_lease_public_fields(jid))
    if request is not None:
        ok, reason = verify_owner_context(raw, request)
        if not ok and not is_historical_job_without_ownership(raw):
            payload["ownershipVerified"] = False
            payload["ownershipError"] = reason
        elif not ok:
            payload["ownershipVerified"] = False
            payload["ownershipError"] = reason
        else:
            payload["ownershipVerified"] = True
    return payload


def request_builder2_resume(
    job_id: str,
    *,
    request: Any = None,
    allow_historical_admin: bool = False,
) -> Dict[str, Any]:
    jid = _clean(job_id)
    raw = video_job_get_raw(jid)
    if not raw:
        return {"ok": False, "error": "not_found", "jobId": jid}

    from engine.builder2_job_cancellation import is_builder2_job_cancelled

    if is_builder2_job_cancelled(jid):
        return {
            "ok": False,
            "error": "builder2_job_cancelled",
            "jobId": jid,
            "canResume": False,
            "status": "cancelled",
        }

    if request is not None:
        ok, reason = verify_owner_context(raw, request, allow_historical_admin=allow_historical_admin)
        if not ok:
            return {
                "ok": False,
                "error": reason or "ownership_required",
                "jobId": jid,
                **public_owner_fields(raw),
            }

    tournament = load_tournament_state(jid)
    resolver = resolve_builder2_resume_stage(raw, tournament)

    if resolver.get("jobAlreadyCompleted"):
        video_url = _clean(raw.get("video_url"))
        media = (tournament or {}).get("mediaResume") if isinstance(tournament, dict) else {}
        if isinstance(media, dict) and not video_url:
            video_url = _clean(media.get("finalPublicUrl"))
        return {
            "ok": True,
            "jobId": jid,
            "status": "done",
            "videoUrl": video_url or None,
            "mediaReused": True,
            **public_owner_fields(raw),
        }

    if not resolver.get("canResume"):
        return {
            "ok": False,
            "error": resolver.get("blockedReason") or "not_resumable",
            "jobId": jid,
            "canResume": False,
            "resumeFromStage": resolver.get("resumeFromStage"),
            **public_owner_fields(raw),
        }

    if has_active_lease(jid) or is_job_queued(jid):
        payload = build_builder2_status_payload(jid, raw, tournament_state=tournament)
        payload["ok"] = True
        payload["status"] = "processing" if has_active_lease(jid) else "queued"
        payload["resumeAlreadyInProgress"] = True
        return payload

    if not resolve_builder2_tournament_enabled():
        return {"ok": False, "error": "builder2_disabled", "jobId": jid}

    if not mark_job_queued(jid):
        payload = build_builder2_status_payload(jid, raw, tournament_state=tournament)
        payload["ok"] = True
        payload["resumeAlreadyInProgress"] = True
        return payload

    now_iso = _utc_now_iso()
    mapping = {
        "lastResumeRequestedAt": now_iso,
        "currentStageStartedAt": now_iso,
        "canResume": "1",
        "resumeFromStage": _clean(resolver.get("resumeFromStage")),
    }
    if not _clean(raw.get("progressStartedAt")):
        mapping["progressStartedAt"] = now_iso

    from engine.builder2_tournament_recovery import _use_memory_recovery, set_memory_job_hash

    if _use_memory_recovery:
        updated = dict(raw)
        updated.update({k: str(v) for k, v in mapping.items()})
        set_memory_job_hash(jid, updated)
    else:
        r = get_redis()
        r.hset(job_key(jid), mapping=mapping)
        r.lpush(QUEUE_KEY, jid)
    register_recoverable_job(jid)
    try:
        video_job_touch_progress(jid)
    except Exception:
        pass

    return {
        "ok": True,
        "jobId": jid,
        "status": "queued",
        "resumeRequested": True,
        "resumeFromStage": resolver.get("resumeFromStage"),
        **public_owner_fields(raw),
    }
