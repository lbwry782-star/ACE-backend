"""
Builder2 operator-only resume for frontend-cancelled jobs.

Explicit CLI entry point only — not exposed to Frontend/API automatic recovery.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.builder2_job_cancellation import (
    CANCEL_REASON_FRONTEND_REFRESH,
    is_builder2_job_cancelled,
    is_builder2_job_hash,
)
from engine.builder2_resume_contract import (
    CANONICAL_BUILDER2_STAGES,
    _final_video_url,
    _media_bucket,
    normalize_builder2_stage,
)
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_store import load_tournament_state, save_tournament_state
from engine.video_jobs_redis import get_redis, job_key, redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

ENV_OPERATOR_JOB_ID = "BUILDER2_OPERATOR_RESUME_CANCELLED_JOB_ID"
ENV_OPERATOR_DRY_RUN = "BUILDER2_OPERATOR_RESUME_CANCELLED_DRY_RUN"

CLASSIFICATION_REASONING_INCOMPLETE = "reasoning_incomplete"
CLASSIFICATION_WINNER_READY_MEDIA_NOT_STARTED = "winner_ready_media_not_started"
CLASSIFICATION_MEDIA_PARTIALLY_COMPLETED = "media_partially_completed"
CLASSIFICATION_RUNWAY_ARTIFACT_PRESENT = "runway_artifact_present"
CLASSIFICATION_LYRIA_PARTIALLY_COMPLETED = "lyria_partially_completed"
CLASSIFICATION_FINAL_OUTPUT_AVAILABLE = "final_output_available"
CLASSIFICATION_UNSAFE = "unsafe"

_REASONING_RESUME_STAGES = frozenset(
    {
        "strategy",
        "creator_generation",
        "creator_complete",
        "judge_generation",
        "judge_complete",
        "winner_selection",
        "winner_development",
        "advertising_closure",
    }
)

_MEDIA_RESUME_STAGES = frozenset(
    stage
    for stage in CANONICAL_BUILDER2_STAGES
    if stage not in _REASONING_RESUME_STAGES and stage not in {"queued", "completed"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truthy(raw: Any) -> bool:
    return _clean(raw).lower() in {"1", "true", "yes"}


def _shadow_resumable_job_hash(job_raw: Dict[str, Any]) -> Dict[str, Any]:
    shadow = dict(job_raw)
    shadow.pop("cancelRequested", None)
    status = _clean(shadow.get("status"))
    if status == "cancelled":
        shadow["status"] = "running"
    shadow["canResume"] = "1"
    progress = _clean(shadow.get("progressStage") or shadow.get("progress_stage"))
    if progress == "cancelled":
        shadow.pop("progressStage", None)
        shadow.pop("progress_stage", None)
    return shadow


def detect_ambiguous_paid_call_outcome(
    tournament_state: Optional[Dict[str, Any]],
    *,
    job_raw: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    media = _media_bucket(tournament_state)
    music_status = _clean(media.get("musicGenerationStatus")).lower()
    if music_status in {"paid_call_outcome_unknown", "generating"}:
        return "builder2_lyria_paid_call_outcome_unknown"

    start_status = _clean(media.get("startImageStatus")).lower()
    if start_status == "generating":
        return "builder2_start_image_paid_call_outcome_unknown"

    runway_task = _clean(media.get("runwayTaskId"))
    runway_output = _clean(media.get("runwayOutputUrl") or media.get("runwayVideoUrl"))
    runway_status = _clean(media.get("runwayStatus")).lower()
    if runway_task and not runway_output:
        if runway_status in {"", "running", "pending", "in_progress", "processing", "queued", "submitted"}:
            return "builder2_runway_paid_call_outcome_unknown"

    if _truthy((job_raw or {}).get("runwaySubmissionInFlight")):
        return "builder2_runway_paid_call_outcome_unknown"
    return None


def classify_cancelled_builder2_job(
    job_id: str,
    *,
    job_raw: Optional[Dict[str, Any]] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid or None,
        "ok": False,
        "safe": False,
        "classification": None,
        "phase": None,
        "resumeType": None,
        "resumeFromStage": None,
        "jobAlreadyCompleted": False,
        "cancelled": False,
        "paidCallOutcomeUnknown": None,
        "reusableArtifacts": [],
        "consistencyFailures": [],
    }
    if not jid:
        report["failureReason"] = "builder2_operator_resume_cancelled_job_id_missing"
        return report

    raw = dict(job_raw) if isinstance(job_raw, dict) else (video_job_get_raw(jid) or {})
    if not raw:
        report["failureReason"] = "job_not_found"
        return report
    if not is_builder2_job_hash(raw):
        report["failureReason"] = "not_builder2_job"
        return report

    tournament = tournament_state if tournament_state is not None else load_tournament_state(jid)
    report["cancelled"] = is_builder2_job_cancelled(jid)
    if not report["cancelled"] and not _truthy(raw.get("operatorResumedFromCancelled")):
        report["failureReason"] = "builder2_operator_resume_not_cancelled"
        return report

    ambiguous = detect_ambiguous_paid_call_outcome(tournament, job_raw=raw)
    if ambiguous:
        report["paidCallOutcomeUnknown"] = ambiguous
        report["classification"] = CLASSIFICATION_UNSAFE
        report["phase"] = "unsafe"
        report["failureReason"] = ambiguous
        report["ok"] = True
        return report

    final_url = _final_video_url(raw, tournament)
    media = _media_bucket(tournament)
    if final_url or (
        _clean(media.get("finalPublicUrl"))
        and _clean(media.get("mediaResumeStatus")) == "completed"
    ):
        report["classification"] = CLASSIFICATION_FINAL_OUTPUT_AVAILABLE
        report["phase"] = "completed"
        report["resumeType"] = "none"
        report["resumeFromStage"] = "completed"
        report["jobAlreadyCompleted"] = True
        report["safe"] = True
        report["ok"] = True
        return report

    shadow = _shadow_resumable_job_hash(raw)
    resolver = resolve_builder2_resume_stage(shadow, tournament, read_only=True)
    resume_stage = normalize_builder2_stage(_clean(resolver.get("resumeFromStage")) or "queued")
    report["resumeFromStage"] = resume_stage
    report["reusableArtifacts"] = list(resolver.get("reusableArtifacts") or [])
    report["consistencyFailures"] = list(resolver.get("consistencyFailures") or [])
    report["jobAlreadyCompleted"] = bool(resolver.get("jobAlreadyCompleted"))

    if report["consistencyFailures"]:
        report["classification"] = CLASSIFICATION_UNSAFE
        report["phase"] = "unsafe"
        report["failureReason"] = report["consistencyFailures"][0]
        report["ok"] = True
        return report

    if report["jobAlreadyCompleted"] or resume_stage == "completed" or _final_video_url(raw, tournament):
        report["classification"] = CLASSIFICATION_FINAL_OUTPUT_AVAILABLE
        report["phase"] = "completed"
        report["resumeType"] = "none"
        report["safe"] = True
        report["ok"] = True
        return report

    media = _media_bucket(tournament)
    music_status = _clean(media.get("musicGenerationStatus")).lower()
    runway_task = _clean(media.get("runwayTaskId"))
    runway_output = _clean(media.get("runwayOutputUrl") or media.get("runwayVideoUrl"))

    if resume_stage in _REASONING_RESUME_STAGES:
        report["classification"] = CLASSIFICATION_REASONING_INCOMPLETE
        report["phase"] = "reasoning"
        report["resumeType"] = "reasoning"
    elif resume_stage == "media_prerequisite_validation" and not _truthy((tournament or {}).get("mediaContinuationRequired")):
        report["classification"] = CLASSIFICATION_WINNER_READY_MEDIA_NOT_STARTED
        report["phase"] = "media"
        report["resumeType"] = "media"
    elif resume_stage == "generating_music" or music_status in {"failed", "succeeded"}:
        report["classification"] = CLASSIFICATION_LYRIA_PARTIALLY_COMPLETED
        report["phase"] = "media"
        report["resumeType"] = "media"
    elif runway_task or runway_output or resume_stage in {"runway_submission", "runway_waiting", "runway_complete", "video_download"}:
        report["classification"] = CLASSIFICATION_RUNWAY_ARTIFACT_PRESENT
        report["phase"] = "media"
        report["resumeType"] = "media"
    elif resume_stage in _MEDIA_RESUME_STAGES:
        report["classification"] = CLASSIFICATION_MEDIA_PARTIALLY_COMPLETED
        report["phase"] = "media"
        report["resumeType"] = "media"
    else:
        report["classification"] = CLASSIFICATION_MEDIA_PARTIALLY_COMPLETED
        report["phase"] = "media"
        report["resumeType"] = "media"

    report["safe"] = True
    report["ok"] = True
    return report


def reactivate_cancelled_builder2_job(
    job_id: str,
    *,
    classification: Dict[str, Any],
    job_raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    jid = _clean(job_id)
    result: Dict[str, Any] = {
        "jobId": jid,
        "ok": False,
        "reactivated": False,
        "redisMutations": 0,
    }
    if not classification.get("safe"):
        result["failureReason"] = classification.get("failureReason") or "builder2_operator_resume_unsafe"
        return result

    raw = dict(job_raw) if isinstance(job_raw, dict) else (video_job_get_raw(jid) or {})
    if not raw:
        result["failureReason"] = "job_not_found"
        return result

    if not is_builder2_job_cancelled(jid) and _truthy(raw.get("operatorResumedFromCancelled")):
        result["ok"] = True
        result["alreadyReactivated"] = True
        return result

    now_iso = _utc_now_iso()
    now_epoch = str(int(time.time()))
    resume_stage = _clean(classification.get("resumeFromStage")) or "running"
    if resume_stage == "completed":
        resume_stage = "running"

    previous_cancel_reason = _clean(raw.get("cancelReason")) or CANCEL_REASON_FRONTEND_REFRESH
    mapping = {
        "cancelRequested": "0",
        "status": "done" if classification.get("jobAlreadyCompleted") else "running",
        "canResume": "0" if classification.get("jobAlreadyCompleted") else "1",
        "progressStage": "completed" if classification.get("jobAlreadyCompleted") else resume_stage,
        "last_progress_ts": now_epoch,
        "operatorResumedFromCancelled": "1",
        "operatorResumedAt": now_iso,
        "previousCancelReason": previous_cancel_reason,
        "error": "",
    }
    if _clean(raw.get("cancelledAt")):
        mapping["previousCancelledAt"] = _clean(raw.get("cancelledAt"))

    from engine.builder2_tournament_recovery import _use_memory_recovery, set_memory_job_hash

    if _use_memory_recovery:
        updated = dict(raw)
        updated.update({k: str(v) for k, v in mapping.items()})
        set_memory_job_hash(jid, updated)
    elif redis_configured():
        get_redis().hset(job_key(jid), mapping=mapping)
    else:
        result["failureReason"] = "builder2_operator_resume_redis_unconfigured"
        return result
    result["redisMutations"] = 1

    tournament = load_tournament_state(jid)
    if isinstance(tournament, dict):
        tournament["canResume"] = not bool(classification.get("jobAlreadyCompleted"))
        tournament["operatorResumedFromCancelled"] = True
        tournament["operatorResumedAt"] = now_iso
        tournament["previousCancelReason"] = previous_cancel_reason
        if _clean(raw.get("cancelReason")):
            tournament["previousCancelReason"] = _clean(raw.get("cancelReason"))
        if classification.get("jobAlreadyCompleted"):
            tournament["status"] = "completed"
            tournament["lastCompletedStep"] = "done"
        elif _clean(tournament.get("status")) == "cancelled":
            tournament["status"] = "running"
        save_tournament_state(jid, tournament)
        result["redisMutations"] = 2

    result["reactivated"] = True
    result["ok"] = True
    return result


def run_operator_resume_cancelled_job(
    job_id: str,
    *,
    dry_run: bool = False,
    execute_resume: bool = True,
) -> Dict[str, Any]:
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid or None,
        "ok": False,
        "dryRun": bool(dry_run),
        "classification": None,
        "reactivated": False,
        "resumeType": None,
        "resumeReport": None,
    }
    if not jid:
        report["failureReason"] = "builder2_operator_resume_cancelled_job_id_missing"
        return report

    logger.info("BUILDER2_OPERATOR_CANCELLED_RESUME_START jobId=%s dryRun=%s", jid, dry_run)
    classification = classify_cancelled_builder2_job(jid)
    report["classification"] = classification
    if not classification.get("ok"):
        report["failureReason"] = classification.get("failureReason") or "builder2_operator_resume_classify_failed"
        return report

    logger.info(
        "BUILDER2_OPERATOR_CANCELLED_RESUME_CLASSIFIED jobId=%s phase=%s safe=%s classification=%s resumeFromStage=%s",
        jid,
        classification.get("phase"),
        classification.get("safe"),
        classification.get("classification"),
        classification.get("resumeFromStage"),
    )

    if not classification.get("safe"):
        report["failureReason"] = classification.get("failureReason") or "builder2_operator_resume_unsafe"
        report["ok"] = False
        return report

    report["resumeType"] = classification.get("resumeType")
    if dry_run:
        report["ok"] = True
        return report

    if classification.get("jobAlreadyCompleted") or classification.get("resumeType") == "none":
        reactivate = reactivate_cancelled_builder2_job(jid, classification=classification)
        report["reactivated"] = bool(reactivate.get("reactivated"))
        report["ok"] = bool(reactivate.get("ok"))
        report["resumeReport"] = {
            "ok": True,
            "jobCompleted": True,
            "mediaReused": True,
            "paidCallsSkipped": True,
        }
        logger.info("BUILDER2_OPERATOR_CANCELLED_RESUME_DONE jobId=%s resumeType=none", jid)
        return report

    reactivate = reactivate_cancelled_builder2_job(jid, classification=classification)
    report["reactivated"] = bool(reactivate.get("reactivated"))
    if not reactivate.get("ok"):
        report["failureReason"] = reactivate.get("failureReason") or "builder2_operator_resume_reactivate_failed"
        return report

    logger.info("BUILDER2_OPERATOR_CANCELLED_RESUME_REACTIVATED jobId=%s", jid)

    if not execute_resume:
        report["ok"] = True
        return report

    resume_type = _clean(classification.get("resumeType"))
    resume_report: Dict[str, Any]
    if resume_type == "reasoning":
        from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume

        resume_report = run_controlled_complete_ad_reasoning_resume(
            job_id=jid,
            stop_before_media=True,
        )
    elif resume_type == "media":
        from engine.builder2_media_resume import run_one_media_resume

        resume_report = run_one_media_resume(job_id=jid)
    else:
        resume_report = {"ok": False, "failureReason": "builder2_operator_resume_unknown_resume_type"}

    report["resumeReport"] = resume_report
    report["ok"] = bool(resume_report.get("ok"))
    if not report["ok"]:
        report["failureReason"] = resume_report.get("failureReason") or "builder2_operator_resume_delegate_failed"

    logger.info(
        "BUILDER2_OPERATOR_CANCELLED_RESUME_DONE jobId=%s resumeType=%s ok=%s",
        jid,
        resume_type,
        report.get("ok"),
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get(ENV_OPERATOR_JOB_ID))
    dry_run = _truthy(os.environ.get(ENV_OPERATOR_DRY_RUN))
    if not job_id:
        payload = {"ok": False, "failureReason": "builder2_operator_resume_cancelled_job_id_missing"}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    report = run_operator_resume_cancelled_job(job_id, dry_run=dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
