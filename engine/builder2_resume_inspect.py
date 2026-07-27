"""
Builder2 durable resume inspector — read-only zero-paid-call diagnostics.

Run:
  BUILDER2_RESUME_INSPECT_JOB_ID=<jobId> python -m engine.builder2_resume_inspect
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from engine.builder2_execution_lease import get_execution_lease_status
from engine.builder2_job_ownership import owner_context_present_in_job
from engine.builder2_resume_contract import _final_video_url, _media_bucket, _runway_bucket
from engine.builder2_resume_resolver import resolve_builder2_resume_stage
from engine.builder2_tournament_recovery import is_job_queued
from engine.builder2_tournament_store import load_tournament_state
from engine.video_jobs_redis import job_key, redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

DEFAULT_INSPECT_JOB_ID = "5a3157a3-532f-44ef-86db-c777cff54d38"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def inspect_builder2_resume_job(
    job_id: str,
    *,
    raw_job_reader: Optional[Callable[[str], Optional[Dict[str, str]]]] = None,
    tournament_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    read_raw = raw_job_reader or video_job_get_raw
    load_tournament = tournament_loader or load_tournament_state
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "jobId": jid or None,
        "jobExists": False,
        "tournamentExists": False,
        "ownerContextPresent": False,
        "jobStatus": None,
        "progressStage": None,
        "completedStageNames": [],
        "resolvedResumeStage": None,
        "executionLeaseStatus": "none",
        "queueMarkerStatus": "absent",
        "startImageReusable": False,
        "runwayTaskReusable": False,
        "finalVideoAvailable": False,
        "canResume": False,
        "jobAlreadyCompleted": False,
        "consistencyFailures": [],
        "resumeRequired": False,
        "reusableArtifacts": [],
        "redisMutations": 0,
        "openAICalls": 0,
        "runwayCalls": 0,
        "imageCalls": 0,
        "ffmpegCalls": 0,
        "ok": False,
    }

    if not jid:
        report["failureReason"] = "builder2_resume_inspect_job_id_missing"
        return report
    if not redis_configured():
        report["failureReason"] = "builder2_resume_inspect_redis_unconfigured"
        return report

    job_raw = read_raw(jid)
    report["jobExists"] = bool(job_raw)
    if not job_raw:
        report["failureReason"] = "job_not_found"
        return report

    tournament = load_tournament(jid)
    report["tournamentExists"] = isinstance(tournament, dict) and bool(tournament)
    report["ownerContextPresent"] = owner_context_present_in_job(job_raw)
    report["jobStatus"] = _clean(job_raw.get("status")) or None
    report["progressStage"] = _clean(job_raw.get("progressStage") or job_raw.get("progress_stage")) or None

    resolver = resolve_builder2_resume_stage(job_raw, tournament)
    report["completedStageNames"] = resolver.get("completedStages") or []
    report["resolvedResumeStage"] = resolver.get("resumeFromStage")
    report["executionLeaseStatus"] = get_execution_lease_status(jid)
    report["queueMarkerStatus"] = "present" if is_job_queued(jid) else "absent"
    report["consistencyFailures"] = resolver.get("consistencyFailures") or []
    report["canResume"] = bool(resolver.get("canResume"))
    report["jobAlreadyCompleted"] = bool(resolver.get("jobAlreadyCompleted"))
    report["resumeRequired"] = bool(resolver.get("resumeRequired"))
    report["reusableArtifacts"] = resolver.get("reusableArtifacts") or []

    media = _media_bucket(tournament)
    runway = _runway_bucket(tournament)
    start_image = _clean(media.get("startImageArtifact") or runway.get("startImageDataUri"))
    report["startImageReusable"] = bool(start_image and _clean(media.get("startImageStatus")) == "completed")
    report["runwayTaskReusable"] = bool(_clean(media.get("runwayTaskId") or runway.get("taskId")))
    report["finalVideoAvailable"] = bool(_final_video_url(job_raw, tournament))
    report["ok"] = True
    return report


def print_builder2_resume_inspect_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_RESUME_INSPECT_JOB_ID", DEFAULT_INSPECT_JOB_ID)
    logger.info("BUILDER2_RESUME_INSPECT_START jobId=%s", job_id)
    report = inspect_builder2_resume_job(job_id)
    print_builder2_resume_inspect_report(report)
    logger.info(
        "BUILDER2_RESUME_INSPECT_DONE jobId=%s ok=%s canResume=%s stage=%s",
        job_id,
        report.get("ok"),
        report.get("canResume"),
        report.get("resolvedResumeStage"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
