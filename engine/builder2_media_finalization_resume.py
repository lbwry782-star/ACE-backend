"""
Builder2 media finalization recovery — completed-but-invalid finalization repair.

Run:
  BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID=<jobId> python -m engine.builder2_media_finalization_resume
  BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT=true ...  # read-only redis + local ffmpeg proof
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from engine.builder2_advertising_closure_pipeline import render_advertising_closure_for_state
from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    classify_url_route_family,
    render_builder2_advertising_closure_endcard,
)
from engine.builder2_execution_lease import acquire_job_lease, release_job_lease
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_media_finalization_contract import (
    backfill_legacy_headline_reference,
    closure_inclusive_artifact_valid,
    finalization_recovery_eligible,
    resolve_legacy_headline_artifact_url,
    resolve_raw_runway_artifact_url,
    validate_builder2_media_completion_contract,
)
from engine.builder2_media_finalization_guard import MediaFinalizationIsolationGuard
from engine.builder2_media_resume_config import build_media_resume_configuration
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import _read_raw, save_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get, video_job_get_raw, video_job_mark_done

logger = logging.getLogger(__name__)

_HTTP_DOWNLOAD_TIMEOUT = 180.0


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _truthy(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _initial_report(*, job_id: str, preflight: bool) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "ok": False,
        "preflight": preflight,
        "eligibleForFinalizationRecovery": False,
        "falseCompletionConfirmed": False,
        "legacyHeadlineArtifactIdentified": False,
        "legacyHeadlineArtifactRouteFamily": None,
        "headlineArtifactDownloadAccepted": False,
        "measuredHeadlineDurationSeconds": None,
        "closureRenderAttempted": False,
        "closureRenderAccepted": False,
        "measuredFinalDurationSeconds": None,
        "finalDurationAccepted": False,
        "exactProductAndSloganReused": False,
        "readyForFinalizationRecovery": False,
        "failureStage": None,
        "failureReason": None,
        "safeFfmpegReturnCode": None,
        "safeFfmpegStderrAvailable": False,
        "openAICalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
        "runwayPollingCalls": 0,
        "ffmpegCalls": 0,
        "publicationCalls": 0,
        "redisMutations": 0,
        "finalizationReused": False,
        "jobCompleted": False,
        "totalReasoningCalls": 0,
    }


def _download_to_path(url: str, path: Path) -> None:
    response = requests.get(url, timeout=_HTTP_DOWNLOAD_TIMEOUT, stream=True)
    response.raise_for_status()
    with open(path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                handle.write(chunk)


def _probe_duration(path: Path) -> float:
    from engine.builder2_closure_render import _ffprobe_duration_seconds, _FFPROBE_TIMEOUT

    return _ffprobe_duration_seconds(path, _FFPROBE_TIMEOUT)


def _closure_source_for_recovery(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
) -> tuple[str, str]:
    headline_required = headline_decision_requires_headline(get_normalized_headline_decision(plan))
    headline_url = backfill_legacy_headline_reference(state, job_video_url=job_video_url)
    if headline_required:
        if not headline_url or classify_url_route_family(headline_url) != "api/video-headline":
            raise Builder2TournamentError("builder2_media_finalization_headline_artifact_unrecognized")
        return headline_url, headline_url
    raw_url = resolve_raw_runway_artifact_url(state)
    if not raw_url:
        raise Builder2TournamentError("builder2_media_finalization_raw_runway_missing")
    return raw_url, ""


def run_finalization_preflight(
    *,
    job_id: str,
    state: Optional[Dict[str, Any]] = None,
    job_video_url: str = "",
) -> Dict[str, Any]:
    report = _initial_report(job_id=job_id, preflight=True)
    if state is None:
        raw = _read_raw(job_id)
        if raw is None:
            report["failureReason"] = "builder2_media_finalization_job_not_found"
            report["failureStage"] = "load"
            return report
        state = deepcopy(raw)
    plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
    if not job_video_url:
        job_raw = video_job_get_raw(job_id) or {}
        job_video_url = str(job_raw.get("video_url") or job_raw.get("videoUrl") or "")

    eligible, missing = finalization_recovery_eligible(state=state, plan=plan, job_video_url=job_video_url)
    report["eligibleForFinalizationRecovery"] = eligible
    report["falseCompletionConfirmed"] = "falseCompletionNotProven" not in missing
    if not eligible:
        report["failureReason"] = f"builder2_media_finalization_not_eligible:{','.join(missing)}"
        report["failureStage"] = "eligibility"
        return report

    closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
    try:
        source_url, headline_url = _closure_source_for_recovery(state=state, plan=plan, job_video_url=job_video_url)
    except Builder2TournamentError as exc:
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_media_finalization_source_missing")
        report["failureStage"] = "legacy_headline"
        return report

    report["legacyHeadlineArtifactIdentified"] = bool(headline_url)
    report["legacyHeadlineArtifactRouteFamily"] = classify_url_route_family(headline_url) if headline_url else None
    report["exactProductAndSloganReused"] = bool(
        str(closure.get("productNameText") or "").strip() and str(closure.get("sloganText") or "").strip()
    )

    job_data = video_job_get(job_id) if redis_configured() else None
    media_config = build_media_resume_configuration(
        job_id=job_id,
        job_data=job_data,
        tournament_state=state,
        start_image_required=False,
        ffmpeg_required=True,
    )

    tmp = Path(tempfile.mkdtemp(prefix="ace_finalization_preflight_"))
    try:
        source_path = tmp / "source.mp4"
        _download_to_path(source_url, source_path)
        report["headlineArtifactDownloadAccepted"] = True
        if headline_url:
            report["measuredHeadlineDurationSeconds"] = _probe_duration(source_path)

        output_path = tmp / "closure_out.mp4"
        render_result = render_builder2_advertising_closure_endcard(
            source_url,
            media_config.publicBaseUrl,
            product_name=str(closure.get("productNameText") or ""),
            slogan=str(closure.get("sloganText") or ""),
            language=str(closure.get("language") or "en"),
            duration_seconds=float(closure.get("durationSeconds") or 2.0),
            job_id=job_id,
            publish=False,
            output_path=output_path,
        )
        report["closureRenderAttempted"] = True
        report["closureRenderAccepted"] = True
        report["measuredFinalDurationSeconds"] = render_result.measured_duration_seconds
        report["finalDurationAccepted"] = True
        report["ffmpegCalls"] = 1
        report["readyForFinalizationRecovery"] = True
        report["ok"] = True
    except Builder2ClosureRenderError as exc:
        report["closureRenderAttempted"] = True
        report["failureStage"] = exc.stage
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_closure_ffmpeg_failed")
        report["safeFfmpegReturnCode"] = exc.return_code
        report["safeFfmpegStderrAvailable"] = bool(exc.stderr_tail)
        report["ffmpegCalls"] = 1
    except Exception as exc:
        report["failureStage"] = "preflight"
        report["failureReason"] = type(exc).__name__
    finally:
        try:
            for path in tmp.iterdir():
                path.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass
    return report


def run_one_media_finalization_resume(
    *,
    job_id: str,
    preflight: bool = False,
    acquire_lease: bool = True,
) -> Dict[str, Any]:
    if preflight:
        return run_finalization_preflight(job_id=job_id)

    report = _initial_report(job_id=job_id, preflight=False)
    if not redis_configured():
        report["failureReason"] = "builder2_media_finalization_redis_unconfigured"
        report["failureStage"] = "configuration"
        return report

    worker_token = f"finalization-{uuid.uuid4().hex}"
    lease_acquired = False
    MediaFinalizationIsolationGuard.begin()
    state: Dict[str, Any] = {}
    try:
        raw = _read_raw(job_id)
        if raw is None:
            report["failureReason"] = "builder2_media_finalization_job_not_found"
            report["failureStage"] = "load"
            return report
        state = deepcopy(raw)
        plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
        job_raw = video_job_get_raw(job_id) or {}
        job_video_url = str(job_raw.get("video_url") or job_raw.get("videoUrl") or "")

        eligible, missing = finalization_recovery_eligible(state=state, plan=plan, job_video_url=job_video_url)
        report["eligibleForFinalizationRecovery"] = eligible
        report["falseCompletionConfirmed"] = "falseCompletionNotProven" not in missing
        if not eligible and "validClosureAlreadyPresent" in missing:
            contract_ok, _, _ = validate_builder2_media_completion_contract(
                state=state,
                plan=plan,
                job_video_url=job_video_url,
            )
            if contract_ok:
                report["finalizationReused"] = True
                report["jobCompleted"] = True
                report["ok"] = True
                return report
        if not eligible:
            report["failureReason"] = f"builder2_media_finalization_not_eligible:{','.join(missing)}"
            report["failureStage"] = "eligibility"
            return report

        if acquire_lease:
            if not acquire_job_lease(job_id, worker_token):
                report["failureReason"] = "builder2_media_finalization_lease_unavailable"
                report["failureStage"] = "lease"
                return report
            lease_acquired = True
            refreshed = _read_raw(job_id)
            if refreshed is not None:
                state = deepcopy(refreshed)
                plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}

        headline_url = backfill_legacy_headline_reference(state, job_video_url=job_video_url)
        report["legacyHeadlineArtifactIdentified"] = bool(headline_url)
        report["legacyHeadlineArtifactRouteFamily"] = classify_url_route_family(headline_url) if headline_url else None

        closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
        source_url, _ = _closure_source_for_recovery(state=state, plan=plan, job_video_url=job_video_url)

        job_data = video_job_get(job_id)
        media_config = build_media_resume_configuration(
            job_id=job_id,
            job_data=job_data,
            tournament_state=state,
            start_image_required=False,
            ffmpeg_required=True,
        )

        MediaFinalizationIsolationGuard.enable_closure()
        MediaFinalizationIsolationGuard.enable_publication()
        MediaFinalizationIsolationGuard.assert_safe_before_closure()
        state, counters = render_advertising_closure_for_state(
            job_id=job_id,
            state=state,
            plan=plan,
            closure=closure,
            public_base_url=media_config.publicBaseUrl,
            source_video_url=source_url,
            render_endcard=render_builder2_advertising_closure_endcard,
        )
        report["ffmpegCalls"] = counters.closure_ffmpeg_calls
        report["publicationCalls"] = 1
        MediaFinalizationIsolationGuard.record_closure_ffmpeg()
        MediaFinalizationIsolationGuard.record_publication()

        final_url = str((state.get("mediaResume") or {}).get("finalVideoWithClosureUrl") or "")
        contract_ok, contract_failure, _ = validate_builder2_media_completion_contract(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
            require_job_video_url_match=False,
        )
        if not contract_ok:
            raise Builder2TournamentError(contract_failure or "builder2_media_completion_contract_failed")

        media = state.setdefault("mediaResume", {})
        media["mediaResumeStatus"] = "completed"
        media["progressStage"] = "completed"
        state["mediaContinuationRequired"] = False
        state["status"] = "completed"
        state["lastCompletedStep"] = "done"
        save_tournament_state(job_id, state)
        report["redisMutations"] = 1

        marketing_text = str(media.get("marketingText") or "")
        overlay_headline = "" if not headline_decision_requires_headline(get_normalized_headline_decision(plan)) else str(
            plan.get("headlineText") or ""
        )
        video_job_mark_done(job_id, final_url, marketing_text, overlay_headline=overlay_headline)
        report["jobCompleted"] = True
        report["ok"] = True
    except Builder2ClosureRenderError as exc:
        report["failureStage"] = exc.stage
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_closure_ffmpeg_failed")
        report["safeFfmpegReturnCode"] = exc.return_code
        report["safeFfmpegStderrAvailable"] = bool(exc.stderr_tail)
        media = state.setdefault("mediaResume", {})
        if isinstance(media, dict):
            media["mediaResumeStatus"] = "finalization_failed"
            media["advertisingClosureStatus"] = "failed"
            media["advertisingClosureFailure"] = {
                "stage": exc.stage,
                "reason": report["failureReason"],
                "returnCode": exc.return_code,
                "stderrTail": exc.stderr_tail,
                "commandCategory": exc.command_category,
            }
            state["status"] = "media_finalization_incomplete"
            state["mediaContinuationRequired"] = True
            save_tournament_state(job_id, state)
            report["redisMutations"] = 1
    except Builder2TournamentError as exc:
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_media_finalization_failed")
        report["failureStage"] = report.get("failureStage") or "finalization"
    finally:
        report.update(
            {
                key: MediaFinalizationIsolationGuard.reasoning_report().get(key, report.get(key))
                for key in ("totalReasoningCalls",)
            }
        )
        if lease_acquired:
            release_job_lease(job_id, worker_token)
        MediaFinalizationIsolationGuard.end()
    return report


def print_media_finalization_resume_report(report: Dict[str, Any]) -> None:
    safe_keys = (
        "jobId",
        "ok",
        "preflight",
        "eligibleForFinalizationRecovery",
        "falseCompletionConfirmed",
        "legacyHeadlineArtifactIdentified",
        "legacyHeadlineArtifactRouteFamily",
        "headlineArtifactDownloadAccepted",
        "measuredHeadlineDurationSeconds",
        "closureRenderAttempted",
        "closureRenderAccepted",
        "measuredFinalDurationSeconds",
        "finalDurationAccepted",
        "exactProductAndSloganReused",
        "readyForFinalizationRecovery",
        "failureStage",
        "failureReason",
        "safeFfmpegReturnCode",
        "safeFfmpegStderrAvailable",
        "openAICalls",
        "imageCalls",
        "runwaySubmissionCalls",
        "runwayPollingCalls",
        "ffmpegCalls",
        "publicationCalls",
        "redisMutations",
        "finalizationReused",
        "jobCompleted",
        "totalReasoningCalls",
    )
    safe = {key: report.get(key) for key in safe_keys if key in report}
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID")
    if not job_id:
        print(json.dumps({"ok": False, "failureReason": "builder2_media_finalization_resume_job_id_missing"}, indent=2))
        return 1
    preflight = _truthy("BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT")
    logger.info(
        "BUILDER2_MEDIA_FINALIZATION_RESUME_START jobId=%s preflight=%s",
        job_id,
        preflight,
    )
    report = run_one_media_finalization_resume(job_id=job_id, preflight=preflight)
    print_media_finalization_resume_report(report)
    logger.info(
        "BUILDER2_MEDIA_FINALIZATION_RESUME_DONE jobId=%s ok=%s preflight=%s",
        job_id,
        report.get("ok"),
        preflight,
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
