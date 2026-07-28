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
from engine.builder2_local_headline_render import (
    VideoHeadlineRenderError,
    render_builder2_accepted_headline_overlay,
)
from engine.builder2_media_finalization_contract import (
    backfill_legacy_headline_reference,
    finalization_recovery_eligible,
    validate_builder2_media_completion_contract,
)
from engine.builder2_media_finalization_download import SafeDownloadDiagnostics
from engine.builder2_media_finalization_guard import MediaFinalizationIsolationGuard
from engine.builder2_media_finalization_source import (
    SOURCE_RAW_RUNWAY_LOCAL_HEADLINE,
    FinalizationSourceDecision,
    resolve_finalization_source_decision,
)
from engine.builder2_media_resume_config import build_media_resume_configuration
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import _read_raw, save_tournament_state
from engine.video_jobs_redis import redis_configured, video_job_get, video_job_get_raw, video_job_mark_done

logger = logging.getLogger(__name__)


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
        "legacyHeadlineDownloadAttempted": False,
        "legacyHeadlineDownloadAccepted": False,
        "legacyHeadlineHttpStatusCode": None,
        "legacyHeadlineDownloadFailureCategory": None,
        "legacyHeadlineArtifactUnavailable": False,
        "rawRunwayFallbackAttempted": False,
        "rawRunwayFallbackAccepted": False,
        "rawRunwayDownloadAccepted": False,
        "localHeadlineRenderRequired": False,
        "localHeadlineRenderAttempted": False,
        "localHeadlineRenderAccepted": False,
        "measuredRawRunwayDurationSeconds": None,
        "measuredHeadlineDurationSeconds": None,
        "closureRenderAttempted": False,
        "closureRenderAccepted": False,
        "measuredFinalDurationSeconds": None,
        "finalDurationAccepted": False,
        "exactProductAndSloganReused": False,
        "selectedFinalizationSourceKind": None,
        "readyForFinalizationRecovery": False,
        "failureStage": None,
        "failureReason": None,
        "safeFfmpegReturnCode": None,
        "safeFfmpegStderrAvailable": False,
        "openAICalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
        "runwayPollingCalls": 0,
        "headlineFfmpegCalls": 0,
        "closureFfmpegCalls": 0,
        "ffprobeCalls": 0,
        "ffmpegCalls": 0,
        "totalFfmpegCalls": 0,
        "publicationCalls": 0,
        "redisMutations": 0,
        "finalizationReused": False,
        "jobCompleted": False,
        "totalReasoningCalls": 0,
    }


def _probe_duration(path: Path) -> float:
    from engine.builder2_closure_render import _ffprobe_duration_seconds, _FFPROBE_TIMEOUT

    return _ffprobe_duration_seconds(path, _FFPROBE_TIMEOUT)


def _apply_download_diagnostics(report: Dict[str, Any], diagnostics: Optional[SafeDownloadDiagnostics]) -> None:
    if diagnostics is None:
        return
    diag = diagnostics.to_report_dict()
    report["legacyHeadlineDownloadAttempted"] = diag.get("requestAttempted", False)
    report["legacyHeadlineDownloadAccepted"] = diag.get("downloadAccepted", False)
    report["legacyHeadlineHttpStatusCode"] = diag.get("httpStatusCode")
    report["legacyHeadlineDownloadFailureCategory"] = diag.get("downloadFailureCategory")
    report["legacyHeadlineArtifactUnavailable"] = diag.get("legacyHeadlineArtifactUnavailable", False)
    report["requestAttempted"] = diag.get("requestAttempted")
    report["requestMethod"] = diag.get("requestMethod")
    report["originalRouteFamily"] = diag.get("originalRouteFamily")
    report["redirectCount"] = diag.get("redirectCount")
    report["finalRouteFamily"] = diag.get("finalRouteFamily")
    report["httpStatusCode"] = diag.get("httpStatusCode")
    report["responseContentType"] = diag.get("responseContentType")
    report["responseContentLength"] = diag.get("responseContentLength")
    report["downloadFailureClass"] = diag.get("downloadFailureClass")
    report["downloadFailureCategory"] = diag.get("downloadFailureCategory")


def _apply_source_decision(report: Dict[str, Any], decision: FinalizationSourceDecision) -> None:
    report.update(decision.to_report_dict())
    report["headlineArtifactDownloadAccepted"] = bool(
        decision.source_kind in {"persisted_headline_artifact", "legacy_headline_artifact"}
        and decision.closure_input_path is not None
    )
    if decision.legacy_headline_diagnostics is not None:
        _apply_download_diagnostics(report, decision.legacy_headline_diagnostics)


def _sync_ffmpeg_counters(report: Dict[str, Any]) -> None:
    report["totalFfmpegCalls"] = int(report.get("headlineFfmpegCalls") or 0) + int(
        report.get("closureFfmpegCalls") or 0
    )
    report["ffmpegCalls"] = report["totalFfmpegCalls"]


def _execute_finalization_render_pipeline(
    *,
    job_id: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
    report: Dict[str, Any],
    preflight: bool,
    public_base_url: str,
) -> Optional[Dict[str, Any]]:
    closure = state.get("advertisingClosure") if isinstance(state.get("advertisingClosure"), dict) else {}
    report["exactProductAndSloganReused"] = bool(
        str(closure.get("productNameText") or "").strip() and str(closure.get("sloganText") or "").strip()
    )

    headline_url = backfill_legacy_headline_reference(state, job_video_url=job_video_url)
    report["legacyHeadlineArtifactIdentified"] = bool(headline_url)
    report["legacyHeadlineArtifactRouteFamily"] = classify_url_route_family(headline_url) if headline_url else None

    tmp = Path(tempfile.mkdtemp(prefix="ace_finalization_"))
    try:
        decision = resolve_finalization_source_decision(
            state=state,
            plan=plan,
            job_video_url=job_video_url,
            work_dir=tmp,
            download_headline=True,
        )
        _apply_source_decision(report, decision)
        if decision.failure_reason:
            report["failureStage"] = decision.failure_stage or "source_selection"
            report["failureReason"] = decision.failure_reason
            return None
        if decision.closure_input_path is None:
            report["failureStage"] = "source_selection"
            report["failureReason"] = "builder2_media_finalization_closure_input_missing"
            return None

        closure_input = decision.closure_input_path
        if decision.raw_runway_diagnostics and decision.raw_runway_diagnostics.download_accepted:
            report["measuredRawRunwayDurationSeconds"] = _probe_duration(closure_input)
            report["ffprobeCalls"] = int(report.get("ffprobeCalls") or 0) + 1

        if decision.local_headline_render_required:
            headline_out = tmp / "headline_local.mp4"
            report["localHeadlineRenderAttempted"] = True
            try:
                headline_result = render_builder2_accepted_headline_overlay(
                    source_video_path=closure_input,
                    output_path=headline_out,
                    plan=plan,
                )
            except VideoHeadlineRenderError as exc:
                report["failureStage"] = exc.stage
                report["failureReason"] = str(exc.args[0] if exc.args else "builder2_local_headline_render_failed")
                report["safeFfmpegReturnCode"] = exc.return_code
                report["safeFfmpegStderrAvailable"] = bool(exc.stderr_tail)
                report["headlineFfmpegCalls"] = 1
                _sync_ffmpeg_counters(report)
                return None
            report["localHeadlineRenderAccepted"] = True
            report["headlineFfmpegCalls"] = 1
            report["measuredHeadlineDurationSeconds"] = headline_result.measured_duration_seconds
            report["ffprobeCalls"] = int(report.get("ffprobeCalls") or 0) + 1
            closure_input = headline_result.output_path
        elif decision.source_kind in {"persisted_headline_artifact", "legacy_headline_artifact"}:
            report["measuredHeadlineDurationSeconds"] = _probe_duration(closure_input)
            report["ffprobeCalls"] = int(report.get("ffprobeCalls") or 0) + 1

        output_path = tmp / "closure_out.mp4"
        source_for_closure = str(closure_input)
        report["closureRenderAttempted"] = True
        try:
            render_result = render_builder2_advertising_closure_endcard(
                source_for_closure,
                public_base_url,
                product_name=str(closure.get("productNameText") or ""),
                slogan=str(closure.get("sloganText") or ""),
                language=str(closure.get("language") or "en"),
                duration_seconds=float(closure.get("durationSeconds") or 2.0),
                job_id=job_id,
                publish=not preflight,
                output_path=output_path if preflight else None,
            )
        except Builder2ClosureRenderError as exc:
            report["failureStage"] = exc.stage
            report["failureReason"] = str(exc.args[0] if exc.args else "builder2_closure_ffmpeg_failed")
            report["safeFfmpegReturnCode"] = exc.return_code
            report["safeFfmpegStderrAvailable"] = bool(exc.stderr_tail)
            report["closureFfmpegCalls"] = 1
            _sync_ffmpeg_counters(report)
            return None

        report["closureRenderAccepted"] = True
        report["closureFfmpegCalls"] = 1
        report["measuredFinalDurationSeconds"] = render_result.measured_duration_seconds
        report["finalDurationAccepted"] = True
        report["ffprobeCalls"] = int(report.get("ffprobeCalls") or 0) + 1
        _sync_ffmpeg_counters(report)

        if preflight:
            report["readyForFinalizationRecovery"] = True
            report["ok"] = True
            return None

        media = state.setdefault("mediaResume", {})
        if decision.local_headline_render_required:
            media["headlineArtifactSource"] = "deterministic_local_reconstruction_from_raw_runway"
            media["headlineReconstructionCompleted"] = True
            media["headlineReconstructionDurationSeconds"] = report["measuredHeadlineDurationSeconds"]
            media["headlinePostprocessStatus"] = "reconstructed"
            media.pop("headlineArtifactUrl", None)
        elif decision.source_kind in {"persisted_headline_artifact", "legacy_headline_artifact"}:
            media["headlineArtifactUrl"] = headline_url or decision.legacy_headline_url or decision.persisted_headline_url
            media["headlinePostprocessStatus"] = "completed"

        media["finalVideoWithClosureUrl"] = render_result.public_url
        media["finalPublicUrl"] = render_result.public_url
        media["finalVideoPath"] = render_result.public_url
        media["actualFinalVideoDurationSeconds"] = render_result.measured_duration_seconds
        media["advertisingClosureRendered"] = True
        media["advertisingClosureStatus"] = "completed"
        state["advertisingClosureStatus"] = "completed"
        return render_result
    finally:
        try:
            for path in tmp.iterdir():
                path.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass


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

    job_data = video_job_get(job_id) if redis_configured() else None
    media_config = build_media_resume_configuration(
        job_id=job_id,
        job_data=job_data,
        tournament_state=state,
        start_image_required=False,
        ffmpeg_required=True,
    )
    _execute_finalization_render_pipeline(
        job_id=job_id,
        state=state,
        plan=plan,
        job_video_url=job_video_url,
        report=report,
        preflight=True,
        public_base_url=media_config.publicBaseUrl,
    )
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

        render_result = _execute_finalization_render_pipeline(
            job_id=job_id,
            state=state,
            plan=plan,
            job_video_url=job_video_url,
            report=report,
            preflight=False,
            public_base_url=media_config.publicBaseUrl,
        )
        if not report.get("ok") and render_result is None:
            if report.get("failureStage") in {"headline_overlay", "duration_probe", "input_validation"}:
                media = state.setdefault("mediaResume", {})
                if isinstance(media, dict):
                    media["mediaResumeStatus"] = "finalization_failed"
                    media["headlinePostprocessStatus"] = "failed"
                    state["status"] = "media_finalization_incomplete"
                    state["mediaContinuationRequired"] = True
                    save_tournament_state(job_id, state)
                    report["redisMutations"] = 1
            elif report.get("failureStage") not in {None, "lease", "eligibility", "load", "configuration"}:
                media = state.setdefault("mediaResume", {})
                if isinstance(media, dict):
                    media["mediaResumeStatus"] = "finalization_failed"
                    media["advertisingClosureStatus"] = "failed"
                    state["status"] = "media_finalization_incomplete"
                    state["mediaContinuationRequired"] = True
                    save_tournament_state(job_id, state)
                    report["redisMutations"] = 1
            return report

        if render_result is None:
            return report

        report["publicationCalls"] = 1
        MediaFinalizationIsolationGuard.record_closure_ffmpeg()
        MediaFinalizationIsolationGuard.record_publication()

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
        video_job_mark_done(job_id, render_result.public_url, marketing_text, overlay_headline=overlay_headline)
        report["jobCompleted"] = True
        report["ok"] = True
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
        "legacyHeadlineDownloadAttempted",
        "legacyHeadlineDownloadAccepted",
        "legacyHeadlineHttpStatusCode",
        "legacyHeadlineDownloadFailureCategory",
        "legacyHeadlineArtifactUnavailable",
        "requestAttempted",
        "requestMethod",
        "originalRouteFamily",
        "redirectCount",
        "finalRouteFamily",
        "httpStatusCode",
        "responseContentType",
        "responseContentLength",
        "downloadFailureClass",
        "downloadFailureCategory",
        "rawRunwayFallbackAttempted",
        "rawRunwayFallbackAccepted",
        "rawRunwayDownloadAccepted",
        "localHeadlineRenderRequired",
        "localHeadlineRenderAttempted",
        "localHeadlineRenderAccepted",
        "measuredRawRunwayDurationSeconds",
        "measuredHeadlineDurationSeconds",
        "closureRenderAttempted",
        "closureRenderAccepted",
        "measuredFinalDurationSeconds",
        "finalDurationAccepted",
        "exactProductAndSloganReused",
        "selectedFinalizationSourceKind",
        "readyForFinalizationRecovery",
        "failureStage",
        "failureReason",
        "safeFfmpegReturnCode",
        "safeFfmpegStderrAvailable",
        "openAICalls",
        "imageCalls",
        "runwaySubmissionCalls",
        "runwayPollingCalls",
        "headlineFfmpegCalls",
        "closureFfmpegCalls",
        "ffprobeCalls",
        "totalFfmpegCalls",
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
