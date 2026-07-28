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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    ClosureRenderResult,
    classify_url_route_family,
    render_builder2_advertising_closure_endcard,
)
from engine.builder2_final_local_staging import Builder2FinalLocalStagingError, prepare_publication_staging
from engine.builder2_final_video_publication import (
    Builder2FinalPublicationError,
    FinalVideoPublicationResult,
    durable_publication_required,
    probe_builder2_final_video_web_storage_capability,
    publish_builder2_final_video,
    resolve_durable_final_video_publisher_kind,
)
from engine.builder2_execution_lease import acquire_job_lease, release_job_lease
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_single_slogan_contract import builder2_requires_headline_overlay
from engine.builder2_local_headline_render import (
    VideoHeadlineRenderError,
    render_builder2_accepted_headline_overlay,
)
from engine.builder2_media_finalization_contract import (
    backfill_legacy_headline_reference,
    evaluate_finalization_recovery_eligibility,
    finalization_recovery_eligible,
    validate_builder2_media_completion_contract,
)
from engine.builder2_media_finalization_download import SafeDownloadDiagnostics
from engine.builder2_media_finalization_reporting import (
    build_minimal_fallback_report,
    emit_fail_safe_media_finalization_report,
    preserve_original_failure,
    sanitize_media_finalization_report,
)
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
        "legacyFalseCompletionConfirmed": False,
        "recoverableFailedFinalizationConfirmed": False,
        "recoveryEligibilityBasis": None,
        "recoverableFailedFinalizationConditionResults": None,
        "recoverableFailedFinalizationReasons": None,
        "recoveryBlockedByValidFinal": False,
        "recoveryBlockedByCompletedPublication": False,
        "recoveryBlockedByMissingIntermediate": False,
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
        "headlineRenderAttempts": 0,
        "headlineFfmpegSubprocessCalls": 0,
        "closureRenderAttempts": 0,
        "closureFfmpegSubprocessCalls": 0,
        "ffprobeCalls": 0,
        "rawRunwayFfprobeCalls": 0,
        "headlineFfprobeCalls": 0,
        "finalClosureFfprobeCalls": 0,
        "totalFfprobeSubprocessCalls": 0,
        "closureFfmpegExecutionAccepted": False,
        "closureOutputFileCreated": False,
        "closureOutputFileSizeBytes": None,
        "closureDurationProbeAttempted": False,
        "closureDurationProbeAccepted": False,
        "measuredClosureOutputDurationSeconds": None,
        "measuredClosureSourceDurationSeconds": None,
        "configuredVisualDurationSeconds": None,
        "configuredEndCardDurationSeconds": None,
        "effectiveClosureSegmentDurationSeconds": None,
        "configuredFinalDurationSeconds": None,
        "calculatedExpectedFinalDurationSeconds": None,
        "actualClosureGainSeconds": None,
        "closureGainAccepted": False,
        "acceptedFinalDurationLowerBoundSeconds": None,
        "acceptedFinalDurationUpperBoundSeconds": None,
        "finalDurationDeltaSeconds": None,
        "finalDurationVerificationFailureCode": None,
        "closureFailureSubstage": None,
        "ffmpegCalls": 0,
        "totalFfmpegCalls": 0,
        "publicationCalls": 0,
        "redisMutations": 0,
        "finalizationReused": False,
        "jobCompleted": False,
        "totalReasoningCalls": 0,
        "acceptedHeadlineDecision": None,
        "acceptedHeadlineFieldPresent": False,
        "acceptedHeadlineKeywordPresent": False,
        "persistedHeadlineTextPresent": False,
        "canonicalHeadlineResolutionAttempted": False,
        "canonicalHeadlineResolutionAccepted": False,
        "canonicalHeadlineSource": None,
        "canonicalHeadlineCharacterCount": 0,
        "canonicalHeadlineWordCount": 0,
        "localHeadlineInputPresent": False,
        "localHeadlineFailureStage": None,
        "localHeadlineFailureCode": None,
        "originalFailureClass": None,
        "originalFailureStage": None,
        "originalFailureCode": None,
        "reportingFailureClass": None,
        "reportingFailureOccurred": False,
        "leaseReleaseAttempted": False,
        "leaseReleaseAccepted": False,
        "cliReportConstructionAccepted": False,
        "cliJsonSerializationAccepted": False,
        "cliStdoutWriteAttempted": False,
        "localFinalRenderCompleted": False,
        "localFinalArtifactPresentAfterRender": False,
        "localFinalArtifactSizeBytes": None,
        "localFinalOwnership": None,
        "publicationStagingPreparationAttempted": False,
        "publicationStagingPreparationAccepted": False,
        "durablePublisherResolved": None,
        "durablePublicationRequired": False,
        "legacyHeadlineStoreRejectedAsFinalDestination": False,
        "finalLocalHandoffFailureCode": None,
        "localRenderAccepted": False,
        "publicationAccepted": False,
        "persistedCompletionAccepted": False,
        "storageCapabilityCalls": 0,
        "storageCapabilityAccepted": False,
        "webDurableStorageConfirmed": False,
        "webPublicationBackendKind": None,
        "webStorageWritable": False,
        "webStorageFailureCode": None,
    }


@dataclass(frozen=True)
class FinalizationPipelineOutcome:
    render_result: ClosureRenderResult
    publication_result: Optional[FinalVideoPublicationResult]
    public_url: str


def _safe_failure_code(code: str) -> str:
    token = (code or "").strip()
    if not token:
        return "builder2_media_finalization_failed"
    lowered = token.lower()
    if "http://" in lowered or "https://" in lowered or "/" in token or "\\" in token:
        return "builder2_media_finalization_failed"
    return token[:128]


def _apply_recovery_eligibility_to_report(
    report: Dict[str, Any],
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
    active_finalization_lease: bool = False,
) -> Dict[str, Any]:
    evaluation = evaluate_finalization_recovery_eligibility(
        state=state,
        plan=plan,
        job_video_url=job_video_url,
        active_finalization_lease=active_finalization_lease,
    )
    report["eligibleForFinalizationRecovery"] = bool(evaluation["eligible"])
    report["legacyFalseCompletionConfirmed"] = bool(evaluation["legacyFalseCompletionConfirmed"])
    report["recoverableFailedFinalizationConfirmed"] = bool(evaluation["recoverableFailedFinalizationConfirmed"])
    report["falseCompletionConfirmed"] = bool(evaluation["falseCompletionConfirmed"])
    report["recoveryEligibilityBasis"] = evaluation.get("recoveryEligibilityBasis")
    report["recoverableFailedFinalizationConditionResults"] = evaluation.get(
        "recoverableFailedFinalizationConditionResults"
    )
    report["recoverableFailedFinalizationReasons"] = evaluation.get("recoverableFailedFinalizationReasons")
    report["recoveryBlockedByValidFinal"] = bool(evaluation.get("recoveryBlockedByValidFinal"))
    report["recoveryBlockedByCompletedPublication"] = bool(evaluation.get("recoveryBlockedByCompletedPublication"))
    report["recoveryBlockedByMissingIntermediate"] = bool(evaluation.get("recoveryBlockedByMissingIntermediate"))
    return evaluation


def _persist_finalization_failure_state(
    state: Dict[str, Any],
    *,
    report: Dict[str, Any],
    headline_failure: bool = False,
) -> None:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        return
    media["mediaResumeStatus"] = "finalization_failed"
    if headline_failure:
        media["headlinePostprocessStatus"] = "failed"
    else:
        media["advertisingClosureStatus"] = "failed"
    state["status"] = "media_finalization_incomplete"
    state["mediaContinuationRequired"] = True
    stage = report.get("originalFailureStage") or report.get("failureStage")
    code = report.get("originalFailureCode") or report.get("failureReason")
    failure_class = report.get("originalFailureClass")
    if stage:
        media["finalizationFailureStage"] = str(stage)[:64]
    if code:
        media["finalizationFailureCode"] = _safe_failure_code(str(code))
    if failure_class:
        media["finalizationFailureClass"] = str(failure_class)[:64]


def _archive_finalization_failure_state(media: Dict[str, Any]) -> None:
    if not isinstance(media, dict):
        return
    stage = media.get("finalizationFailureStage")
    code = media.get("finalizationFailureCode")
    failure_class = media.get("finalizationFailureClass")
    if stage or code or failure_class:
        media["lastFinalizationFailure"] = {
            "stage": stage,
            "code": code,
            "class": failure_class,
        }
    media.pop("finalizationFailureStage", None)
    media.pop("finalizationFailureCode", None)
    media.pop("finalizationFailureClass", None)


def _probe_duration(path: Path) -> float:
    from engine.builder2_closure_render import _ffprobe_duration_seconds, _FFPROBE_TIMEOUT

    return _ffprobe_duration_seconds(path, _FFPROBE_TIMEOUT)


def _record_ffprobe_call(report: Dict[str, Any], *, category: str) -> None:
    key = {
        "raw_runway": "rawRunwayFfprobeCalls",
        "headline": "headlineFfprobeCalls",
        "final_closure": "finalClosureFfprobeCalls",
    }.get(category)
    if key is None:
        raise ValueError(f"unknown ffprobe category: {category}")
    report[key] = int(report.get(key) or 0) + 1
    total = (
        int(report.get("rawRunwayFfprobeCalls") or 0)
        + int(report.get("headlineFfprobeCalls") or 0)
        + int(report.get("finalClosureFfprobeCalls") or 0)
    )
    report["totalFfprobeSubprocessCalls"] = total
    report["ffprobeCalls"] = total


def _apply_closure_duration_diagnostics(
    report: Dict[str, Any],
    exc: Builder2ClosureRenderError,
) -> None:
    if exc.closure_ffmpeg_execution_accepted is not None:
        report["closureFfmpegExecutionAccepted"] = bool(exc.closure_ffmpeg_execution_accepted)
    if exc.closure_output_file_created is not None:
        report["closureOutputFileCreated"] = bool(exc.closure_output_file_created)
    if exc.closure_output_file_size_bytes is not None:
        report["closureOutputFileSizeBytes"] = int(exc.closure_output_file_size_bytes)
    if exc.duration_diagnostics is not None:
        report.update(exc.duration_diagnostics.to_report_dict())
        measured = exc.duration_diagnostics.measured_closure_output_duration_seconds
        report["measuredFinalDurationSeconds"] = measured
        report["measuredClosureOutputDurationSeconds"] = measured
    if exc.closure_ffprobe_calls:
        report["finalClosureFfprobeCalls"] = int(report.get("finalClosureFfprobeCalls") or 0) + int(
            exc.closure_ffprobe_calls
        )
        total = (
            int(report.get("rawRunwayFfprobeCalls") or 0)
            + int(report.get("headlineFfprobeCalls") or 0)
            + int(report.get("finalClosureFfprobeCalls") or 0)
        )
        report["totalFfprobeSubprocessCalls"] = total
        report["ffprobeCalls"] = total


def _apply_closure_success_diagnostics(
    report: Dict[str, Any],
    render_result: Any,
) -> None:
    report["closureFfmpegExecutionAccepted"] = True
    report["closureDurationProbeAttempted"] = True
    report["closureDurationProbeAccepted"] = True
    report["measuredFinalDurationSeconds"] = render_result.measured_duration_seconds
    if render_result.duration_diagnostics is not None:
        report.update(render_result.duration_diagnostics.to_report_dict())
        report["finalDurationAccepted"] = True
        report["closureGainAccepted"] = bool(render_result.duration_diagnostics.closure_gain_accepted)
    if render_result.closure_ffprobe_calls:
        report["finalClosureFfprobeCalls"] = int(report.get("finalClosureFfprobeCalls") or 0) + int(
            render_result.closure_ffprobe_calls
        )
        total = (
            int(report.get("rawRunwayFfprobeCalls") or 0)
            + int(report.get("headlineFfprobeCalls") or 0)
            + int(report.get("finalClosureFfprobeCalls") or 0)
        )
        report["totalFfprobeSubprocessCalls"] = total
        report["ffprobeCalls"] = total


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
    headline_sub = int(report.get("headlineFfmpegSubprocessCalls") or 0)
    closure_sub = int(report.get("closureFfmpegSubprocessCalls") or 0)
    report["headlineFfmpegCalls"] = headline_sub
    report["closureFfmpegCalls"] = closure_sub
    report["totalFfmpegCalls"] = headline_sub + closure_sub
    report["ffmpegCalls"] = report["totalFfmpegCalls"]


def _apply_unexpected_failure(report: Dict[str, Any], exc: BaseException) -> None:
    preserve_original_failure(report, exc)
    if report.get("failureReason"):
        return
    report["failureStage"] = report.get("failureStage") or "internal"
    if isinstance(exc, Builder2TournamentError):
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_media_finalization_failed")
    else:
        report["failureReason"] = "builder2_media_finalization_unexpected_internal_error"
    logger.exception("BUILDER2_MEDIA_FINALIZATION_UNEXPECTED_FAILURE")


def _release_execution_lease_safely(
    report: Dict[str, Any],
    *,
    job_id: str,
    worker_token: str,
    lease_acquired: bool,
) -> None:
    if not lease_acquired:
        report["leaseReleaseAttempted"] = False
        report["leaseReleaseAccepted"] = False
        return
    report["leaseReleaseAttempted"] = True
    try:
        release_job_lease(job_id, worker_token)
        report["leaseReleaseAccepted"] = True
    except Exception:
        report["leaseReleaseAccepted"] = False
        logger.exception("BUILDER2_MEDIA_FINALIZATION_LEASE_RELEASE_FAILED jobId=%s", job_id)


def _finalize_isolation_guard_safely() -> None:
    try:
        MediaFinalizationIsolationGuard.end()
    except Exception:
        logger.exception("BUILDER2_MEDIA_FINALIZATION_ISOLATION_GUARD_END_FAILED")


def _closure_subprocess_ran(stage: str) -> bool:
    return stage in {"card_generation", "concatenation", "duration_probe", "duration_verification", "publication"}


def _preserve_render_diagnostics_after_render_success(
    report: Dict[str, Any],
    render_result: ClosureRenderResult,
    *,
    local_final_path: Path,
) -> None:
    report["localRenderAccepted"] = True
    report["closureRenderAccepted"] = True
    report["localFinalRenderCompleted"] = True
    report["localFinalOwnership"] = "caller_owned"
    report["legacyHeadlineStoreRejectedAsFinalDestination"] = True
    report["durablePublisherResolved"] = resolve_durable_final_video_publisher_kind()
    report["durablePublicationRequired"] = durable_publication_required()
    if local_final_path.is_file():
        report["localFinalArtifactPresentAfterRender"] = True
        report["localFinalArtifactSizeBytes"] = int(local_final_path.stat().st_size)
    report["closureFfmpegSubprocessCalls"] = int(report.get("closureFfmpegSubprocessCalls") or 0) or 1
    _apply_closure_success_diagnostics(report, render_result)
    report["finalDurationAccepted"] = True
    _sync_ffmpeg_counters(report)


def _apply_web_storage_capability_to_report(
    report: Dict[str, Any],
    capability: Any,
    *,
    increment_calls: bool = True,
) -> bool:
    if increment_calls:
        report["storageCapabilityCalls"] = int(report.get("storageCapabilityCalls") or 0) + 1
    report.update(capability.to_report_dict())
    report["storageCapabilityAccepted"] = bool(capability.accepted)
    if capability.accepted:
        report["webDurableStorageConfirmed"] = True
        report["webPublicationBackendKind"] = capability.publication_backend_kind
        report["webStorageWritable"] = capability.storage_writable
        report["durablePublisherResolved"] = resolve_durable_final_video_publisher_kind()
    return bool(capability.accepted)


def _probe_web_storage_capability_or_fail(
    report: Dict[str, Any],
    *,
    public_base_url: str,
) -> bool:
    capability = probe_builder2_final_video_web_storage_capability(public_base_url)
    accepted = _apply_web_storage_capability_to_report(report, capability)
    if not accepted:
        report["failureStage"] = "publication_capability"
        report["failureReason"] = _safe_failure_code(capability.failure_code or "builder2_web_storage_not_persistent")
        report["webStorageFailureCode"] = capability.failure_code or None
    return accepted


def _apply_publication_failure(
    report: Dict[str, Any],
    exc: BaseException,
    *,
    render_result: ClosureRenderResult,
    local_final_path: Path,
) -> None:
    _preserve_render_diagnostics_after_render_success(report, render_result, local_final_path=local_final_path)
    preserve_original_failure(report, exc)
    report["failureStage"] = getattr(exc, "stage", None) or "publication"
    report["failureReason"] = str(exc.args[0] if exc.args else "builder2_final_publication_failed")
    report["publicationAccepted"] = False
    report["persistedCompletionAccepted"] = False
    if isinstance(exc, Builder2FinalPublicationError):
        server_code = getattr(exc, "server_failure_code", "") or ""
        report["finalLocalHandoffFailureCode"] = _safe_failure_code(
            server_code or str(exc.args[0] if exc.args else "")
        )
        if server_code:
            report["webStorageFailureCode"] = server_code
        partial = getattr(exc, "verification", None)
        if partial is not None:
            report["postUploadVerificationAttempted"] = partial.post_upload_verification_attempted
            report["postUploadVerificationAccepted"] = partial.post_upload_verification_accepted
            report["postUploadHttpStatusCode"] = partial.post_upload_http_status_code
            report["durableStorageConfirmed"] = partial.durable_storage_confirmed
            if exc.stage == "publication_verification":
                report["failureStage"] = "publication_verification"
    elif isinstance(exc, Builder2FinalLocalStagingError):
        report["failureStage"] = exc.stage
        report["finalLocalHandoffFailureCode"] = str(exc.args[0] if exc.args else "")


def _execute_finalization_render_pipeline(
    *,
    job_id: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_video_url: str,
    report: Dict[str, Any],
    preflight: bool,
    public_base_url: str,
) -> Optional[FinalizationPipelineOutcome]:
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
            _record_ffprobe_call(report, category="raw_runway")

        if decision.local_headline_render_required:
            headline_out = tmp / "headline_local.mp4"
            report["localHeadlineRenderAttempted"] = True
            report["headlineRenderAttempts"] = 1
            try:
                headline_result = render_builder2_accepted_headline_overlay(
                    source_video_path=closure_input,
                    output_path=headline_out,
                    plan=plan,
                    report=report,
                )
            except VideoHeadlineRenderError as exc:
                report["failureStage"] = exc.stage
                report["failureReason"] = str(exc.args[0] if exc.args else "builder2_local_headline_render_failed")
                report["localHeadlineFailureStage"] = exc.stage
                report["localHeadlineFailureCode"] = str(exc.args[0] if exc.args else "")
                report["safeFfmpegReturnCode"] = exc.return_code
                report["safeFfmpegStderrAvailable"] = bool(exc.stderr_tail)
                if exc.stage == "headline_overlay":
                    report["headlineFfmpegSubprocessCalls"] = 1
                _sync_ffmpeg_counters(report)
                return None
            report["localHeadlineRenderAccepted"] = True
            report["headlineFfmpegSubprocessCalls"] = 1
            report["measuredHeadlineDurationSeconds"] = headline_result.measured_duration_seconds
            _record_ffprobe_call(report, category="headline")
            closure_input = headline_result.output_path
            _sync_ffmpeg_counters(report)
        elif decision.source_kind in {"persisted_headline_artifact", "legacy_headline_artifact"}:
            report["measuredHeadlineDurationSeconds"] = _probe_duration(closure_input)
            _record_ffprobe_call(report, category="headline")

        output_path = tmp / "builder2_final.mp4"
        source_for_closure = str(closure_input)
        report["closureRenderAttempted"] = True
        report["closureRenderAttempts"] = 1
        render_result: Optional[ClosureRenderResult] = None
        try:
            render_result = render_builder2_advertising_closure_endcard(
                source_for_closure,
                product_name=str(closure.get("productNameText") or ""),
                slogan=str(closure.get("sloganText") or ""),
                output_path=output_path,
                language=str(closure.get("language") or "en"),
                duration_seconds=float(closure.get("durationSeconds")) if closure.get("durationSeconds") is not None else None,
                job_id=job_id,
            )
        except Builder2ClosureRenderError as exc:
            preserve_original_failure(report, exc)
            report["failureStage"] = exc.stage
            report["failureReason"] = str(exc.args[0] if exc.args else "builder2_closure_ffmpeg_failed")
            report["safeFfmpegReturnCode"] = exc.return_code
            report["safeFfmpegStderrAvailable"] = bool(exc.stderr_tail)
            if exc.duration_diagnostics is not None or exc.closure_ffmpeg_execution_accepted:
                _apply_closure_duration_diagnostics(report, exc)
                if exc.duration_diagnostics is not None:
                    report["localRenderAccepted"] = True
                    report["localFinalRenderCompleted"] = True
            if _closure_subprocess_ran(exc.stage):
                report["closureFfmpegSubprocessCalls"] = 1
            report["finalLocalHandoffFailureCode"] = str(exc.args[0] if exc.args else "") if exc.stage == "local_staging" else None
            _sync_ffmpeg_counters(report)
            return None
        except Exception as exc:
            _apply_unexpected_failure(report, exc)
            report["failureStage"] = report.get("failureStage") or "closure_render"
            return None

        _preserve_render_diagnostics_after_render_success(
            report,
            render_result,
            local_final_path=output_path,
        )

        staging = prepare_publication_staging(local_final_path=output_path)
        report.update(staging)

        if preflight:
            if not staging.get("publicationStagingPreparationAccepted"):
                report["failureStage"] = "local_staging"
                report["failureReason"] = "builder2_final_local_artifact_missing_after_render"
                return None
            report["readyForFinalizationRecovery"] = True
            report["ok"] = True
            return FinalizationPipelineOutcome(
                render_result=render_result,
                publication_result=None,
                public_url="",
            )

        try:
            publication_result = publish_builder2_final_video(
                output_path,
                public_base_url,
                job_id=job_id,
                output_token=render_result.output_token,
            )
        except (Builder2FinalPublicationError, Builder2FinalLocalStagingError) as exc:
            _apply_publication_failure(report, exc, render_result=render_result, local_final_path=output_path)
            return None
        except Exception as exc:
            _apply_publication_failure(report, exc, render_result=render_result, local_final_path=output_path)
            return None

        report["publicationCalls"] = 1
        report["publicationAccepted"] = publication_result.publication_accepted
        report["durableStorageConfirmed"] = publication_result.durable_storage_confirmed
        report["postUploadVerificationAttempted"] = publication_result.post_upload_verification_attempted
        report["postUploadVerificationAccepted"] = publication_result.post_upload_verification_accepted
        report["postUploadHttpStatusCode"] = publication_result.post_upload_http_status_code
        report["uploadedByteCount"] = publication_result.uploaded_byte_count
        report["webDurableStorageConfirmed"] = publication_result.durable_storage_confirmed
        report["webPublicationBackendKind"] = publication_result.publication_backend_kind
        report["webStorageWritable"] = publication_result.web_storage_writable
        MediaFinalizationIsolationGuard.record_publication()

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

        public_url = publication_result.public_url
        media["finalVideoWithClosureUrl"] = public_url
        media["finalPublicUrl"] = public_url
        media["finalVideoPath"] = public_url
        media["actualFinalVideoDurationSeconds"] = render_result.measured_duration_seconds
        media["finalDurationAccepted"] = True
        media["finalPublicationVerificationAccepted"] = publication_result.post_upload_verification_accepted
        media["finalPublicationDurableStorageConfirmed"] = publication_result.durable_storage_confirmed
        media["finalPublicationBackendKind"] = publication_result.publication_backend_kind
        media["finalPublicationReferencePresent"] = publication_result.publication_reference_present
        media["finalPublicationUploadedByteCount"] = publication_result.uploaded_byte_count
        media["advertisingClosureRendered"] = True
        media["advertisingClosureStatus"] = "completed"
        state["advertisingClosureStatus"] = "completed"
        media.pop("brokenFinalPublicationUrl", None)
        media.pop("invalidFinalPublicationRouteFamily", None)
        return FinalizationPipelineOutcome(
            render_result=render_result,
            publication_result=publication_result,
            public_url=public_url,
        )
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
    try:
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

        eligible_eval = _apply_recovery_eligibility_to_report(
            report,
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )
        if not eligible_eval["eligible"]:
            report["failureReason"] = f"builder2_media_finalization_not_eligible:{','.join(eligible_eval['missing'])}"
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
        if not _probe_web_storage_capability_or_fail(report, public_base_url=media_config.publicBaseUrl):
            return report
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
    except Exception as exc:
        _apply_unexpected_failure(report, exc)
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

        eligible_eval = _apply_recovery_eligibility_to_report(
            report,
            state=state,
            plan=plan,
            job_video_url=job_video_url,
        )
        if not eligible_eval["eligible"] and "validClosureAlreadyPresent" in eligible_eval["missing"]:
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
        if not eligible_eval["eligible"]:
            report["failureReason"] = f"builder2_media_finalization_not_eligible:{','.join(eligible_eval['missing'])}"
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

        render_outcome = _execute_finalization_render_pipeline(
            job_id=job_id,
            state=state,
            plan=plan,
            job_video_url=job_video_url,
            report=report,
            preflight=False,
            public_base_url=media_config.publicBaseUrl,
        )
        if not report.get("ok") and render_outcome is None:
            if report.get("failureStage") in {"headline_overlay", "duration_probe", "input_validation"}:
                _persist_finalization_failure_state(state, report=report, headline_failure=True)
                save_tournament_state(job_id, state)
                report["redisMutations"] = 1
            elif report.get("failureStage") not in {None, "lease", "eligibility", "load", "configuration"}:
                _persist_finalization_failure_state(state, report=report, headline_failure=False)
                save_tournament_state(job_id, state)
                report["redisMutations"] = 1
            return report

        if render_outcome is None:
            return report

        render_result = render_outcome.render_result
        report["publicationCalls"] = int(report.get("publicationCalls") or 0)
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
        _archive_finalization_failure_state(media)
        media["mediaResumeStatus"] = "completed"
        media["progressStage"] = "completed"
        media["advertisingClosureStatus"] = "completed"
        media["advertisingClosureRendered"] = True
        state["advertisingClosureStatus"] = "completed"
        state["mediaContinuationRequired"] = False
        state["status"] = "completed"
        state["lastCompletedStep"] = "done"
        save_tournament_state(job_id, state)
        report["redisMutations"] = 1

        marketing_text = str(media.get("marketingText") or "")
        overlay_headline = "" if not builder2_requires_headline_overlay(plan=plan, state=state) else str(
            plan.get("headlineText") or ""
        )
        video_job_mark_done(job_id, render_outcome.public_url, marketing_text, overlay_headline=overlay_headline)
        report["persistedCompletionAccepted"] = True
        report["jobCompleted"] = True
        report["ok"] = True
    except Builder2TournamentError as exc:
        preserve_original_failure(report, exc)
        report["failureReason"] = str(exc.args[0] if exc.args else "builder2_media_finalization_failed")
        report["failureStage"] = report.get("failureStage") or "finalization"
    except Exception as exc:
        _apply_unexpected_failure(report, exc)
    finally:
        report.update(
            {
                key: MediaFinalizationIsolationGuard.reasoning_report().get(key, report.get(key))
                for key in ("totalReasoningCalls",)
            }
        )
        _release_execution_lease_safely(
            report,
            job_id=job_id,
            worker_token=worker_token,
            lease_acquired=lease_acquired,
        )
        _finalize_isolation_guard_safely()
    return report


def print_media_finalization_resume_report(report: Dict[str, Any]) -> None:
    safe = sanitize_media_finalization_report(report)
    print(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def emit_media_finalization_resume_report(
    report: Dict[str, Any],
    *,
    job_id: str,
    preflight: bool,
) -> Dict[str, Any]:
    return emit_fail_safe_media_finalization_report(report, job_id=job_id, preflight=preflight)


def main(argv: Optional[list[str]] = None) -> int:
    from engine.builder2_media_finalization_child_diagnostics import register_child_lifecycle_diagnostics

    register_child_lifecycle_diagnostics()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID")
    preflight = _truthy("BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT")
    if not job_id:
        report = {"ok": False, "failureReason": "builder2_media_finalization_resume_job_id_missing"}
        emit_media_finalization_resume_report(report, job_id="", preflight=preflight)
        return 1

    report: Dict[str, Any] = _initial_report(job_id=job_id, preflight=preflight)
    exit_code = 1
    logger.info(
        "BUILDER2_MEDIA_FINALIZATION_RESUME_START jobId=%s preflight=%s",
        job_id,
        preflight,
    )
    try:
        report = run_one_media_finalization_resume(job_id=job_id, preflight=preflight)
        exit_code = 0 if report.get("ok") else 1
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        _apply_unexpected_failure(report, exc)
        exit_code = 1
    finally:
        try:
            emit_media_finalization_resume_report(report, job_id=job_id, preflight=preflight)
        except Exception:
            emit_fail_safe_media_finalization_report(
                build_minimal_fallback_report(job_id=job_id, preflight=preflight),
                job_id=job_id,
                preflight=preflight,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
