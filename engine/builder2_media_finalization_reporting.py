"""
Fail-safe CLI reporting for Builder2 media finalization resume.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger("engine.builder2_media_finalization_resume")

MEDIA_FINALIZATION_REPORT_SAFE_KEYS: tuple[str, ...] = (
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
    "originalFailureClass",
    "originalFailureStage",
    "originalFailureCode",
    "reportingFailureClass",
    "reportingFailureOccurred",
    "leaseReleaseAttempted",
    "leaseReleaseAccepted",
    "cliReportConstructionAccepted",
    "cliJsonSerializationAccepted",
    "cliStdoutWriteAccepted",
    "cliDoneLogAttempted",
    "safeFfmpegReturnCode",
    "safeFfmpegStderrAvailable",
    "openAICalls",
    "imageCalls",
    "runwaySubmissionCalls",
    "runwayPollingCalls",
    "headlineFfmpegCalls",
    "closureFfmpegCalls",
    "headlineRenderAttempts",
    "headlineFfmpegSubprocessCalls",
    "closureRenderAttempts",
    "closureFfmpegSubprocessCalls",
    "ffprobeCalls",
    "rawRunwayFfprobeCalls",
    "headlineFfprobeCalls",
    "finalClosureFfprobeCalls",
    "totalFfprobeSubprocessCalls",
    "closureFfmpegExecutionAccepted",
    "closureOutputFileCreated",
    "closureOutputFileSizeBytes",
    "closureDurationProbeAttempted",
    "closureDurationProbeAccepted",
    "measuredClosureOutputDurationSeconds",
    "measuredClosureSourceDurationSeconds",
    "configuredVisualDurationSeconds",
    "configuredEndCardDurationSeconds",
    "effectiveClosureSegmentDurationSeconds",
    "configuredFinalDurationSeconds",
    "calculatedExpectedFinalDurationSeconds",
    "actualClosureGainSeconds",
    "closureGainAccepted",
    "acceptedFinalDurationLowerBoundSeconds",
    "acceptedFinalDurationUpperBoundSeconds",
    "finalDurationDeltaSeconds",
    "finalDurationVerificationFailureCode",
    "closureFailureSubstage",
    "totalFfmpegCalls",
    "ffmpegCalls",
    "publicationCalls",
    "redisMutations",
    "finalizationReused",
    "jobCompleted",
    "totalReasoningCalls",
    "acceptedHeadlineDecision",
    "acceptedHeadlineFieldPresent",
    "acceptedHeadlineKeywordPresent",
    "persistedHeadlineTextPresent",
    "canonicalHeadlineResolutionAttempted",
    "canonicalHeadlineResolutionAccepted",
    "canonicalHeadlineSource",
    "canonicalHeadlineCharacterCount",
    "canonicalHeadlineWordCount",
    "localHeadlineInputPresent",
    "localHeadlineFailureStage",
    "localHeadlineFailureCode",
)


def json_safe_value(value: Any, *, _seen: Optional[set[int]] = None) -> Any:
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Mapping):
        obj_id = id(value)
        if obj_id in _seen:
            return None
        _seen.add(obj_id)
        return {str(key): json_safe_value(item, _seen=_seen) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in _seen:
            return None
        _seen.add(obj_id)
        return [json_safe_value(item, _seen=_seen) for item in value]
    if isinstance(value, Path):
        return "Path"
    if isinstance(value, BaseException):
        return type(value).__name__
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return type(value).__name__
    return type(value).__name__


def sanitize_media_finalization_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key in MEDIA_FINALIZATION_REPORT_SAFE_KEYS:
        if key not in report:
            continue
        safe[key] = json_safe_value(report.get(key))
    return safe


def preserve_original_failure(report: Dict[str, Any], exc: BaseException) -> None:
    if report.get("originalFailureCode"):
        return
    report["originalFailureClass"] = type(exc).__name__
    from engine.builder2_closure_render import Builder2ClosureRenderError

    if isinstance(exc, Builder2ClosureRenderError):
        report["originalFailureStage"] = exc.stage
        report["originalFailureCode"] = str(exc.args[0] if exc.args else "")
        report["failureStage"] = report.get("failureStage") or exc.stage
        report["failureReason"] = report.get("failureReason") or report["originalFailureCode"]
        return
    from engine.builder2_tournament_contracts import Builder2TournamentError

    if isinstance(exc, Builder2TournamentError):
        code = str(exc.args[0] if exc.args else "builder2_media_finalization_failed")
        report["originalFailureStage"] = report.get("failureStage") or "finalization"
        report["originalFailureCode"] = code
        report["failureStage"] = report.get("failureStage") or report["originalFailureStage"]
        report["failureReason"] = report.get("failureReason") or code
        return
    report["originalFailureStage"] = report.get("failureStage") or "internal"
    report["originalFailureCode"] = report.get("failureReason") or "builder2_media_finalization_unexpected_internal_error"


def build_minimal_fallback_report(
    *,
    job_id: str,
    preflight: bool,
    failure_reason: str = "final_report_serialization_failed",
) -> Dict[str, Any]:
    return {
        "jobId": job_id,
        "ok": False,
        "preflight": preflight,
        "readyForFinalizationRecovery": False,
        "failureStage": "cli_reporting",
        "failureReason": failure_reason,
        "reportingFailureOccurred": True,
        "cliReportConstructionAccepted": False,
        "cliJsonSerializationAccepted": False,
        "cliStdoutWriteAccepted": False,
        "cliDoneLogAttempted": False,
        "leaseReleaseAttempted": False,
        "leaseReleaseAccepted": False,
    }


def _write_stdout(payload: str) -> bool:
    try:
        sys.stdout.write(payload)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return True
    except Exception:
        try:
            encoded = payload.encode("utf-8", errors="replace")
            buffer = getattr(sys.stdout, "buffer", None)
            if buffer is not None:
                buffer.write(encoded)
                buffer.write(b"\n")
                buffer.flush()
                return True
        except Exception:
            return False
    return False


def emit_fail_safe_media_finalization_report(
    report: Mapping[str, Any],
    *,
    job_id: str,
    preflight: bool,
) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "cliReportConstructionAccepted": False,
        "cliJsonSerializationAccepted": False,
        "cliStdoutWriteAccepted": False,
        "cliDoneLogAttempted": False,
        "reportingFailureOccurred": False,
        "reportingFailureClass": None,
    }
    payload = build_minimal_fallback_report(job_id=job_id, preflight=preflight)
    try:
        payload = sanitize_media_finalization_report(report)
        payload["jobId"] = payload.get("jobId") or job_id
        payload["preflight"] = bool(payload.get("preflight", preflight))
        status["cliReportConstructionAccepted"] = True
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        status["cliJsonSerializationAccepted"] = True
    except Exception as exc:
        status["reportingFailureOccurred"] = True
        status["reportingFailureClass"] = type(exc).__name__
        payload = build_minimal_fallback_report(job_id=job_id, preflight=preflight)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        status["cliJsonSerializationAccepted"] = True

    status["cliStdoutWriteAccepted"] = _write_stdout(serialized)
    if not status["cliStdoutWriteAccepted"]:
        status["reportingFailureOccurred"] = True
        status["reportingFailureClass"] = status.get("reportingFailureClass") or "stdout_write_failed"

    try:
        logger.info(
            "BUILDER2_MEDIA_FINALIZATION_RESUME_DONE jobId=%s ok=%s preflight=%s",
            job_id,
            payload.get("ok"),
            preflight,
        )
        status["cliDoneLogAttempted"] = True
    except Exception as exc:
        status["reportingFailureOccurred"] = True
        status["reportingFailureClass"] = type(exc).__name__
        status["cliDoneLogAttempted"] = False

    payload.update(status)
    return payload
