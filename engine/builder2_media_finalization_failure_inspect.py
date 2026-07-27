"""
Builder2 media-finalization failure inspector — read-only post-media state diagnostics.

Run:
  BUILDER2_MEDIA_FINALIZATION_FAILURE_INSPECT_JOB_ID=<jobId> \\
    python -m engine.builder2_media_finalization_failure_inspect
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from engine.builder2_advertising_closure_contract import advertising_closure_is_required
from engine.builder2_headline_decision_contract import (
    get_normalized_headline_decision,
    headline_decision_requires_headline,
)
from engine.builder2_new_format_config import (
    DEFAULT_BUILDER2_FINAL_VIDEO_DURATION_SECONDS,
    DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS,
    FINAL_DURATION_TOLERANCE_SECONDS,
    resolve_builder2_end_card_duration_seconds,
    resolve_builder2_final_video_duration_seconds,
)
from engine.builder2_read_only_inspection import read_only_builder2_inspection
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_tournament_store import _read_raw
from engine.video_jobs_redis import redis_configured, video_job_get_raw

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = frozenset(
    {
        "startImageArtifact",
        "startImageDataUri",
        "marketingText",
        "marketing_text",
        "prompt",
        "headlineText",
        "headline",
        "sloganText",
        "productNameText",
        "productDescription",
        "product_description",
        "OPENAI_API_KEY",
        "REDIS_URL",
        "RUNWAY_API_KEY",
    }
)

_CREATIVE_TEXT_FIELD_SUFFIXES = (
    "Text",
    "text",
    "Prompt",
    "prompt",
    "Slogan",
    "slogan",
    "Headline",
    "headline",
)

_INSPECTION_CALL_COUNTS = {
    "openAICalls": 0,
    "imageCalls": 0,
    "runwaySubmissionCalls": 0,
    "runwayPollingCalls": 0,
    "ffmpegCalls": 0,
    "publicationCalls": 0,
    "redisMutations": 0,
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def _runway_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    runway = state.get("runway")
    return runway if isinstance(runway, dict) else {}


def _metrics_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    metrics = state.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _call_counters(media: Dict[str, Any]) -> Dict[str, Any]:
    counters = media.get("callCounters")
    return counters if isinstance(counters, dict) else {}


def _stable_url_hash(url: str) -> Optional[str]:
    token = _clean(url)
    if not token:
        return None
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _classify_url_route_family(url: str) -> str:
    parsed = urlparse(_clean(url))
    path = (parsed.path or "").lower()
    if not path:
        return "missing"
    if "/api/video-headline-artifact" in path:
        return "api/video-headline-artifact"
    if "/api/video-headline/" in path:
        return "api/video-headline"
    if "/api/video-closure" in path or "/video-closure" in path:
        return "api/video-closure"
    host = (parsed.hostname or "").lower()
    if "runway" in host or "cloudfront" in host or "amazonaws.com" in host:
        return "runway-artifact"
    if path.endswith(".mp4") or "/video/" in path:
        return "other-video"
    return "other"


def _sanitize_url_field(value: Any) -> Dict[str, Any]:
    url = _clean(value)
    if not url:
        return {
            "keyExists": value is not None,
            "valuePresent": False,
            "scheme": None,
            "host": None,
            "routeFamily": "missing",
            "stableHash": None,
        }
    parsed = urlparse(url)
    return {
        "keyExists": True,
        "valuePresent": True,
        "scheme": parsed.scheme or None,
        "host": parsed.hostname or None,
        "routeFamily": _classify_url_route_family(url),
        "stableHash": _stable_url_hash(url),
    }


def _presence_field(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        present = bool(value.strip())
    elif value is None:
        present = False
    else:
        present = bool(value)
    return {
        "keyExists": value is not None,
        "valuePresent": present,
    }


def _compare_url_identities(left: str, right: str) -> str:
    left_clean = _clean(left)
    right_clean = _clean(right)
    if not left_clean and not right_clean:
        return "missing"
    if not left_clean or not right_clean:
        return "missing"
    if left_clean == right_clean:
        return "same"
    if _stable_url_hash(left_clean) == _stable_url_hash(right_clean):
        return "same"
    return "different"


def _first_present_url(*values: Any) -> str:
    for value in values:
        token = _clean(value)
        if token:
            return token
    return ""


def _resolve_raw_runway_url(state: Dict[str, Any], media: Dict[str, Any], runway: Dict[str, Any]) -> str:
    return _first_present_url(
        media.get("rawRunwayVideoUrl"),
        media.get("rawRunwayVideoPath"),
        media.get("runwayVideoUrl"),
        media.get("downloadedVideoPath"),
        runway.get("videoUrl"),
    )


def _resolve_downloaded_raw(media: Dict[str, Any]) -> str:
    return _first_present_url(media.get("downloadedVideoPath"), media.get("rawRunwayVideoPath"), media.get("rawRunwayVideoUrl"))


def _resolve_headline_artifact_url(
    *,
    state: Dict[str, Any],
    media: Dict[str, Any],
    job_raw: Dict[str, str],
    final_public_url: str,
    raw_url: str,
    headline_required: bool,
    duration_meta: Dict[str, Optional[float]],
    closure_url: str,
) -> str:
    explicit = _first_present_url(
        media.get("headlineArtifactUrl"),
        media.get("headlineVideoUrl"),
        media.get("headlineOverlayUrl"),
        state.get("headlineArtifactUrl"),
    )
    if explicit:
        return explicit
    if not headline_required:
        return ""
    candidate = _first_present_url(final_public_url, job_raw.get("video_url"), job_raw.get("videoUrl"))
    if not candidate:
        return ""
    if raw_url and _compare_url_identities(candidate, raw_url) == "same":
        return ""
    final_duration = duration_meta.get("finalVideoDurationSeconds")
    if (
        closure_url
        and _compare_url_identities(candidate, closure_url) == "same"
        and _approx_equal(final_duration, resolve_builder2_final_video_duration_seconds())
    ):
        return ""
    observed_duration = final_duration or duration_meta.get("rawRunwayDurationSeconds")
    if _classify_url_route_family(candidate) != "api/video-headline":
        return ""
    if observed_duration is not None and _approx_equal(
        observed_duration,
        resolve_builder2_final_video_duration_seconds(),
        tolerance=max(0.5, FINAL_DURATION_TOLERANCE_SECONDS),
    ):
        return ""
    return candidate


def _duration_metadata(media: Dict[str, Any]) -> Dict[str, Optional[float]]:
    def _float(key: str) -> Optional[float]:
        raw = media.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return {
        "rawRunwayDurationSeconds": _float("rawRunwayDurationSeconds"),
        "finalVideoDurationSeconds": _float("finalVideoDurationSeconds"),
        "endCardDurationSeconds": _float("endCardDurationSeconds"),
        "headlineOverlayDurationSeconds": _float("headlineOverlayDurationSeconds"),
    }


def _approx_equal(left: Optional[float], right: float, *, tolerance: float = FINAL_DURATION_TOLERANCE_SECONDS) -> bool:
    if left is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


def _closure_inclusive_artifact_present(
    *,
    closure_url: str,
    raw_url: str,
    headline_url: str,
    duration_meta: Dict[str, Optional[float]],
    expected_final_duration: float,
) -> bool:
    if not closure_url:
        return False
    if raw_url and _compare_url_identities(closure_url, raw_url) == "same":
        return False
    if headline_url and _compare_url_identities(closure_url, headline_url) == "same":
        return False
    if _approx_equal(duration_meta.get("finalVideoDurationSeconds"), expected_final_duration):
        return True
    if _classify_url_route_family(closure_url) == "api/video-headline" and headline_url:
        if _compare_url_identities(closure_url, headline_url) == "same":
            return False
    return _compare_url_identities(closure_url, raw_url) == "different" and bool(closure_url)


def _build_artifact_identity_graph(
    *,
    raw_url: str,
    downloaded_url: str,
    headline_url: str,
    closure_url: str,
    final_public_url: str,
    job_video_url: str,
) -> Dict[str, Any]:
    artifacts = {
        "rawRunwayArtifact": raw_url,
        "downloadedRawArtifact": downloaded_url,
        "headlineOverlayArtifact": headline_url,
        "closureInclusiveArtifact": closure_url,
        "finalPublicUrl": final_public_url,
        "jobVideoUrl": job_video_url,
    }
    keys = list(artifacts.keys())
    pairwise: Dict[str, str] = {}
    for left_key in keys:
        for right_key in keys:
            if left_key >= right_key:
                continue
            relation = _compare_url_identities(artifacts[left_key], artifacts[right_key])
            pairwise[f"{left_key}__{right_key}"] = relation

    job_done_via_headline = False
    if job_video_url and headline_url and _compare_url_identities(job_video_url, headline_url) == "same":
        job_done_via_headline = True
    elif job_video_url and closure_url and headline_url:
        if (
            _compare_url_identities(job_video_url, headline_url) == "same"
            and _compare_url_identities(job_video_url, closure_url) == "same"
            and raw_url
            and _compare_url_identities(job_video_url, raw_url) == "different"
        ):
            job_done_via_headline = True

    return {
        "artifacts": {key: _sanitize_url_field(value) for key, value in artifacts.items()},
        "pairwiseRelations": pairwise,
        "jobMarkedDoneViaHeadlineArtifact": job_done_via_headline,
        "jobVideoUrlEqualsFinalPublicUrl": _compare_url_identities(job_video_url, final_public_url),
        "jobVideoUrlEqualsFinalVideoWithClosureUrl": _compare_url_identities(job_video_url, closure_url),
        "jobVideoUrlEqualsHeadlineArtifact": _compare_url_identities(job_video_url, headline_url),
        "jobVideoUrlEqualsRawRunwayArtifact": _compare_url_identities(job_video_url, raw_url),
    }


def _pipeline_order_audit(*, headline_required: bool) -> Dict[str, Any]:
    return {
        "documentedOrderForHeadlineUse": (
            "raw_runway -> closure_render_attempt -> headline_overlay_on_downloaded_raw -> publish"
        ),
        "closureBeforeHeadlineInCode": True,
        "headlineInputInCode": "mediaResume.downloadedVideoPath (raw Runway URL)",
        "closureInputInCode": "resolve_raw_runway_video(state) (raw Runway URL first)",
        "canProduceCorrectTwelveSecondHeadlinePlusClosure": False,
        "orderingDefectSummary": (
            "Closure runs on raw Runway before headline overlay; headline overlay then runs on raw "
            "downloaded_path, not on a closure-inclusive intermediate. Even when closure succeeds, "
            "headline is not applied to the 12-second artifact."
        ),
        "continuesAfterClosureFailureBecause": (
            "append_advertising_closure_endcard catches exceptions and returns source_video_url; "
            "render_advertising_closure_for_state treats any returned URL as success and sets "
            "advertisingClosureStatus=completed without verifying a distinct closure artifact."
        ),
        "videoJobMarkDoneCalledBecause": (
            "run_one_media_resume calls video_job_mark_done whenever mediaResume.finalPublicUrl is "
            "truthy after execute_builder2_media_pipeline returns, without validating closure contract."
        ),
        "finalUrlVariableInPipeline": (
            "execute_builder2_media_pipeline local final_url; after headline overlay it becomes the "
            "headline artifact URL and is copied to finalPublicUrl and finalVideoWithClosureUrl."
        ),
        "headlineRequiredForThisAudit": headline_required,
    }


def _closure_failure_code_audit(*, media: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    failure = media.get("advertisingClosureFailure")
    media_failure = media.get("mediaFailure")
    persisted_reason = ""
    if isinstance(failure, dict):
        persisted_reason = _clean(failure.get("reason") or failure.get("failureReason"))
    elif isinstance(media_failure, dict):
        persisted_reason = _clean(media_failure.get("reason"))

    return_code_available = False
    stderr_available = False
    if isinstance(failure, dict):
        return_code_available = failure.get("returnCode") is not None or failure.get("ffmpegReturnCode") is not None
        stderr_available = bool(_clean(failure.get("stderr")) or _clean(failure.get("ffmpegStderr")))

    reconstructable = bool(
        _clean(media.get("rawRunwayVideoUrl"))
        and isinstance(state.get("advertisingClosure"), dict)
        and _clean((state.get("advertisingClosure") or {}).get("productNameText"))
        and _clean((state.get("advertisingClosure") or {}).get("sloganText"))
    )

    return {
        "raisingOrCatchingFunction": "append_advertising_closure_endcard._run via checked subprocess runner",
        "catchSiteFunction": "append_advertising_closure_endcard outer except Exception",
        "ffmpegCommandConstructionPath": (
            "append_advertising_closure_endcard builds card_cmd and concat_cmd; runner defaults to "
            "checked subprocess invocation with captured stdout/stderr"
        ),
        "stdoutStderrCapturedInCode": True,
        "stderrPersistedToState": False,
        "returnCodePersistedToState": False,
        "returnCodeLoggedInCode": False,
        "failedCommandReconstructableFromPersistedState": reconstructable,
        "observedLogReasonMatchesCodePath": persisted_reason == "CalledProcessError"
        or "CalledProcessError" in persisted_reason,
        "deterministicFailureReasons": [
            "missing_ffmpeg_binary",
            "missing_font_path",
            "missing_public_base_url",
            "source_video_download_failure",
            "ffprobe_duration_failure",
            "card_ffmpeg_called_process_error",
            "concat_ffmpeg_called_process_error",
            "output_path_token_resolution_failure",
            "upload_request_failure",
        ],
        "reasonsRuledOutFromSuppliedLogs": [
            "missing_public_base_url",
            "missing_ffmpeg_or_font",
            "source_video_download_failure",
        ],
        "unknowableWithoutPersistedStderr": [
            "exact_ffmpeg_argument_or_filter_syntax_error",
            "exact_missing_codec_or_filter_name",
            "exact_font_rendering_failure_message",
            "which_of_card_cmd_vs_concat_cmd_failed",
        ],
        "closureFailureDiagnosticAvailable": bool(persisted_reason),
        "closureFailureReturnCodeAvailable": return_code_available,
        "closureFailureStderrAvailable": stderr_available,
        "exactClosureFailureCauseKnown": False,
        "persistedClosureFailureReason": persisted_reason or None,
        "codeSwallowsCalledProcessError": True,
        "codeFallbackReturnsSourceVideoUrl": True,
    }


def _reusability_assessment(
    *,
    state: Dict[str, Any],
    media: Dict[str, Any],
    runway: Dict[str, Any],
    raw_url: str,
    headline_url: str,
    closure_valid: bool,
) -> Dict[str, Any]:
    start_image = _first_present_url(media.get("startImageArtifact"), runway.get("startImageDataUri"))
    task_id = _first_present_url(media.get("runwayTaskId"), runway.get("taskId"))
    runway_completed_url = _first_present_url(media.get("runwayVideoUrl"), raw_url)
    closure = state.get("advertisingClosure")
    closure_dict = closure if isinstance(closure, dict) else {}

    start_image_reusable = bool(start_image)
    runway_submission_reusable = bool(task_id)
    runway_polling_skippable = bool(runway_completed_url)
    raw_reusable = bool(raw_url)
    headline_reusable = bool(headline_url)
    closure_reusable = bool(closure_dict.get("sloganText"))

    recovery_requires_ffmpeg = not closure_valid
    recovery_requires_publication = not closure_valid

    return {
        "startImageCanBeReused": start_image_reusable,
        "runwaySubmissionCanBeReused": runway_submission_reusable,
        "runwayPollingCanBeSkipped": runway_polling_skippable,
        "rawRunwayArtifactCanBeReused": raw_reusable,
        "headlineArtifactCanBeReused": headline_reusable,
        "advertisingClosureCanBeReused": closure_reusable,
        "recoveryRequiresOpenAI": False,
        "recoveryRequiresRunwaySubmission": not runway_submission_reusable,
        "recoveryRequiresRunwayPolling": not runway_polling_skippable,
        "recoveryRequiresFFmpeg": recovery_requires_ffmpeg,
        "recoveryRequiresPublication": recovery_requires_publication,
        "additionalPaidGenerationRequired": not runway_submission_reusable or not start_image_reusable,
    }


def _duration_inspection_feasibility(*, media: Dict[str, Any]) -> Dict[str, Any]:
    duration_meta = _duration_metadata(media)
    return {
        "executesFfprobeInInspector": False,
        "executesNetworkMediaDownloadInInspector": False,
        "persistedDurationMetadataAvailable": any(value is not None for value in duration_meta.values()),
        "persistedDurationMetadata": duration_meta,
        "safestLaterMechanisms": {
            "rawVideoDurationApproxTenSeconds": [
                "read mediaResume.rawRunwayDurationSeconds if persisted",
                "offline ffprobe against persisted local path if ever stored",
                "signed HEAD/range against public URL only in dedicated probe job",
            ],
            "headlineArtifactDurationApproxTenSeconds": [
                "compare job video_url route family api/video-headline with rawRunwayDurationSeconds metadata",
                "offline ffprobe against published headline artifact URL in dedicated probe job",
            ],
            "finalClosureArtifactDurationApproxTwelveSeconds": [
                "read mediaResume.finalVideoDurationSeconds if persisted and trustworthy",
                "offline ffprobe against finalVideoWithClosureUrl only after contract fix ensures true closure artifact",
            ],
        },
    }


def _recommended_correction_surface() -> Dict[str, Any]:
    return {
        "files": [
            "engine/video_endcard_postprocess.py",
            "engine/builder2_advertising_closure_pipeline.py",
            "engine/builder2_media_pipeline.py",
            "engine/builder2_media_resume.py",
            "engine/video_jobs_redis.py",
        ],
        "functions": [
            "append_advertising_closure_endcard",
            "render_advertising_closure_for_state",
            "execute_builder2_media_pipeline",
            "run_one_media_resume",
            "video_job_mark_done",
        ],
        "requiredGuarantees": [
            "closure_render_failure_is_fatal",
            "finalPublicUrl_never_headline_only_when_closure_required",
            "finalVideoWithClosureUrl_only_after_successful_closure_publish",
            "video_job_mark_done_only_with_closure_inclusive_url",
            "headline_then_closure_or_closure_on_headline_backed_intermediate_for_twelve_second_final",
            "preserve_raw_and_intermediate_artifacts",
            "zero_openai_zero_runway_recovery",
            "idempotent_recovery",
            "reopen_completed_invalid_without_rerunning_prior_media_stages",
            "builder1_unchanged",
        ],
    }


def inspect_builder2_media_finalization_failure(
    job_id: str,
    *,
    raw_state_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    raw_job_reader: Optional[Callable[[str], Optional[Dict[str, str]]]] = None,
) -> Dict[str, Any]:
    read_state = raw_state_reader or _read_raw
    read_job = raw_job_reader or video_job_get_raw
    jid = _clean(job_id)
    report: Dict[str, Any] = {
        "ok": False,
        "inspectionCompleted": False,
        "jobId": jid or None,
        "redisMutations": 0,
        "persistedCompletionStatus": None,
        "effectiveCompletionStatus": None,
        "falseCompletionDetected": False,
        "falseCompletionReasons": [],
        "artifactIdentityGraph": {},
        "completionContractAudit": {},
        "pipelineOrder": {},
        "closureFailureInvestigation": {},
        "durationInspectionFeasibility": {},
        "recommendedCorrectionSurface": _recommended_correction_surface(),
        "persistedState": {},
        "jobDeliveryHash": {},
        "closureFailureDiagnosticAvailable": False,
        "closureFailureReturnCodeAvailable": False,
        "closureFailureStderrAvailable": False,
        "exactClosureFailureCauseKnown": False,
        "startImageCanBeReused": False,
        "runwaySubmissionCanBeReused": False,
        "runwayPollingCanBeSkipped": False,
        "rawRunwayArtifactCanBeReused": False,
        "headlineArtifactCanBeReused": False,
        "advertisingClosureCanBeReused": False,
        "recoveryRequiresOpenAI": False,
        "recoveryRequiresRunwaySubmission": True,
        "recoveryRequiresRunwayPolling": True,
        "recoveryRequiresFFmpeg": True,
        "recoveryRequiresPublication": True,
        "additionalPaidGenerationRequired": True,
        "inspectionCallCounts": dict(_INSPECTION_CALL_COUNTS),
    }

    if not jid:
        report["failureReason"] = "builder2_media_finalization_failure_inspect_job_id_missing"
        return report
    if not redis_configured():
        report["failureReason"] = "builder2_media_finalization_failure_inspect_redis_unconfigured"
        return report

    with read_only_builder2_inspection() as mutation_counter:
        raw_state = read_state(jid)
        if raw_state is None:
            report["failureReason"] = "builder2_media_finalization_failure_inspect_job_not_found"
            return report
        state = deepcopy(raw_state)
        job_raw = read_job(jid) or {}
        media = _media_bucket(state)
        runway = _runway_bucket(state)
        counters = _call_counters(media)
        metrics = _metrics_bucket(state)
        plan = state.get("winnerDevelopmentPlan") if isinstance(state.get("winnerDevelopmentPlan"), dict) else {}
        headline_decision = get_normalized_headline_decision(plan)
        headline_required = headline_decision_requires_headline(headline_decision)
        closure_required = advertising_closure_is_required(plan) or bool(
            _clean((state.get("advertisingClosure") or {}).get("sloganText"))
        )

        raw_url = _resolve_raw_runway_url(state, media, runway)
        downloaded_url = _resolve_downloaded_raw(media)
        final_public_url = _first_present_url(media.get("finalPublicUrl"), media.get("finalVideoPath"))
        closure_url = _first_present_url(media.get("finalVideoWithClosureUrl"))
        job_video_url = _first_present_url(job_raw.get("video_url"), job_raw.get("videoUrl"))
        duration_meta = _duration_metadata(media)
        headline_url = _resolve_headline_artifact_url(
            state=state,
            media=media,
            job_raw=job_raw,
            final_public_url=final_public_url or job_video_url,
            raw_url=raw_url,
            headline_required=headline_required,
            duration_meta=duration_meta,
            closure_url=closure_url,
        )

        expected_final_duration = resolve_builder2_final_video_duration_seconds()
        closure_rendered_flag = bool(_clean(media.get("closureRenderedAt"))) or _clean(media.get("advertisingClosureStatus")) == "completed"
        closure_inclusive_present = _closure_inclusive_artifact_present(
            closure_url=closure_url,
            raw_url=raw_url,
            headline_url=headline_url,
            duration_meta=duration_meta,
            expected_final_duration=expected_final_duration,
        )

        media_failure = media.get("mediaFailure")
        media_failure_dict = media_failure if isinstance(media_failure, dict) else {}
        report["persistedState"] = {
            "status": _clean(state.get("status")) or None,
            "failureStage": _clean(state.get("failureStage") or media_failure_dict.get("stage")) or None,
            "failureReason": _clean(state.get("failureReason") or media_failure_dict.get("reason")) or None,
            "mediaContinuationRequired": state.get("mediaContinuationRequired"),
            "reasoningComplete": bool(state.get("reasoningComplete")),
            "winnerDevelopmentAccepted": is_valid_persisted_winner_development(state),
            "headlineDecision": headline_decision,
            "mediaResumeStatus": _clean(media.get("mediaResumeStatus")) or None,
            "progressStage": _clean(media.get("progressStage")) or None,
            "finalPublicUrl": _sanitize_url_field(media.get("finalPublicUrl")),
            "finalVideoWithClosureUrl": _sanitize_url_field(media.get("finalVideoWithClosureUrl")),
            "finalVideoPath": _sanitize_url_field(media.get("finalVideoPath")),
            "advertisingClosureStatus": _clean(state.get("advertisingClosureStatus") or media.get("advertisingClosureStatus")) or None,
            "advertisingClosureRendered": closure_rendered_flag,
            "advertisingClosureFailureReason": _clean(
                (media.get("advertisingClosureFailure") or {}).get("reason")
                if isinstance(media.get("advertisingClosureFailure"), dict)
                else ""
            )
            or None,
            "headlinePostprocessStatus": _clean(media.get("headlinePostprocessStatus") or media.get("ffmpegStatus")) or None,
            "headlineArtifactUrl": _sanitize_url_field(
                media.get("headlineArtifactUrl") or media.get("headlineVideoUrl") or headline_url
            ),
            "headlineArtifactPath": _presence_field(media.get("headlineArtifactPath")),
            "startImageStatus": _clean(media.get("startImageStatus")) or None,
            "startImageArtifact": _presence_field(media.get("startImageArtifact")),
            "startImageDataUri": _presence_field(runway.get("startImageDataUri")),
            "startImageCalls": int(counters.get("startImageCalls") or metrics.get("startImageCalls") or 0),
            "startImageGeneratedCount": int(media.get("startImageGeneratedCount") or counters.get("startImageGeneratedCount") or 0),
            "runwayTaskId": _presence_field(media.get("runwayTaskId") or runway.get("taskId")),
            "runwaySubmissionStatus": _clean(media.get("runwaySubmissionStatus")) or None,
            "runwayStatus": _clean(media.get("runwayStatus")) or None,
            "runwayVideoUrl": _sanitize_url_field(media.get("runwayVideoUrl")),
            "rawRunwayVideoUrl": _sanitize_url_field(media.get("rawRunwayVideoUrl")),
            "rawRunwayVideoPath": _sanitize_url_field(media.get("rawRunwayVideoPath")),
            "downloadedVideoPath": _sanitize_url_field(media.get("downloadedVideoPath")),
            "runwaySubmissionCalls": int(counters.get("runwaySubmissionCalls") or metrics.get("runwaySubmissionCalls") or 0),
            "runwayTaskCreatedCount": int(counters.get("runwayTaskCreatedCount") or 0),
            "runwayPollingCalls": int(counters.get("runwayPollingCalls") or 0),
            "durationMetadata": duration_meta,
        }

        report["jobDeliveryHash"] = {
            "jobStatus": _clean(job_raw.get("status")) or None,
            "videoUrl": _sanitize_url_field(job_raw.get("video_url") or job_raw.get("videoUrl")),
            "videoUrlEqualsFinalPublicUrl": _compare_url_identities(job_video_url, final_public_url),
            "videoUrlEqualsFinalVideoWithClosureUrl": _compare_url_identities(job_video_url, closure_url),
            "videoUrlEqualsHeadlineArtifactUrl": _compare_url_identities(job_video_url, headline_url),
            "videoUrlEqualsRawRunwayArtifact": _compare_url_identities(job_video_url, raw_url),
        }

        artifact_graph = _build_artifact_identity_graph(
            raw_url=raw_url,
            downloaded_url=downloaded_url,
            headline_url=headline_url,
            closure_url=closure_url,
            final_public_url=final_public_url,
            job_video_url=job_video_url,
        )
        report["artifactIdentityGraph"] = artifact_graph

        runway_present = bool(raw_url)
        headline_present = bool(headline_url)
        job_points_to_closure = _compare_url_identities(job_video_url, closure_url) == "same" and closure_inclusive_present
        observed_duration_present = any(value is not None for value in duration_meta.values())

        false_reasons: List[str] = []
        if closure_required and not closure_inclusive_present:
            false_reasons.append("closure_required_but_closure_inclusive_artifact_missing_or_invalid")
        if closure_required and closure_rendered_flag and not closure_inclusive_present:
            false_reasons.append("advertising_closure_marked_rendered_without_distinct_closure_artifact")
        if headline_required and not headline_present:
            false_reasons.append("headline_required_but_headline_artifact_not_identified")
        if job_video_url and headline_url and _compare_url_identities(job_video_url, headline_url) == "same" and closure_required:
            if not closure_inclusive_present or _compare_url_identities(job_video_url, closure_url) != "same" or not closure_inclusive_present:
                false_reasons.append("job_video_url_points_to_headline_artifact_not_closure_inclusive_final")
        if closure_url and raw_url and _compare_url_identities(closure_url, raw_url) == "same" and closure_required:
            false_reasons.append("finalVideoWithClosureUrl_matches_raw_runway_artifact")
        if closure_url and headline_url and _compare_url_identities(closure_url, headline_url) == "same" and closure_required:
            false_reasons.append("finalVideoWithClosureUrl_matches_headline_artifact_only")
        if _approx_equal(duration_meta.get("finalVideoDurationSeconds"), expected_final_duration) is False and closure_required:
            if duration_meta.get("finalVideoDurationSeconds") is not None:
                false_reasons.append("final_video_duration_metadata_not_twelve_seconds")
        if _clean(state.get("status")) == "completed" and false_reasons:
            false_reasons.append("persisted_status_completed_despite_contract_violation")

        completion_contract_satisfied = (
            runway_present
            and (not headline_required or headline_present)
            and (not closure_required or closure_inclusive_present)
            and bool(closure_url)
            and job_points_to_closure
        )

        report["completionContractAudit"] = {
            "runwayArtifactPresent": runway_present,
            "headlineRequired": headline_required,
            "headlineArtifactPresent": headline_present,
            "closureRequired": closure_required,
            "closureRendered": closure_rendered_flag,
            "closureInclusiveArtifactPresent": closure_inclusive_present,
            "finalVideoWithClosureUrlPresent": bool(closure_url),
            "jobVideoUrlPointsToClosureArtifact": job_points_to_closure,
            "expectedFinalDurationSeconds": expected_final_duration,
            "expectedRunwayDurationSeconds": DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS,
            "expectedEndCardDurationSeconds": resolve_builder2_end_card_duration_seconds(),
            "observedDurationMetadataPresent": observed_duration_present,
            "completionContractSatisfied": completion_contract_satisfied,
        }

        persisted_status = _clean(state.get("status")) or _clean(media.get("mediaResumeStatus")) or None
        effective_status = "completed" if completion_contract_satisfied else "incomplete"
        report["persistedCompletionStatus"] = persisted_status
        report["effectiveCompletionStatus"] = effective_status
        report["falseCompletionDetected"] = persisted_status == "completed" and effective_status != "completed"
        report["falseCompletionReasons"] = false_reasons

        closure_audit = _closure_failure_code_audit(media=media, state=state)
        report["closureFailureInvestigation"] = closure_audit
        report["closureFailureDiagnosticAvailable"] = bool(closure_audit["closureFailureDiagnosticAvailable"])
        report["closureFailureReturnCodeAvailable"] = bool(closure_audit["closureFailureReturnCodeAvailable"])
        report["closureFailureStderrAvailable"] = bool(closure_audit["closureFailureStderrAvailable"])
        report["exactClosureFailureCauseKnown"] = bool(closure_audit["exactClosureFailureCauseKnown"])

        report["pipelineOrder"] = _pipeline_order_audit(headline_required=headline_required)
        report["durationInspectionFeasibility"] = _duration_inspection_feasibility(media=media)

        reuse = _reusability_assessment(
            state=state,
            media=media,
            runway=runway,
            raw_url=raw_url,
            headline_url=headline_url,
            closure_valid=closure_inclusive_present,
        )
        report.update(reuse)

        report["redisMutations"] = mutation_counter.redis_mutations
        report["inspectionCallCounts"]["redisMutations"] = mutation_counter.redis_mutations
        report["inspectionCompleted"] = True
        report["ok"] = True
        return report


def _sanitize_report_for_output(report: Dict[str, Any]) -> Dict[str, Any]:
    def _walk(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: _walk(v, k) for k, v in value.items() if k not in _SENSITIVE_KEYS}
        if isinstance(value, list):
            return [_walk(item, key) for item in value]
        if isinstance(value, str):
            lowered_key = key.lower()
            if any(marker in lowered_key for marker in _CREATIVE_TEXT_FIELD_SUFFIXES):
                return {"redacted": True, "characterCount": len(value)}
            if value.startswith("data:"):
                return {"redacted": True, "valueType": "data_uri"}
            if len(value) > 256 and "http" in value:
                return {
                    "redacted": True,
                    "routeFamily": _classify_url_route_family(value),
                    "stableHash": _stable_url_hash(value),
                }
        return value

    return _walk(report)


def print_builder2_media_finalization_failure_report(report: Dict[str, Any]) -> None:
    safe = _sanitize_report_for_output(report)
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _clean(os.environ.get("BUILDER2_MEDIA_FINALIZATION_FAILURE_INSPECT_JOB_ID"))
    if not job_id:
        print(
            json.dumps(
                {"ok": False, "failureReason": "builder2_media_finalization_failure_inspect_job_id_missing"},
                indent=2,
            )
        )
        return 1
    logger.info("BUILDER2_MEDIA_FINALIZATION_FAILURE_INSPECT_START jobId=%s", job_id)
    report = inspect_builder2_media_finalization_failure(job_id)
    print_builder2_media_finalization_failure_report(report)
    logger.info(
        "BUILDER2_MEDIA_FINALIZATION_FAILURE_INSPECT_DONE jobId=%s ok=%s falseCompletion=%s effective=%s persisted=%s",
        job_id,
        report.get("ok"),
        report.get("falseCompletionDetected"),
        report.get("effectiveCompletionStatus"),
        report.get("persistedCompletionStatus"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
