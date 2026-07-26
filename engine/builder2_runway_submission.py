"""
Builder2 Runway submission helpers — URL validation, prompt budget, safe metadata.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests

from engine.builder2_runway_config import BUILDER2_RUNWAY_VIDEO_RATIO, builder2_runway_requires_start_image
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.runway_api_urls import (
    RunwayUrlConfigurationError,
    RunwayUrlResolution,
    build_runway_image_to_video_url,
    build_runway_task_poll_url,
    build_runway_text_to_video_url,
    resolve_configured_runway_base,
    validate_runway_api_url,
)
from engine.runway_prompt_budget import (
    RunwayPromptBudgetError,
    RunwayPromptBudgetResult,
    count_utf16_code_units,
    prepare_builder2_runway_prompt_text,
)
from engine.runway_video import RUNWAY_API_VERSION_HEADER, _HTTP_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class Builder2RunwaySubmissionResult:
    task_id: Optional[str] = None
    request_submitted: bool = False
    task_created: bool = False
    http_status: Optional[int] = None
    create_url: Optional[RunwayUrlResolution] = None
    prompt: Optional[RunwayPromptBudgetResult] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _runway_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Runway-Version": RUNWAY_API_VERSION_HEADER,
    }


def audit_media_resume_start_image(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume") if isinstance(state.get("mediaResume"), dict) else {}
    runway = state.get("runway") if isinstance(state.get("runway"), dict) else {}
    artifact = str(media.get("startImageArtifact") or runway.get("startImageDataUri") or "").strip()
    output_w = media.get("startImageOutputWidth")
    output_h = media.get("startImageOutputHeight")
    status = str(media.get("startImageStatus") or "").strip()
    available = bool(artifact)
    reusable = available and status in {"completed", "reused"}
    if output_w == 1280 and output_h == 720:
        reusable = reusable or available
    return {
        "startImageAvailable": available,
        "startImageReusable": reusable,
        "startImageOutputSize": "1280x720" if (output_w, output_h) == (1280, 720) else media.get("startImageOutputSize"),
        "startImageStatus": status or None,
        "startImageOutputWidth": output_w,
        "startImageOutputHeight": output_h,
    }


def build_builder2_runway_dry_run_report(
    *,
    plan: Dict[str, Any],
    state: Dict[str, Any],
    runway_model: str,
    duration_seconds: int,
    ratio: str,
) -> Dict[str, Any]:
    start_image = audit_media_resume_start_image(state)
    _, configured_base = resolve_configured_runway_base()
    origin_configured = bool(configured_base)
    create_path = "/v1/image_to_video" if builder2_runway_requires_start_image(runway_model) else "/v1/text_to_video"
    try:
        if builder2_runway_requires_start_image(runway_model):
            create_resolution = build_runway_image_to_video_url()
        else:
            create_resolution = build_runway_text_to_video_url()
        validate_runway_api_url(create_resolution)
        poll_resolution = build_runway_task_poll_url("dry-run-task-id")
        validate_runway_api_url(poll_resolution)
        prompt = prepare_builder2_runway_prompt_text(plan)
        runway_endpoint_accepted = True
        runway_prompt_accepted = prompt.utf16Length <= prompt.maximumUtf16Units
        version_prefix_count = create_resolution.normalizedPath.count("/v1")
    except RunwayUrlConfigurationError as exc:
        raise Builder2TournamentError(str(exc)) from exc
    except RunwayPromptBudgetError as exc:
        raise Builder2TournamentError(str(exc)) from exc

    return {
        **start_image,
        "runwayOriginConfigured": origin_configured,
        "runwayCreateEndpointPath": create_path,
        "runwayTaskEndpointTemplate": "/v1/tasks/{taskId}",
        "runwayVersionPrefixCount": version_prefix_count,
        "runwayEndpointAccepted": runway_endpoint_accepted,
        **prompt.to_safe_metadata(),
        "runwayModel": runway_model,
        "ratio": ratio,
        "durationSeconds": duration_seconds,
        "readyForMediaResume": runway_endpoint_accepted and runway_prompt_accepted,
    }


def submit_builder2_runway_task(
    *,
    session: requests.Session,
    api_key: str,
    plan: Dict[str, Any],
    runway_model: str,
    duration_seconds: int,
    prompt_image_data_uri: str = "",
) -> Builder2RunwaySubmissionResult:
    result = Builder2RunwaySubmissionResult()
    try:
        prompt = prepare_builder2_runway_prompt_text(plan)
    except RunwayPromptBudgetError as exc:
        result.failure_stage = "runway_configuration"
        result.failure_reason = str(exc)
        raise Builder2TournamentError(str(exc)) from exc
    result.prompt = prompt

    try:
        if builder2_runway_requires_start_image(runway_model):
            create_resolution = build_runway_image_to_video_url()
        else:
            create_resolution = build_runway_text_to_video_url()
        validate_runway_api_url(create_resolution)
    except RunwayUrlConfigurationError as exc:
        result.failure_stage = "runway_configuration"
        result.failure_reason = str(exc)
        raise Builder2TournamentError(str(exc)) from exc
    result.create_url = create_resolution

    image_uri = (prompt_image_data_uri or "").strip()
    if builder2_runway_requires_start_image(runway_model) and not image_uri:
        raise Builder2TournamentError("builder2_media_start_image_failed")

    body: Dict[str, Any] = {
        "model": runway_model,
        "promptText": prompt.promptText,
        "ratio": BUILDER2_RUNWAY_VIDEO_RATIO,
        "duration": duration_seconds,
    }
    if builder2_runway_requires_start_image(runway_model):
        body["promptImage"] = image_uri

    result.request_submitted = True
    result.metadata = {
        "requestSubmitted": True,
        "taskCreated": False,
        "httpStatus": None,
        "normalizedEndpointPath": create_resolution.normalizedPath,
        "configuredBaseSource": create_resolution.configuredBaseSource,
        "runwayPromptUtf16Length": prompt.utf16Length,
        "runwayModel": runway_model,
        "ratio": BUILDER2_RUNWAY_VIDEO_RATIO,
        "durationSeconds": duration_seconds,
    }
    resp = session.post(
        create_resolution.absoluteUrl,
        json=body,
        headers=_runway_headers(api_key),
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    result.http_status = resp.status_code
    result.metadata["httpStatus"] = resp.status_code
    if resp.status_code >= 400:
        result.failure_stage = "runway_submission"
        result.failure_reason = "builder2_runway_submission_http_error"
        logger.error(
            "BUILDER2_RUNWAY_SUBMISSION_HTTP_ERROR status=%s path=%s utf16=%s",
            resp.status_code,
            create_resolution.normalizedPath,
            prompt.utf16Length,
        )
        exc = Builder2TournamentError(result.failure_reason)
        exc.failure_stage = result.failure_stage  # type: ignore[attr-defined]
        exc.runway_submission_result = result  # type: ignore[attr-defined]
        raise exc from None
    try:
        data = resp.json()
    except ValueError as exc:
        result.failure_stage = "runway_submission"
        result.failure_reason = "builder2_runway_submission_http_error"
        err = Builder2TournamentError(result.failure_reason)
        err.failure_stage = result.failure_stage  # type: ignore[attr-defined]
        err.runway_submission_result = result  # type: ignore[attr-defined]
        raise err from exc
    task_id = str((data or {}).get("id") or "").strip()
    if not task_id:
        result.failure_stage = "runway_submission"
        result.failure_reason = "builder2_runway_submission_http_error"
        err = Builder2TournamentError(result.failure_reason)
        err.failure_stage = result.failure_stage  # type: ignore[attr-defined]
        err.runway_submission_result = result  # type: ignore[attr-defined]
        raise err
    result.task_id = task_id
    result.task_created = True
    result.metadata["taskCreated"] = True
    result.metadata["runwayTaskId"] = task_id
    return result
