"""
Builder2 media-only pipeline — persisted Winner plan to delivery artifacts.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from engine.builder2_headline_decision_contract import get_normalized_headline_decision, headline_decision_requires_headline
from engine.builder2_media_resume_config import MediaResumeConfiguration
from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
from engine.builder2_runway_config import (
    BUILDER2_RUNWAY_VIDEO_RATIO,
    builder2_runway_requires_start_image,
    resolve_builder2_runway_video_model,
    resolve_builder2_video_duration_seconds,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import patch_tournament_state
from engine.builder2_winner_downstream import Builder2WinnerDownstreamError, validate_builder2_pre_runway
from engine.video_planning import (
    RUNWAY_PHYSICS_REALISM_CONSTRAINT,
    build_runway_prompt_from_plan,
    sanitize_runway_prompt_for_video_text_policy,
)

logger = logging.getLogger(__name__)

MEDIA_PROGRESS_STAGES = (
    "preparing_start_image",
    "generating_start_image",
    "submitting_runway",
    "waiting_for_runway",
    "downloading_video",
    "postprocessing_video",
    "finalizing_delivery",
    "completed",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MediaPipelineCounters:
    start_image_calls: int = 0
    runway_submission_calls: int = 0
    runway_polling_resumed: bool = False
    ffmpeg_calls: int = 0
    media_reused: bool = False


@dataclass
class MediaPipelineDeps:
    generate_start_image: Callable[..., str]
    submit_runway_task: Callable[..., str]
    poll_runway_task: Callable[..., Tuple[str, str]]
    postprocess_video: Callable[..., str]
    compose_marketing_copy: Callable[..., str]


def _env_runway_api_key() -> str:
    return (os.environ.get("RUNWAY_API_KEY") or "").strip()


def _env_runway_base_url() -> str:
    return (os.environ.get("RUNWAY_API_BASE") or "https://api.dev.runwayml.com/v1").strip()


def _default_generate_start_image(plan: Dict[str, Any]) -> str:
    from engine.video_start_image import generate_video_start_image_data_uri

    return generate_video_start_image_data_uri(plan) or ""


def _default_submit_runway_task(
    *,
    plan: Dict[str, Any],
    prompt_image_data_uri: str,
    runway_model: str,
    duration_seconds: int,
) -> str:
    from engine.runway_video import _create_image_to_video_task, _create_text_to_video_task

    session = requests.Session()
    base = _env_runway_base_url()
    prompt = build_runway_prompt_from_plan(plan)
    prompt, _ = sanitize_runway_prompt_for_video_text_policy(prompt)
    prompt = f"{prompt.rstrip()} {RUNWAY_PHYSICS_REALISM_CONSTRAINT}".strip()
    if builder2_runway_requires_start_image(runway_model):
        return _create_image_to_video_task(
            session,
            base,
            runway_model,
            prompt,
            prompt_image_data_uri,
            duration_seconds=duration_seconds,
            sanitize_text_policy_modified=False,
            physics_suffix_appended=True,
        )
    return _create_text_to_video_task(
        session,
        base,
        runway_model,
        prompt,
        duration_seconds=duration_seconds,
        sanitize_text_policy_modified=False,
        physics_suffix_appended=True,
    )


def _default_poll_runway_task(*, task_id: str) -> Tuple[str, str]:
    from engine.runway_video import (
        _MAX_WAIT_SECONDS,
        _POLL_HTTP_TIMEOUT_SECONDS,
        _SUCCESS_STATUSES,
        _extract_video_url,
        _normalize_task_status,
        _poll_get_task_once,
        _sleep_poll_interval,
    )

    session = requests.Session()
    base = _env_runway_base_url()
    poll_start = time.monotonic()
    deadline = poll_start + _MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        task = _poll_get_task_once(session, base, task_id, _POLL_HTTP_TIMEOUT_SECONDS)
        if task is None:
            if time.monotonic() >= deadline:
                break
            _sleep_poll_interval(deadline)
            continue
        status = _normalize_task_status(task)
        if status in _SUCCESS_STATUSES:
            url = _extract_video_url(task)
            if url:
                return status, url
        if time.monotonic() >= deadline:
            break
        _sleep_poll_interval(deadline)
    raise Builder2TournamentError("builder2_media_runway_poll_timeout")


def _default_postprocess_video(
    *,
    runway_url: str,
    public_base_url: str,
    plan: Dict[str, Any],
    job_id: str,
) -> str:
    from engine.video_headline_postprocess import postprocess_video_headline
    from engine.video_bidi import prepare_ffmpeg_overlay_headline

    headline = (plan.get("headlineText") or "").strip()
    video_lang = str(plan.get("language") or "en")
    canonical_name = str(plan.get("productNameResolved") or "")
    overlay_prep = prepare_ffmpeg_overlay_headline(
        headline,
        content_language=video_lang,
        canonical_name=canonical_name,
    )
    return postprocess_video_headline(
        runway_url,
        public_base_url or "",
        headline=overlay_prep.text_plain,
        job_id=job_id,
        overlay_language=video_lang,
        overlay_render_mode=overlay_prep.render_mode,
        overlay_dual_latin=overlay_prep.dual_latin,
        overlay_dual_hebrew=overlay_prep.dual_hebrew,
        overlay_canonical_name=canonical_name,
    )


def _default_compose_marketing_copy(*, product_name: str, product_description: str, plan: Dict[str, Any]) -> str:
    from engine.runway_video import _fallback_packaging_marketing_copy
    from engine.video_language import detect_text_language

    marketing_lang = detect_text_language(product_description)
    return _fallback_packaging_marketing_copy(
        product_name,
        product_description,
        (plan.get("advertisingPromise") or "").strip(),
        output_language=marketing_lang,
        headline_text=(plan.get("headlineText") or "").strip(),
    )


def default_media_pipeline_deps() -> MediaPipelineDeps:
    return MediaPipelineDeps(
        generate_start_image=_default_generate_start_image,
        submit_runway_task=_default_submit_runway_task,
        poll_runway_task=_default_poll_runway_task,
        postprocess_video=_default_postprocess_video,
        compose_marketing_copy=_default_compose_marketing_copy,
    )


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    bucket = state.setdefault("mediaResume", {})
    if not isinstance(bucket, dict):
        bucket = {}
        state["mediaResume"] = bucket
    return bucket


def _sync_legacy_runway_state(state: Dict[str, Any]) -> Dict[str, Any]:
    runway = state.setdefault("runway", {})
    media = _media_bucket(state)
    if media.get("runwayTaskId"):
        runway["taskId"] = media.get("runwayTaskId")
        runway["submissionState"] = media.get("runwaySubmissionStatus") or runway.get("submissionState") or "submitted"
    if media.get("startImageArtifact"):
        runway["startImageDataUri"] = media.get("startImageArtifact")
        runway["startImageCompleted"] = media.get("startImageStatus") == "completed"
    return runway


def _update_media_progress(state: Dict[str, Any], stage: str, **fields: Any) -> None:
    media = _media_bucket(state)
    media["progressStage"] = stage
    media["mediaResumeStatus"] = "running" if stage != "completed" else "completed"
    media.update(fields)
    _sync_legacy_runway_state(state)


def execute_builder2_media_pipeline(
    *,
    job_id: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    public_base_url: str,
    product_description: str,
    dry_run: bool = False,
    deps: Optional[MediaPipelineDeps] = None,
    media_config: Optional[MediaResumeConfiguration] = None,
) -> Tuple[Dict[str, Any], MediaPipelineCounters]:
    counters = MediaPipelineCounters()
    media = _media_bucket(state)
    if not media.get("mediaResumeStartedAt"):
        media["mediaResumeStartedAt"] = _utc_now_iso()
    runway_model = media_config.runwayModel if media_config else resolve_builder2_runway_video_model()
    duration_seconds = media_config.durationSeconds if media_config else resolve_builder2_video_duration_seconds()
    ratio = media_config.ratio if media_config else BUILDER2_RUNWAY_VIDEO_RATIO
    media["runwayModel"] = runway_model
    media["durationSeconds"] = duration_seconds
    media["runwayRatio"] = ratio
    if media_config is not None:
        media["publicBaseUrlSource"] = media_config.public_base_url.source
        media["publicBaseUrl"] = media_config.publicBaseUrl
    headline_decision = get_normalized_headline_decision(plan)

    if media.get("mediaResumeStatus") == "completed" and media.get("finalPublicUrl"):
        counters.media_reused = True
        media["progressStage"] = "completed"
        return state, counters

    validate_builder2_pre_runway(plan)
    _update_media_progress(state, "preparing_start_image")

    if dry_run:
        media["mediaResumeStatus"] = "dry_run_validated"
        return state, counters

    prompt_image_data_uri = str(media.get("startImageArtifact") or (state.get("runway") or {}).get("startImageDataUri") or "")
    if builder2_runway_requires_start_image(runway_model):
        _update_media_progress(state, "generating_start_image")
        MediaResumeIsolationGuard.assert_safe_before_start_image()
        if prompt_image_data_uri:
            media["startImageStatus"] = "reused"
        else:
            prompt_image_data_uri = (deps or default_media_pipeline_deps()).generate_start_image(plan)
            counters.start_image_calls += 1
            if not prompt_image_data_uri:
                raise Builder2TournamentError("builder2_media_start_image_failed")
            media["startImageStatus"] = "completed"
        media["startImageArtifact"] = prompt_image_data_uri

        def _persist_start_image(st: Dict[str, Any]) -> None:
            _update_media_progress(st, "generating_start_image", startImageArtifact=prompt_image_data_uri, startImageStatus=media["startImageStatus"])

        patch_tournament_state(job_id, _persist_start_image)

    existing_task_id = str(
        media.get("runwayTaskId")
        or (state.get("runway") or {}).get("taskId")
        or ""
    ).strip()
    _update_media_progress(state, "submitting_runway")
    MediaResumeIsolationGuard.assert_safe_before_runway()
    if existing_task_id:
        task_id = existing_task_id
        media["runwaySubmissionStatus"] = "reused"
        counters.runway_polling_resumed = True
    elif (state.get("runway") or {}).get("submissionState") == "pending" and not existing_task_id:
        raise Builder2TournamentError("builder2_media_runway_resume_ambiguous")
    else:
        pipeline_deps = deps or default_media_pipeline_deps()
        task_id = pipeline_deps.submit_runway_task(
            plan=plan,
            prompt_image_data_uri=prompt_image_data_uri,
            runway_model=runway_model,
            duration_seconds=duration_seconds,
        )
        counters.runway_submission_calls += 1
        media["runwaySubmissionStatus"] = "submitted"
        media["runwaySubmittedAt"] = _utc_now_iso()

        def _persist_task(st: Dict[str, Any]) -> None:
            _update_media_progress(
                st,
                "submitting_runway",
                runwayTaskId=task_id,
                runwaySubmissionStatus="submitted",
                runwaySubmittedAt=media["runwaySubmittedAt"],
            )

        patch_tournament_state(job_id, _persist_task)
    media["runwayTaskId"] = task_id

    runway_url = str(media.get("runwayVideoUrl") or "").strip()
    if runway_url:
        media["runwayStatus"] = "reused"
    else:
        _update_media_progress(state, "waiting_for_runway", runwayTaskId=task_id)
        pipeline_deps = deps or default_media_pipeline_deps()
        status, runway_url = pipeline_deps.poll_runway_task(task_id=task_id)
        media["runwayStatus"] = status
        media["runwayVideoUrl"] = runway_url

        def _persist_runway(st: Dict[str, Any]) -> None:
            _update_media_progress(st, "waiting_for_runway", runwayStatus=status, runwayVideoUrl=runway_url)

        patch_tournament_state(job_id, _persist_runway)

    downloaded_path = str(media.get("downloadedVideoPath") or "").strip()
    if downloaded_path:
        counters.media_reused = True
    else:
        _update_media_progress(state, "downloading_video", downloadedVideoPath=runway_url)
        media["downloadedVideoPath"] = runway_url
        downloaded_path = runway_url

    final_url = str(media.get("finalPublicUrl") or "").strip()
    marketing_text = str(media.get("marketingText") or "")
    overlay_headline = ""
    if final_url:
        counters.media_reused = True
    else:
        pipeline_deps = deps or default_media_pipeline_deps()
        marketing_text = pipeline_deps.compose_marketing_copy(
            product_name=str(plan.get("productNameResolved") or ""),
            product_description=product_description,
            plan=plan,
        )
        if headline_decision_requires_headline(headline_decision):
            _update_media_progress(state, "postprocessing_video")
            MediaResumeIsolationGuard.assert_safe_before_ffmpeg()
            final_url = pipeline_deps.postprocess_video(
                runway_url=downloaded_path,
                public_base_url=public_base_url,
                plan=plan,
                job_id=job_id,
            )
            counters.ffmpeg_calls += 1
            media["ffmpegStatus"] = "completed"
            overlay_headline = (plan.get("headlineText") or "").strip()
        else:
            _update_media_progress(state, "finalizing_delivery")
            final_url = downloaded_path
            media["ffmpegStatus"] = "skipped_omit"
        media["finalVideoPath"] = final_url
        media["finalPublicUrl"] = final_url
        media["marketingText"] = marketing_text
        media["deliveryArtifactPaths"] = [final_url]

    _update_media_progress(
        state,
        "completed",
        mediaCompletedAt=_utc_now_iso(),
        finalPublicUrl=final_url,
        finalVideoPath=final_url,
        marketingText=marketing_text,
    )
    state["mediaContinuationRequired"] = False
    state["status"] = "completed"
    state["lastCompletedStep"] = "done"
    return state, counters
