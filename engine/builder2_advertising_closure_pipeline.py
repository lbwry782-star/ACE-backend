"""
Builder2 Advertising Closure render pipeline — append end card without touching Runway output.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Union

from engine.builder2_advertising_closure_contract import normalize_advertising_closure, validate_advertising_closure_object
from engine.builder2_advertising_closure_resume_guard import AdvertisingClosureResumeGuard
from engine.builder2_closure_render import Builder2ClosureRenderError, ClosureRenderResult, url_fingerprint
from engine.builder2_new_format_config import resolve_builder2_end_card_duration_seconds


@dataclass
class AdvertisingClosureRenderCounters:
    closure_ffmpeg_calls: int = 0
    media_reused: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    bucket = state.setdefault("mediaResume", {})
    if not isinstance(bucket, dict):
        bucket = {}
        state["mediaResume"] = bucket
    return bucket


def resolve_raw_runway_video(state: Dict[str, Any]) -> str:
    media = _media_bucket(state)
    for key in ("rawRunwayVideoUrl", "runwayVideoUrl", "downloadedVideoPath"):
        value = str(media.get(key) or "").strip()
        if value:
            return value
    runway = state.get("runway") or {}
    if isinstance(runway, dict):
        return str(runway.get("videoUrl") or "").strip()
    return ""


def _render_result_public_url(result: Union[str, ClosureRenderResult]) -> ClosureRenderResult:
    if isinstance(result, ClosureRenderResult):
        return result
    raise Builder2ClosureRenderError(
        "builder2_closure_invalid_render_result",
        stage="render_validation",
        command_category="validation",
    )


def render_advertising_closure_for_state(
    *,
    job_id: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    closure: Dict[str, Any],
    public_base_url: str,
    source_video_url: str,
    render_endcard: Callable[..., Union[str, ClosureRenderResult]],
) -> tuple[Dict[str, Any], AdvertisingClosureRenderCounters]:
    counters = AdvertisingClosureRenderCounters()
    media = _media_bucket(state)
    normalized = normalize_advertising_closure(closure)
    validate_advertising_closure_object(normalized, plan=plan)

    existing_url = str(media.get("finalVideoWithClosureUrl") or "").strip()
    if (
        existing_url
        and media.get("advertisingClosureRendered") is True
        and media.get("advertisingClosureStatus") == "completed"
        and media.get("actualFinalVideoDurationSeconds") is not None
    ):
        counters.media_reused = True
        return state, counters

    source = str(source_video_url or "").strip()
    if not source:
        raise Builder2ClosureRenderError(
            "builder2_advertising_closure_missing_source_video",
            stage="input_validation",
            command_category="validation",
        )

    raw_url = resolve_raw_runway_video(state)
    if raw_url and not media.get("rawRunwayVideoUrl"):
        media["rawRunwayVideoUrl"] = raw_url
        media["rawRunwayVideoPath"] = raw_url

    AdvertisingClosureResumeGuard.assert_safe_before_closure_ffmpeg()
    media["advertisingClosureStatus"] = "rendering"
    state["advertisingClosureStatus"] = "rendering"
    counters.closure_ffmpeg_calls = 1
    try:
        rendered = render_endcard(
            source,
            public_base_url,
            product_name=str(normalized.get("productNameText") or ""),
            slogan=str(normalized.get("sloganText") or ""),
            language=str(normalized.get("language") or "en"),
            duration_seconds=float(normalized.get("durationSeconds") or resolve_builder2_end_card_duration_seconds()),
            job_id=job_id,
        )
        result = _render_result_public_url(rendered)
    except Builder2ClosureRenderError as exc:
        media["advertisingClosureStatus"] = "failed"
        state["advertisingClosureStatus"] = "failed"
        media["advertisingClosureFailure"] = {
            "stage": exc.stage,
            "reason": str(exc.args[0] if exc.args else "builder2_closure_ffmpeg_failed"),
            "returnCode": exc.return_code,
            "stderrTail": exc.stderr_tail,
            "commandCategory": exc.command_category,
            "sourceFingerprint": url_fingerprint(source),
        }
        raise

    if not result.public_url or result.public_url == source:
        media["advertisingClosureStatus"] = "failed"
        state["advertisingClosureStatus"] = "failed"
        raise Builder2ClosureRenderError(
            "builder2_closure_output_same_as_input",
            stage="render_validation",
            command_category="validation",
        )

    media["advertisingClosureArtifact"] = normalized
    media["finalVideoWithClosurePath"] = result.public_url
    media["finalVideoWithClosureUrl"] = result.public_url
    media["finalPublicUrl"] = result.public_url
    media["finalVideoPath"] = result.public_url
    media["actualFinalVideoDurationSeconds"] = result.measured_duration_seconds
    media["expectedFinalVideoDurationSeconds"] = resolve_builder2_end_card_duration_seconds() + float(
        media.get("rawRunwayDurationSeconds") or 0
    )
    media["closureRenderedAt"] = _utc_now_iso()
    media["advertisingClosureRendered"] = True
    media["advertisingClosureStatus"] = "completed"
    media.pop("advertisingClosureFailure", None)
    state["advertisingClosure"] = normalized
    state["advertisingClosureStatus"] = "completed"
    return state, counters
