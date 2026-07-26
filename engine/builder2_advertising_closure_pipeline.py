"""
Builder2 Advertising Closure render pipeline — append end card without touching Runway output.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from engine.builder2_advertising_closure_contract import normalize_advertising_closure, validate_advertising_closure_object
from engine.builder2_advertising_closure_resume_guard import AdvertisingClosureResumeGuard


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
    for key in ("rawRunwayVideoUrl", "runwayVideoUrl", "downloadedVideoPath", "finalPublicUrl", "finalVideoPath"):
        value = str(media.get(key) or "").strip()
        if value:
            return value
    runway = state.get("runway") or {}
    if isinstance(runway, dict):
        return str(runway.get("videoUrl") or "").strip()
    return ""


def render_advertising_closure_for_state(
    *,
    job_id: str,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    closure: Dict[str, Any],
    public_base_url: str,
    render_endcard: Callable[..., str],
) -> tuple[Dict[str, Any], AdvertisingClosureRenderCounters]:
    counters = AdvertisingClosureRenderCounters()
    media = _media_bucket(state)
    normalized = normalize_advertising_closure(closure)
    validate_advertising_closure_object(normalized, plan=plan)

    existing_url = str(media.get("finalVideoWithClosureUrl") or "").strip()
    if existing_url and media.get("advertisingClosureStatus") == "completed":
        counters.media_reused = True
        return state, counters

    raw_url = resolve_raw_runway_video(state)
    if raw_url and not media.get("rawRunwayVideoUrl"):
        media["rawRunwayVideoUrl"] = raw_url
        media["rawRunwayVideoPath"] = raw_url

    if not raw_url:
        raise RuntimeError("builder2_advertising_closure_missing_raw_video")

    AdvertisingClosureResumeGuard.assert_safe_before_closure_ffmpeg()
    media["advertisingClosureStatus"] = "rendering"
    state["advertisingClosureStatus"] = "rendering"
    counters.closure_ffmpeg_calls = 1
    final_url = render_endcard(
        raw_url,
        public_base_url,
        product_name=str(normalized.get("productNameText") or ""),
        slogan=str(normalized.get("sloganText") or ""),
        language=str(normalized.get("language") or "en"),
        duration_seconds=float(normalized.get("durationSeconds") or 1.5),
        job_id=job_id,
    )
    media["advertisingClosureArtifact"] = normalized
    media["finalVideoWithClosurePath"] = final_url
    media["finalVideoWithClosureUrl"] = final_url
    media["finalPublicUrl"] = final_url
    media["finalVideoPath"] = final_url
    media["closureRenderedAt"] = _utc_now_iso()
    media["advertisingClosureStatus"] = "completed"
    state["advertisingClosure"] = normalized
    state["advertisingClosureStatus"] = "completed"
    state["status"] = "completed"
    state["lastCompletedStep"] = "done"
    return state, counters
