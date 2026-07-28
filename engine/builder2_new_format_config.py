"""
Builder2 new complete-ad format configuration — gen4.5, 10s visual, 2s end card, 12s final.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from engine.builder2_runway_config import (
    BUILDER2_RUNWAY_VIDEO_RATIO,
    Builder2RunwayConfigError,
    resolve_builder2_runway_video_model,
    resolve_builder2_video_duration_seconds,
)

logger = logging.getLogger(__name__)

BUILDER2_NEW_FORMAT_VERSION = "builder2_complete_ad_v1"
NORMAL_REASONING_CALL_BUDGET = 14

DEFAULT_BUILDER2_RUNWAY_MODEL = "gen4.5"
DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS = 10
DEFAULT_BUILDER2_END_CARD_DURATION_SECONDS = 2.0
DEFAULT_BUILDER2_FINAL_VIDEO_DURATION_SECONDS = 12.0
FINAL_DURATION_TOLERANCE_SECONDS = 0.35

LEGACY_RUNWAY_MODEL = "gen4_turbo"
LEGACY_RUNWAY_DURATION_SECONDS = 7
LEGACY_END_CARD_DURATION_SECONDS = 1.5


def resolve_builder2_end_card_duration_seconds() -> float:
    raw = (os.environ.get("BUILDER2_END_CARD_DURATION_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_BUILDER2_END_CARD_DURATION_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise Builder2RunwayConfigError(f"builder2_invalid_end_card_duration:{raw}")
    if value <= 0 or value > 5:
        raise Builder2RunwayConfigError(f"builder2_invalid_end_card_duration:{value}")
    return value


def resolve_builder2_effective_closure_segment_duration_seconds(
    requested_duration_seconds: float | None = None,
) -> float:
    """
    Authoritative Builder2 closure segment duration.

    Fades/transitions must fit inside this segment; nothing may be appended outside it.
    """
    effective = float(resolve_builder2_end_card_duration_seconds())
    if requested_duration_seconds is not None:
        requested = float(requested_duration_seconds)
        if abs(requested - effective) > 0.01:
            logger.info(
                "BUILDER2_CLOSURE_SEGMENT_DURATION_COERCED requested=%.3f effective=%.3f",
                requested,
                effective,
            )
    return effective


def resolve_builder2_final_video_duration_seconds() -> float:
    raw = (os.environ.get("BUILDER2_FINAL_VIDEO_DURATION_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_BUILDER2_FINAL_VIDEO_DURATION_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise Builder2RunwayConfigError(f"builder2_invalid_final_video_duration:{raw}")
    if value <= 0 or value > 30:
        raise Builder2RunwayConfigError(f"builder2_invalid_final_video_duration:{value}")
    return value


def resolved_new_format_runway_settings() -> Dict[str, object]:
    return {
        "model": resolve_builder2_runway_video_model(),
        "durationSeconds": resolve_builder2_video_duration_seconds(),
        "ratio": BUILDER2_RUNWAY_VIDEO_RATIO,
        "mode": "image_to_video",
        "endCardDurationSeconds": resolve_builder2_end_card_duration_seconds(),
        "finalVideoDurationSeconds": resolve_builder2_final_video_duration_seconds(),
    }


def builder2_media_requires_closure_ffmpeg(*, state: Dict[str, object] | None, plan: Dict[str, object] | None) -> bool:
    closure = {}
    if isinstance(state, dict):
        raw = state.get("advertisingClosure")
        if isinstance(raw, dict):
            closure = raw
    if not closure and isinstance(plan, dict):
        raw = plan.get("advertisingClosure")
        if isinstance(raw, dict):
            closure = raw
    if not isinstance(closure, dict):
        return False
    if closure.get("required") is not True:
        return False
    return bool(str(closure.get("sloganText") or "").strip())


def validate_new_format_runway_configuration(*, dry_run: bool = True) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    model = resolve_builder2_runway_video_model()
    duration = resolve_builder2_video_duration_seconds()
    end_card = resolve_builder2_end_card_duration_seconds()
    final_duration = resolve_builder2_final_video_duration_seconds()
    if model != DEFAULT_BUILDER2_RUNWAY_MODEL:
        failures.append(f"runway_model_expected_{DEFAULT_BUILDER2_RUNWAY_MODEL}_actual_{model}")
    if duration != DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS:
        failures.append(
            f"runway_duration_expected_{DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS}_actual_{duration}"
        )
    if abs(end_card - DEFAULT_BUILDER2_END_CARD_DURATION_SECONDS) > 0.01:
        failures.append(
            f"end_card_duration_expected_{DEFAULT_BUILDER2_END_CARD_DURATION_SECONDS}_actual_{end_card}"
        )
    expected_final = float(duration) + float(end_card)
    if abs(final_duration - expected_final) > 0.01:
        failures.append(
            f"final_duration_expected_{expected_final}_actual_{final_duration}"
        )
    if dry_run and failures:
        logger.error("BUILDER2_NEW_FORMAT_CONFIG_MISMATCH failures=%s", failures)
    return not failures, failures


def log_new_format_configuration(*, job_id: str = "") -> None:
    settings = resolved_new_format_runway_settings()
    logger.info(
        "BUILDER2_NEW_FORMAT_CONFIG jobId=%s version=%s model=%s duration=%s endCard=%s final=%s ratio=%s mode=%s",
        (job_id or "").strip() or "(none)",
        BUILDER2_NEW_FORMAT_VERSION,
        settings["model"],
        settings["durationSeconds"],
        settings["endCardDurationSeconds"],
        settings["finalVideoDurationSeconds"],
        settings["ratio"],
        settings["mode"],
    )
