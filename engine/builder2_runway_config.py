"""
Builder2 Runway video model configuration — isolated from Builder1 and global vars.
"""
from __future__ import annotations

import logging
import os
from typing import FrozenSet

logger = logging.getLogger(__name__)

VALID_BUILDER2_RUNWAY_VIDEO_MODELS: FrozenSet[str] = frozenset({"gen4_turbo", "gen4.5"})
DEFAULT_BUILDER2_RUNWAY_VIDEO_MODEL = "gen4_turbo"
BUILDER2_RUNWAY_VIDEO_RATIO = "1280:720"
DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS = 7
BUILDER2_VIDEO_DURATION_MIN_SECONDS = 2
BUILDER2_VIDEO_DURATION_MAX_SECONDS = 10
DEFAULT_BUILDER2_START_IMAGE_SIZE = "1280x720"


class Builder2RunwayConfigError(ValueError):
    """Invalid Builder2 Runway configuration."""


def resolve_builder2_runway_video_model() -> str:
    raw = (os.environ.get("BUILDER2_RUNWAY_VIDEO_MODEL") or "").strip()
    if not raw:
        return DEFAULT_BUILDER2_RUNWAY_VIDEO_MODEL
    if raw not in VALID_BUILDER2_RUNWAY_VIDEO_MODELS:
        logger.error(
            "BUILDER2_RUNWAY_VIDEO_MODEL_INVALID model=%s allowed=%s",
            raw,
            sorted(VALID_BUILDER2_RUNWAY_VIDEO_MODELS),
        )
        raise Builder2RunwayConfigError(f"unsupported_builder2_runway_model:{raw}")
    return raw


def resolve_builder2_video_duration_seconds() -> int:
    raw = (os.environ.get("BUILDER2_VIDEO_DURATION_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.error(
            "BUILDER2_VIDEO_DURATION_INVALID value=%s reason=not_integer fallback=none",
            raw,
        )
        raise Builder2RunwayConfigError(f"builder2_invalid_video_duration:{raw}")
    if value < BUILDER2_VIDEO_DURATION_MIN_SECONDS or value > BUILDER2_VIDEO_DURATION_MAX_SECONDS:
        logger.error(
            "BUILDER2_VIDEO_DURATION_INVALID value=%s allowed=%s-%s",
            value,
            BUILDER2_VIDEO_DURATION_MIN_SECONDS,
            BUILDER2_VIDEO_DURATION_MAX_SECONDS,
        )
        raise Builder2RunwayConfigError(f"builder2_invalid_video_duration:{value}")
    return value


def builder2_runway_requires_start_image(model: str) -> bool:
    return (model or "").strip() == "gen4_turbo"


def builder2_runway_generation_mode(model: str) -> str:
    return "image_to_video" if builder2_runway_requires_start_image(model) else "text_to_video"


def resolve_builder2_start_image_size() -> str:
    raw = (
        (os.environ.get("BUILDER2_START_IMAGE_SIZE") or "").strip()
        or (os.environ.get("VIDEO_START_IMAGE_SIZE") or "").strip()
        or DEFAULT_BUILDER2_START_IMAGE_SIZE
    )
    return raw


def log_builder2_runway_model_selected(
    *,
    job_id: str,
    model: str,
    mode: str,
    duration: int,
    ratio: str = BUILDER2_RUNWAY_VIDEO_RATIO,
) -> None:
    logger.info(
        "BUILDER2_RUNWAY_MODEL_SELECTED jobId=%s model=%s mode=%s duration=%s ratio=%s",
        (job_id or "").strip() or "(none)",
        model,
        mode,
        duration,
        ratio,
    )


def log_builder2_video_duration_selected(
    *,
    job_id: str,
    duration_seconds: int,
    model: str,
    mode: str,
) -> None:
    logger.info(
        "BUILDER2_VIDEO_DURATION_SELECTED jobId=%s durationSeconds=%s model=%s mode=%s",
        (job_id or "").strip() or "(none)",
        duration_seconds,
        model,
        mode,
    )
