"""
Builder2 start-image geometry — OpenAI generation size vs Runway delivery size.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple

SUPPORTED_OPENAI_GENERATION_SIZES: FrozenSet[str] = frozenset(
    {"1024x1024", "1024x1536", "1536x1024", "auto"}
)
DEFAULT_BUILDER2_START_IMAGE_GENERATION_SIZE = "1536x1024"
DEFAULT_BUILDER2_START_IMAGE_OUTPUT_SIZE = "1280x720"
DEFAULT_BUILDER2_OUTPUT_ASPECT_RATIO = "16:9"
DEFAULT_BUILDER2_CROP_STRATEGY = "center_crop"
LEGACY_RUNWAY_OUTPUT_SIZE = "1280x720"

Builder2StartImageGeometryError = type(
    "Builder2StartImageGeometryError",
    (ValueError,),
    {},
)


@dataclass(frozen=True)
class Builder2StartImageGeometry:
    imageGenerationSize: str
    startImageOutputSize: str
    outputAspectRatio: str
    cropStrategy: str
    resizeRequired: bool
    cropBox: Dict[str, int]
    generationWidth: int
    generationHeight: int
    croppedWidth: int
    croppedHeight: int
    outputWidth: int
    outputHeight: int

    def to_safe_metadata(self) -> Dict[str, Any]:
        return {
            "imageGenerationSize": self.imageGenerationSize,
            "startImageOutputSize": self.startImageOutputSize,
            "outputAspectRatio": self.outputAspectRatio,
            "cropStrategy": self.cropStrategy,
            "cropBox": dict(self.cropBox),
            "resizeRequired": self.resizeRequired,
            "startImageGeometryAccepted": True,
        }


def parse_size_token(raw: str) -> Tuple[int, int]:
    token = (raw or "").strip().lower()
    if "x" not in token:
        raise Builder2StartImageGeometryError(f"builder2_start_image_invalid_size:{raw}")
    left, right = token.split("x", 1)
    try:
        width = int(left.strip())
        height = int(right.strip())
    except ValueError as exc:
        raise Builder2StartImageGeometryError(f"builder2_start_image_invalid_size:{raw}") from exc
    if width <= 0 or height <= 0:
        raise Builder2StartImageGeometryError(f"builder2_start_image_invalid_size:{raw}")
    return width, height


def format_size_token(width: int, height: int) -> str:
    return f"{width}x{height}"


def _resolve_generation_size_token() -> str:
    explicit = (os.environ.get("BUILDER2_START_IMAGE_GENERATION_SIZE") or "").strip()
    if explicit:
        if explicit.lower() == LEGACY_RUNWAY_OUTPUT_SIZE.lower():
            raise Builder2StartImageGeometryError("builder2_start_image_unsupported_generation_size")
        return explicit
    for env_name in ("BUILDER2_START_IMAGE_SIZE", "VIDEO_START_IMAGE_SIZE"):
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        if raw.lower() == LEGACY_RUNWAY_OUTPUT_SIZE.lower():
            return DEFAULT_BUILDER2_START_IMAGE_GENERATION_SIZE
        return raw
    return DEFAULT_BUILDER2_START_IMAGE_GENERATION_SIZE


def _resolve_output_size_token() -> str:
    for env_name in ("BUILDER2_START_IMAGE_OUTPUT_SIZE", "VIDEO_START_IMAGE_OUTPUT_SIZE"):
        raw = (os.environ.get(env_name) or "").strip()
        if raw:
            return raw
    legacy = (os.environ.get("VIDEO_START_IMAGE_SIZE") or "").strip()
    if legacy.lower() == LEGACY_RUNWAY_OUTPUT_SIZE.lower():
        return LEGACY_RUNWAY_OUTPUT_SIZE
    return DEFAULT_BUILDER2_START_IMAGE_OUTPUT_SIZE


def _output_aspect_ratio(output_width: int, output_height: int) -> str:
    from math import gcd

    divisor = gcd(output_width, output_height)
    return f"{output_width // divisor}:{output_height // divisor}"


def _compute_center_crop_box(
    *,
    generation_width: int,
    generation_height: int,
    output_width: int,
    output_height: int,
) -> Dict[str, int]:
    crop_height = int(round(generation_width * output_height / output_width))
    if crop_height > generation_height:
        raise Builder2StartImageGeometryError("builder2_start_image_invalid_crop_geometry")
    top = (generation_height - crop_height) // 2
    bottom = top + crop_height
    return {
        "left": 0,
        "top": top,
        "right": generation_width,
        "bottom": bottom,
    }


def assert_pillow_available() -> None:
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise Builder2StartImageGeometryError("builder2_start_image_processing_unavailable") from exc


def resolve_builder2_start_image_geometry() -> Builder2StartImageGeometry:
    generation_token = _resolve_generation_size_token()
    output_token = _resolve_output_size_token()

    if generation_token.lower() == LEGACY_RUNWAY_OUTPUT_SIZE.lower():
        raise Builder2StartImageGeometryError("builder2_start_image_unsupported_generation_size")

    if generation_token not in SUPPORTED_OPENAI_GENERATION_SIZES:
        raise Builder2StartImageGeometryError("builder2_start_image_unsupported_generation_size")

    generation_width, generation_height = parse_size_token(generation_token)
    output_width, output_height = parse_size_token(output_token)
    crop_box = _compute_center_crop_box(
        generation_width=generation_width,
        generation_height=generation_height,
        output_width=output_width,
        output_height=output_height,
    )
    cropped_width = crop_box["right"] - crop_box["left"]
    cropped_height = crop_box["bottom"] - crop_box["top"]
    return Builder2StartImageGeometry(
        imageGenerationSize=generation_token,
        startImageOutputSize=output_token,
        outputAspectRatio=_output_aspect_ratio(output_width, output_height),
        cropStrategy=DEFAULT_BUILDER2_CROP_STRATEGY,
        resizeRequired=True,
        cropBox=crop_box,
        generationWidth=generation_width,
        generationHeight=generation_height,
        croppedWidth=cropped_width,
        croppedHeight=cropped_height,
        outputWidth=output_width,
        outputHeight=output_height,
    )


def validate_builder2_start_image_geometry(geometry: Builder2StartImageGeometry) -> None:
    if geometry.imageGenerationSize not in SUPPORTED_OPENAI_GENERATION_SIZES:
        raise Builder2StartImageGeometryError("builder2_start_image_unsupported_generation_size")
    if geometry.imageGenerationSize.lower() == LEGACY_RUNWAY_OUTPUT_SIZE.lower():
        raise Builder2StartImageGeometryError("builder2_start_image_unsupported_generation_size")
    if geometry.croppedWidth <= 0 or geometry.croppedHeight <= 0:
        raise Builder2StartImageGeometryError("builder2_start_image_invalid_crop_geometry")
    expected_ratio = _output_aspect_ratio(geometry.outputWidth, geometry.outputHeight)
    if geometry.outputAspectRatio != expected_ratio:
        raise Builder2StartImageGeometryError("builder2_start_image_output_ratio_mismatch")
    assert_pillow_available()
