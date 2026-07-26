"""
Builder2 start-image pipeline — OpenAI generation, crop/resize, Runway delivery artifact.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

from engine import openai_retry
from engine.builder2_start_image_geometry import (
    Builder2StartImageGeometry,
    Builder2StartImageGeometryError,
    assert_pillow_available,
    format_size_token,
    parse_size_token,
    resolve_builder2_start_image_geometry,
    validate_builder2_start_image_geometry,
)
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<payload>.+)$", re.DOTALL)
_VIDEO_START_IMAGE_TIMEOUT = float((os.environ.get("VIDEO_START_IMAGE_TIMEOUT_SECONDS") or "90").strip() or "90")
_START_IMAGE_MIME_TYPE = "image/png"


@dataclass
class StartImageCallCounters:
    startImageNormalCalls: int = 0
    startImageRepairCalls: int = 0
    startImageRetryCalls: int = 0
    startImageGeneratedCount: int = 0

    @property
    def submitted_calls(self) -> int:
        return self.startImageNormalCalls + self.startImageRepairCalls + self.startImageRetryCalls


@dataclass
class Builder2StartImageResult:
    data_uri: Optional[str] = None
    counters: StartImageCallCounters = field(default_factory=StartImageCallCounters)
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    geometry: Optional[Builder2StartImageGeometry] = None
    api_submitted: bool = False
    api_status: Optional[int] = None
    api_error_category: Optional[str] = None
    submitted_size: Optional[str] = None
    model_name: Optional[str] = None
    call_kind: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)


class Builder2StartImagePipelineError(Builder2TournamentError):
    def __init__(
        self,
        reason: str,
        *,
        failure_stage: str,
        result: Optional[Builder2StartImageResult] = None,
    ) -> None:
        super().__init__(reason)
        self.failure_stage = failure_stage
        self.result = result


def _decode_data_uri(data_uri: str) -> tuple[str, bytes]:
    match = _DATA_URI_RE.match((data_uri or "").strip())
    if not match:
        raise Builder2StartImagePipelineError(
            "builder2_start_image_invalid_artifact",
            failure_stage="pre_runway_image_validation",
        )
    return match.group("mime"), base64.b64decode(match.group("payload"))


def _encode_data_uri(image_bytes: bytes, *, mime_type: str = _START_IMAGE_MIME_TYPE) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def transform_builder2_start_image(
    image_bytes: bytes,
    geometry: Builder2StartImageGeometry,
) -> tuple[bytes, Dict[str, Any]]:
    from PIL import Image

    validate_builder2_start_image_geometry(geometry)
    image = Image.open(io.BytesIO(image_bytes))
    actual_width, actual_height = image.size
    if (actual_width, actual_height) != (geometry.generationWidth, geometry.generationHeight):
        raise Builder2StartImagePipelineError(
            "builder2_start_image_source_dimension_mismatch",
            failure_stage="start_image_postprocess",
        )
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGBA")
    crop = geometry.cropBox
    cropped = image.crop((crop["left"], crop["top"], crop["right"], crop["bottom"]))
    if cropped.size != (geometry.croppedWidth, geometry.croppedHeight):
        raise Builder2StartImagePipelineError(
            "builder2_start_image_crop_dimension_mismatch",
            failure_stage="start_image_postprocess",
        )
    resized = cropped.resize((geometry.outputWidth, geometry.outputHeight), Image.Resampling.LANCZOS)
    if resized.size != (geometry.outputWidth, geometry.outputHeight):
        raise Builder2StartImagePipelineError(
            "builder2_start_image_output_dimension_mismatch",
            failure_stage="start_image_postprocess",
        )
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    output_bytes = buffer.getvalue()
    metadata = {
        "startImageGenerationSize": geometry.imageGenerationSize,
        "startImageSourceWidth": actual_width,
        "startImageSourceHeight": actual_height,
        "startImageCropBox": dict(geometry.cropBox),
        "startImageOutputWidth": geometry.outputWidth,
        "startImageOutputHeight": geometry.outputHeight,
        "startImageMimeType": _START_IMAGE_MIME_TYPE,
        "startImageStatus": "completed",
    }
    return output_bytes, metadata


def validate_builder2_runway_start_image_artifact(
    data_uri: str,
    geometry: Optional[Builder2StartImageGeometry] = None,
) -> Dict[str, Any]:
    resolved = geometry or resolve_builder2_start_image_geometry()
    _, image_bytes = _decode_data_uri(data_uri)
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    if (width, height) != (resolved.outputWidth, resolved.outputHeight):
        raise Builder2StartImagePipelineError(
            "builder2_start_image_runway_dimension_mismatch",
            failure_stage="pre_runway_image_validation",
        )
    return {
        "startImageOutputWidth": width,
        "startImageOutputHeight": height,
        "startImageMimeType": _START_IMAGE_MIME_TYPE,
    }


def _resolve_openai_image_model() -> str:
    return (os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1.5").strip()


def _increment_submitted_counter(counters: StartImageCallCounters, call_kind: str) -> None:
    if call_kind == "repair":
        counters.startImageRepairCalls += 1
    elif call_kind == "retry":
        counters.startImageRetryCalls += 1
    else:
        counters.startImageNormalCalls += 1


def generate_builder2_start_image(
    plan: Dict[str, Any],
    *,
    call_kind: str = "normal",
) -> Builder2StartImageResult:
    from engine.video_start_image import build_ace_start_frame_image_prompt

    result = Builder2StartImageResult(call_kind=call_kind)
    try:
        geometry = resolve_builder2_start_image_geometry()
        validate_builder2_start_image_geometry(geometry)
    except Builder2StartImageGeometryError as exc:
        result.failure_stage = "start_image_configuration"
        result.failure_reason = str(exc)
        raise Builder2StartImagePipelineError(str(exc), failure_stage="start_image_configuration", result=result) from exc

    result.geometry = geometry
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        result.failure_stage = "start_image_configuration"
        result.failure_reason = "builder2_start_image_missing_openai_api_key"
        raise Builder2StartImagePipelineError(result.failure_reason, failure_stage="start_image_configuration", result=result)

    prompt = build_ace_start_frame_image_prompt(plan)
    model = _resolve_openai_image_model()
    quality = (os.environ.get("VIDEO_START_IMAGE_QUALITY") or "low").strip()
    result.submitted_size = geometry.imageGenerationSize
    result.model_name = model

    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(_VIDEO_START_IMAGE_TIMEOUT),
        max_retries=0,
    )
    _increment_submitted_counter(result.counters, call_kind)
    result.api_submitted = True
    try:
        response = openai_retry.openai_call_with_retry(
            lambda: client.images.generate(
                model=model,
                prompt=prompt,
                size=geometry.imageGenerationSize,
                quality=quality,
            ),
            endpoint="images",
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code is None and getattr(exc, "response", None) is not None:
            status_code = getattr(exc.response, "status_code", None)
        result.api_status = status_code
        result.api_error_category = type(exc).__name__
        result.failure_stage = "start_image_generation"
        result.failure_reason = "builder2_media_start_image_api_rejected"
        logger.warning(
            "BUILDER2_START_IMAGE_API_REJECTED status=%s err_type=%s submitted_size=%s model=%s",
            status_code,
            type(exc).__name__,
            geometry.imageGenerationSize,
            model,
        )
        raise Builder2StartImagePipelineError(
            result.failure_reason,
            failure_stage="start_image_generation",
            result=result,
        ) from exc

    b64 = response.data[0].b64_json if response.data else None
    if not b64:
        result.failure_stage = "start_image_generation"
        result.failure_reason = "builder2_media_start_image_api_empty_response"
        raise Builder2StartImagePipelineError(result.failure_reason, failure_stage="start_image_generation", result=result)

    try:
        source_bytes = base64.b64decode(b64)
        output_bytes, metadata = transform_builder2_start_image(source_bytes, geometry)
    except Builder2StartImagePipelineError:
        raise
    except Exception as exc:
        result.failure_stage = "start_image_postprocess"
        result.failure_reason = "builder2_start_image_postprocess_failed"
        result.metadata = {
            "sourceDimensions": format_size_token(geometry.generationWidth, geometry.generationHeight),
            "requestedOutputDimensions": geometry.startImageOutputSize,
            "cropStrategy": geometry.cropStrategy,
            "exceptionClass": type(exc).__name__,
        }
        raise Builder2StartImagePipelineError(
            result.failure_reason,
            failure_stage="start_image_postprocess",
            result=result,
        ) from exc

    result.data_uri = _encode_data_uri(output_bytes)
    result.counters.startImageGeneratedCount = 1
    result.metadata = {
        **metadata,
        "startImageArtifact": result.data_uri,
        "callSubmitted": True,
        "submittedSize": geometry.imageGenerationSize,
        "modelName": model,
    }
    logger.info(
        "BUILDER2_START_IMAGE_OK generationSize=%s outputSize=%s model=%s",
        geometry.imageGenerationSize,
        geometry.startImageOutputSize,
        model,
    )
    return result


def generate_builder2_start_image_data_uri(plan: Dict[str, Any]) -> Optional[str]:
    try:
        return generate_builder2_start_image(plan).data_uri
    except Builder2StartImagePipelineError:
        return None
