"""
Builder1 single-ad image generation (active production — one image per user action).
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_visual_prompt import build_visual_prompt

logger = logging.getLogger(__name__)

ImageCaller = Callable[[str, str], bytes]


class ImageRateLimitError(Exception):
    def __init__(self, message: str = "image_rate_limited", *, retry_after_seconds: Optional[int] = None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


@dataclass
class Builder1AdImageResult:
    index: int
    visual_prompt: str
    image_bytes: bytes


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"RateLimitError", "APIStatusError"}:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def _retry_after_seconds(exc: Exception) -> Optional[int]:
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None) or {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return None


def generate_builder1_ad_image(
    series_plan: Builder1SeriesPlan,
    ad_index: int,
    image_caller: ImageCaller,
) -> Builder1AdImageResult:
    """
    Generate exactly one image for one planned ad index.
    One retry on transient non-rate-limit failures. Rate limits are not retried here.
    """
    ad = next(a for a in series_plan.ads if a.index == ad_index)
    prompt = build_visual_prompt(series_plan, ad)
    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            logger.info("BUILDER1_SERIES_IMAGE_START adIndex=%s attempt=%s", ad_index, attempt)
            image_bytes = image_caller(prompt, series_plan.format)
            logger.info("BUILDER1_SERIES_IMAGE_OK adIndex=%s", ad_index)
            return Builder1AdImageResult(index=ad_index, visual_prompt=prompt, image_bytes=image_bytes)
        except ImageRateLimitError:
            raise
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                retry_after = _retry_after_seconds(exc)
                logger.error(
                    "BUILDER1_IMAGE_RATE_LIMITED adIndex=%s retryAfterSeconds=%s",
                    ad_index,
                    retry_after,
                )
                raise ImageRateLimitError(retry_after_seconds=retry_after) from exc
            logger.error(
                "BUILDER1_SERIES_IMAGE_FAILED adIndex=%s attempt=%s err=%s",
                ad_index,
                attempt,
                exc,
            )
    raise RuntimeError(f"image_generation_failed_ad_{ad_index}") from last_exc


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def generate_builder1_series_images(*_args, **_kwargs):
    raise NotImplementedError("Use generate_builder1_ad_image for incremental generation")


def generate_builder1_image(*_args, **_kwargs):
    raise NotImplementedError("Use generate_builder1_ad_image")
