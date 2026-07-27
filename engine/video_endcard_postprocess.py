"""
Builder2 Advertising Closure end-card rendering — append plain-text end card after Runway video.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    ClosureRenderResult,
    render_builder2_advertising_closure_endcard,
)

logger = logging.getLogger(__name__)

_DEFAULT_DURATION = 1.5


def append_advertising_closure_endcard(
    source_video_url: str,
    public_base_url: str,
    *,
    product_name: str,
    slogan: str,
    language: str = "he",
    duration_seconds: float = _DEFAULT_DURATION,
    job_id: str = "",
    ffmpeg_runner: Optional[Callable[..., None]] = None,
) -> str:
    adapter = None
    if ffmpeg_runner is not None:

        def adapter(cmd: list[str], stage: str, category: str) -> None:
            ffmpeg_runner(cmd)

    result = render_builder2_advertising_closure_endcard(
        source_video_url,
        public_base_url,
        product_name=product_name,
        slogan=slogan,
        language=language,
        duration_seconds=duration_seconds,
        job_id=job_id,
        publish=True,
        ffmpeg_runner=adapter,
    )
    return result.public_url


__all__ = [
    "append_advertising_closure_endcard",
    "Builder2ClosureRenderError",
    "ClosureRenderResult",
    "render_builder2_advertising_closure_endcard",
]
