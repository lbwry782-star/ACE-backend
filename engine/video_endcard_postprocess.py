"""
Builder2 advertising-closure end-card rendering — append plain-text end card after Runway video.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    ClosureRenderResult,
    render_builder2_advertising_closure_endcard,
)
from engine.builder2_final_video_publication import publish_builder2_final_video

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

    with tempfile.TemporaryDirectory(prefix="ace_closure_publish_") as workdir:
        local_final = Path(workdir) / "builder2_final.mp4"
        result = render_builder2_advertising_closure_endcard(
            source_video_url,
            product_name=product_name,
            slogan=slogan,
            output_path=local_final,
            language=language,
            duration_seconds=duration_seconds,
            job_id=job_id,
            ffmpeg_runner=adapter,
        )
        publication = publish_builder2_final_video(
            Path(result.local_path or local_final),
            public_base_url,
            job_id=job_id,
            output_token=result.output_token,
        )
    return publication.public_url


__all__ = [
    "append_advertising_closure_endcard",
    "Builder2ClosureRenderError",
    "ClosureRenderResult",
    "render_builder2_advertising_closure_endcard",
]
