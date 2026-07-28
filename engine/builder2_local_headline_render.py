"""
Builder2 local headline overlay rendering from accepted Winner plan — no publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from engine.video_bidi import prepare_ffmpeg_overlay_headline
from engine.video_headline_postprocess import (
    VideoHeadlineRenderError,
    render_local_video_headline_overlay,
)


@dataclass
class Builder2LocalHeadlineRenderResult:
    output_path: Path
    measured_duration_seconds: float


def render_builder2_accepted_headline_overlay(
    *,
    source_video_path: Path,
    output_path: Path,
    plan: Dict[str, Any],
) -> Builder2LocalHeadlineRenderResult:
    headline = str(plan.get("headlineText") or "").strip()
    video_lang = str(plan.get("language") or "en")
    canonical_name = str(plan.get("productNameResolved") or "")
    overlay_prep = prepare_ffmpeg_overlay_headline(
        headline,
        content_language=video_lang,
        canonical_name=canonical_name,
    )
    result = render_local_video_headline_overlay(
        source_video_path=source_video_path,
        output_path=output_path,
        headline=overlay_prep.text_plain,
        overlay_language=video_lang,
        overlay_render_mode=overlay_prep.render_mode,
        overlay_dual_latin=overlay_prep.dual_latin,
        overlay_dual_hebrew=overlay_prep.dual_hebrew,
        overlay_canonical_name=canonical_name,
    )
    return Builder2LocalHeadlineRenderResult(
        output_path=result.output_path,
        measured_duration_seconds=result.measured_duration_seconds,
    )


__all__ = [
    "Builder2LocalHeadlineRenderResult",
    "VideoHeadlineRenderError",
    "render_builder2_accepted_headline_overlay",
]
