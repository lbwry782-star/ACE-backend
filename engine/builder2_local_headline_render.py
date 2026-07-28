"""
Builder2 local headline overlay rendering from accepted Winner plan — no publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from engine.builder2_winner_downstream import (
    AcceptedWinnerHeadlineResolution,
    apply_accepted_headline_resolution_observability,
    resolve_accepted_winner_headline_for_media,
)
from engine.video_bidi import prepare_ffmpeg_overlay_headline
from engine.video_headline_postprocess import (
    VideoHeadlineRenderError,
    render_local_video_headline_overlay,
)


@dataclass
class Builder2LocalHeadlineRenderResult:
    output_path: Path
    measured_duration_seconds: float
    headline_resolution: AcceptedWinnerHeadlineResolution


def resolve_builder2_finalization_headline(
    plan: Dict[str, Any],
    *,
    report: Optional[Dict[str, Any]] = None,
) -> AcceptedWinnerHeadlineResolution:
    resolution = resolve_accepted_winner_headline_for_media(plan)
    if report is not None:
        apply_accepted_headline_resolution_observability(report, resolution)
    return resolution


def render_builder2_accepted_headline_overlay(
    *,
    source_video_path: Path,
    output_path: Path,
    plan: Dict[str, Any],
    report: Optional[Dict[str, Any]] = None,
) -> Builder2LocalHeadlineRenderResult:
    resolution = resolve_builder2_finalization_headline(plan, report=report)
    if not resolution.headline_required:
        raise VideoHeadlineRenderError(
            "headline_omitted_by_decision",
            stage="canonical_headline_resolution",
        )
    if resolution.failure_code:
        raise VideoHeadlineRenderError(
            resolution.failure_code,
            stage=resolution.failure_stage or "canonical_headline_resolution",
        )
    headline = resolution.headline_text.strip()
    if not headline:
        raise VideoHeadlineRenderError(
            "local_renderer_received_empty_headline",
            stage="input_validation",
        )
    video_lang = resolution.language or str(plan.get("language") or "en")
    canonical_name = resolution.product_name_resolved or str(plan.get("productNameResolved") or "")
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
        headline_resolution=resolution,
    )


__all__ = [
    "Builder2LocalHeadlineRenderResult",
    "VideoHeadlineRenderError",
    "render_builder2_accepted_headline_overlay",
    "resolve_builder2_finalization_headline",
]
