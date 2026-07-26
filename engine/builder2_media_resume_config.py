"""
Builder2 media-resume configuration — canonical runtime object for dry run and execution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from engine.builder2_runway_config import (
    BUILDER2_RUNWAY_VIDEO_RATIO,
    resolve_builder2_runway_video_model,
    resolve_builder2_video_duration_seconds,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.public_base_url import PublicBaseUrlResolution, require_public_base_url


@dataclass(frozen=True)
class MediaResumeConfiguration:
    public_base_url: PublicBaseUrlResolution
    publicBaseUrl: str
    runwayModel: str
    durationSeconds: int
    ratio: str
    runwayConfigured: bool
    startImageConfigured: bool
    ffmpegRequired: bool

    def to_safe_metadata(self) -> Dict[str, Any]:
        return {
            "publicBaseUrl": self.publicBaseUrl,
            "publicBaseUrlSource": self.public_base_url.source,
            "runwayModel": self.runwayModel,
            "durationSeconds": self.durationSeconds,
            "ratio": self.ratio,
            "runwayConfigured": self.runwayConfigured,
            "startImageConfigured": self.startImageConfigured,
            "ffmpegRequired": self.ffmpegRequired,
        }


def build_media_resume_configuration(
    *,
    job_id: str,
    job_data: Optional[Dict[str, Any]] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
    start_image_required: bool,
    ffmpeg_required: bool,
) -> MediaResumeConfiguration:
    public = require_public_base_url(job_data=job_data, tournament_state=tournament_state)
    runway_model = resolve_builder2_runway_video_model()
    duration_seconds = resolve_builder2_video_duration_seconds()
    runway_key = bool((os.environ.get("RUNWAY_API_KEY") or "").strip())
    openai_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    if not runway_key:
        raise Builder2TournamentError("builder2_media_resume_not_configured:runwayApiKey")
    if start_image_required and not openai_key:
        raise Builder2TournamentError("builder2_media_resume_not_configured:openaiApiKey")
    return MediaResumeConfiguration(
        public_base_url=public,
        publicBaseUrl=public.value,
        runwayModel=runway_model,
        durationSeconds=duration_seconds,
        ratio=BUILDER2_RUNWAY_VIDEO_RATIO,
        runwayConfigured=runway_key,
        startImageConfigured=(not start_image_required) or openai_key,
        ffmpegRequired=ffmpeg_required,
    )
