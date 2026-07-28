"""
Builder2 durable final-video publication — HTTP artifact upload, distinct from local render.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_final_local_staging import is_legacy_headline_store_path
from engine.builder2_tournament_contracts import Builder2TournamentError

_UPLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "180").strip() or "180")


class Builder2FinalPublicationError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        stage: str = "publication",
        http_status: int | None = None,
    ) -> None:
        super().__init__(code)
        self.stage = stage
        self.http_status = http_status


@dataclass(frozen=True)
class FinalVideoPublicationResult:
    public_url: str
    output_token: str
    route_family: str
    upload_accepted: bool
    publisher_kind: str = "video_headline_artifact_upload"


def resolve_durable_final_video_publisher_kind() -> str:
    return "video_headline_artifact_upload"


def durable_publication_required() -> bool:
    return bool((os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip())


def publish_builder2_final_video(
    local_final_path: Path,
    public_base_url: str,
    *,
    job_id: str = "",
    output_token: str | None = None,
) -> FinalVideoPublicationResult:
    _ = job_id
    if not local_final_path.is_file():
        raise Builder2FinalPublicationError(
            "builder2_final_local_source_missing",
            stage="publication",
        )
    if is_legacy_headline_store_path(local_final_path):
        raise Builder2FinalPublicationError(
            "builder2_final_legacy_headline_store_rejected",
            stage="publication",
        )

    base = (public_base_url or "").strip().rstrip("/")
    if not base:
        from engine.public_base_url import resolve_public_base_url

        resolution = resolve_public_base_url()
        if resolution.configured:
            base = resolution.value
    if not base:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_missing_public_base_url",
            stage="publication",
        )

    upload_secret = (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()
    if not upload_secret:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_missing_upload_secret",
            stage="publication",
        )

    token = (output_token or uuid.uuid4().hex).strip()
    upload_endpoint = f"{base}/api/video-headline-artifact"
    try:
        with open(local_final_path, "rb") as handle:
            upload = requests.post(
                upload_endpoint,
                headers={"X-ACE-Video-Headline-Upload-Secret": upload_secret},
                files={"file": ("builder2_final.mp4", handle, "video/mp4")},
                data={"token": token},
                timeout=_UPLOAD_TIMEOUT,
            )
    except requests.RequestException as exc:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_failed",
            stage="publication",
        ) from exc

    if not upload.ok:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_failed",
            stage="publication",
            http_status=upload.status_code,
        )

    public_url = f"{base}/api/video-headline/{token}"
    return FinalVideoPublicationResult(
        public_url=public_url,
        output_token=token,
        route_family=classify_url_route_family(public_url),
        upload_accepted=True,
    )
