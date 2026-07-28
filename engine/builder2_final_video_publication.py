"""
Builder2 durable final-video publication — upload, durable-store check, bounded verification.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_final_local_staging import is_legacy_headline_store_path
from engine.builder2_final_video_store import (
    classify_publication_backend_kind,
    is_durable_publication_backend,
)
from engine.builder2_final_video_verification import verify_published_final_video_artifact
from engine.builder2_tournament_contracts import Builder2TournamentError

_UPLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "180").strip() or "180")


class Builder2FinalPublicationError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        stage: str = "publication",
        http_status: int | None = None,
        verification: "FinalVideoPublicationResult | None" = None,
    ) -> None:
        super().__init__(code)
        self.stage = stage
        self.http_status = http_status
        self.verification = verification


@dataclass(frozen=True)
class FinalVideoPublicationResult:
    public_url: str
    output_token: str
    route_family: str
    publication_accepted: bool
    durable_storage_confirmed: bool
    publication_backend_kind: str
    publication_reference_present: bool
    uploaded_byte_count: int
    post_upload_verification_attempted: bool
    post_upload_verification_accepted: bool
    post_upload_http_status_code: int | None
    post_upload_content_type: str
    post_upload_content_length: int | None
    artifact_fingerprint_verified: bool
    publisher_kind: str = "builder2_final_video_artifact_upload"

    @property
    def upload_accepted(self) -> bool:
        return self.publication_accepted


def resolve_durable_final_video_publisher_kind() -> str:
    return "builder2_final_video_artifact_upload"


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

    backend_kind = classify_publication_backend_kind()
    durable_confirmed = is_durable_publication_backend()
    if not durable_confirmed:
        raise Builder2FinalPublicationError(
            "final_publication_not_durable",
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

    uploaded_byte_count = int(local_final_path.stat().st_size)
    if uploaded_byte_count <= 0:
        raise Builder2FinalPublicationError(
            "final_publication_verification_failed",
            stage="publication",
        )

    token = (output_token or uuid.uuid4().hex).strip()
    upload_endpoint = f"{base}/api/builder2-final-video-artifact"
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

    public_url = f"{base}/api/builder2-final-video/{token}"
    verification = verify_published_final_video_artifact(
        public_url,
        expected_byte_count=uploaded_byte_count,
        durable_storage_confirmed=durable_confirmed,
    )
    result = FinalVideoPublicationResult(
        public_url=public_url,
        output_token=token,
        route_family=classify_url_route_family(public_url),
        publication_accepted=verification.post_upload_verification_accepted,
        durable_storage_confirmed=verification.durable_storage_confirmed,
        publication_backend_kind=backend_kind,
        publication_reference_present=True,
        uploaded_byte_count=uploaded_byte_count,
        post_upload_verification_attempted=verification.post_upload_verification_attempted,
        post_upload_verification_accepted=verification.post_upload_verification_accepted,
        post_upload_http_status_code=verification.final_url_http_status_code,
        post_upload_content_type=verification.final_url_content_type,
        post_upload_content_length=verification.final_url_content_length,
        artifact_fingerprint_verified=verification.artifact_fingerprint_verified,
    )
    if not verification.post_upload_verification_accepted:
        raise Builder2FinalPublicationError(
            verification.failure_code or "final_publication_verification_failed",
            stage="publication_verification",
            http_status=verification.final_url_http_status_code,
            verification=result,
        )
    return result
