"""
Builder2 music artifact publication — Worker HTTP client; Web Service is durability authority.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import requests

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_lyria_config import resolve_builder2_lyria_job_artifact_path
from engine.builder2_tournament_contracts import Builder2TournamentError

logger = logging.getLogger(__name__)

_UPLOAD_TIMEOUT = float((os.environ.get("BUILDER2_LYRIA_HTTP_TIMEOUT_SECONDS") or "300").strip() or "300")
_DOWNLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_DOWNLOAD_TIMEOUT_SECONDS") or "180").strip() or "180")
_UPLOAD_SECRET_HEADER = "X-ACE-Video-Headline-Upload-Secret"
_MAX_MUSIC_UPLOAD_BYTES = int(
    (os.environ.get("BUILDER2_MUSIC_ARTIFACT_MAX_UPLOAD_BYTES") or "52428800").strip() or "52428800"
)


class Builder2MusicPublicationError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        stage: str = "music_publication",
        http_status: int | None = None,
        server_failure_code: str = "",
    ) -> None:
        super().__init__(code)
        self.stage = stage
        self.http_status = http_status
        self.server_failure_code = server_failure_code


@dataclass(frozen=True)
class MusicArtifactPublicationResult:
    music_artifact_url: str
    output_token: str
    publication_accepted: bool
    durable_storage_confirmed: bool
    publication_backend_kind: str
    uploaded_byte_count: int
    stored_byte_count: int
    artifact_fingerprint_verified: bool
    server_failure_code: str = ""


def _upload_secret() -> str:
    return (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()


def _resolve_public_base(public_base_url: str) -> str:
    base = (public_base_url or "").strip().rstrip("/")
    if base:
        return base
    from engine.public_base_url import resolve_public_base_url

    resolution = resolve_public_base_url()
    return resolution.value.rstrip("/") if resolution.configured else ""


def _auth_headers() -> dict[str, str]:
    secret = _upload_secret()
    if not secret:
        return {}
    return {_UPLOAD_SECRET_HEADER: secret}


def _parse_json_response(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def publish_builder2_music_artifact(
    local_mp3_path: Path,
    public_base_url: str,
    *,
    job_id: str = "",
    output_token: str | None = None,
) -> MusicArtifactPublicationResult:
    _ = job_id
    if not local_mp3_path.is_file():
        raise Builder2MusicPublicationError("builder2_music_artifact_local_missing")
    base = _resolve_public_base(public_base_url)
    if not base:
        raise Builder2MusicPublicationError("builder2_music_artifact_missing_public_base_url")
    if not _upload_secret():
        raise Builder2MusicPublicationError("builder2_music_artifact_missing_upload_secret")

    local_byte_count = int(local_mp3_path.stat().st_size)
    if local_byte_count <= 0 or local_byte_count > _MAX_MUSIC_UPLOAD_BYTES:
        raise Builder2MusicPublicationError("builder2_music_artifact_invalid_size")

    token = (output_token or uuid.uuid4().hex).strip()
    source_fingerprint = hashlib.sha256(local_mp3_path.read_bytes()).hexdigest()
    upload_endpoint = f"{base}/api/builder2-music-artifact"
    try:
        with open(local_mp3_path, "rb") as handle:
            upload = requests.post(
                upload_endpoint,
                headers=_auth_headers(),
                files={"file": ("soundtrack.mp3", handle, "audio/mpeg")},
                data={"token": token, "sourceFingerprint": source_fingerprint},
                timeout=_UPLOAD_TIMEOUT,
            )
    except requests.RequestException as exc:
        raise Builder2MusicPublicationError("builder2_music_artifact_upload_failed") from exc

    payload = _parse_json_response(upload)
    if not upload.ok or not payload.get("ok"):
        server_code = str(payload.get("failureCode") or "")
        raise Builder2MusicPublicationError(
            server_code or "builder2_music_artifact_upload_rejected",
            http_status=int(upload.status_code),
            server_failure_code=server_code,
        )
    if not payload.get("durableStorageConfirmed"):
        server_code = str(payload.get("failureCode") or "builder2_music_artifact_not_durable")
        raise Builder2MusicPublicationError(server_code, server_failure_code=server_code)

    backend = str(payload.get("publicationBackendKind") or "")
    if backend != "persistent_disk":
        raise Builder2MusicPublicationError(
            "builder2_music_artifact_not_durable",
            server_failure_code="builder2_web_storage_not_persistent",
        )

    music_url = str(payload.get("musicArtifactUrl") or "").strip()
    if not music_url or classify_url_route_family(music_url) != "api/builder2-music-artifact":
        raise Builder2MusicPublicationError("builder2_music_artifact_invalid_public_url")

    stored = int(payload.get("storedByteCount") or 0)
    uploaded = int(payload.get("uploadedByteCount") or local_byte_count)
    fingerprint_ok = bool(payload.get("artifactFingerprintVerified"))
    if stored != uploaded or not fingerprint_ok:
        raise Builder2MusicPublicationError("builder2_music_artifact_verification_failed")

    return MusicArtifactPublicationResult(
        music_artifact_url=music_url,
        output_token=str(payload.get("outputToken") or token),
        publication_accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind=backend,
        uploaded_byte_count=uploaded,
        stored_byte_count=stored,
        artifact_fingerprint_verified=fingerprint_ok,
    )


def download_builder2_music_artifact_to_local(
    *,
    music_artifact_url: str,
    job_id: str,
    session: Optional[requests.Session] = None,
) -> Path:
    url = (music_artifact_url or "").strip()
    if not url or classify_url_route_family(url) != "api/builder2-music-artifact":
        raise Builder2MusicPublicationError("builder2_music_artifact_invalid_download_url")

    http = session or requests.Session()
    try:
        response = http.get(url, timeout=_DOWNLOAD_TIMEOUT)
    except requests.RequestException as exc:
        raise Builder2MusicPublicationError("builder2_music_artifact_download_failed") from exc
    if response.status_code >= 400 or not response.content:
        raise Builder2MusicPublicationError(
            "builder2_music_artifact_download_not_found",
            http_status=int(response.status_code),
        )

    dest = resolve_builder2_lyria_job_artifact_path(job_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    if dest.stat().st_size <= 0:
        raise Builder2MusicPublicationError("builder2_music_artifact_download_empty")
    return dest


def extract_music_artifact_token(reference: Mapping[str, Any]) -> str:
    token = str(reference.get("musicArtifactToken") or reference.get("musicOutputToken") or "").strip()
    if token:
        return token
    url = str(reference.get("musicArtifactUrl") or "").strip()
    if not url:
        return ""
    path = url.rstrip("/").split("/")[-1]
    return path if len(path) == 32 else ""


def durable_music_reference_present(media: Mapping[str, Any]) -> bool:
    if extract_music_artifact_token(media):
        return True
    url = str(media.get("musicArtifactUrl") or "").strip()
    return bool(url) and classify_url_route_family(url) == "api/builder2-music-artifact"
