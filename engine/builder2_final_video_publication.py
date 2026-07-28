"""
Builder2 durable final-video publication — worker HTTP client; Web Service is durability authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_final_local_staging import is_legacy_headline_store_path
from engine.builder2_final_video_verification import verify_published_final_video_artifact
from engine.builder2_tournament_contracts import Builder2TournamentError

_UPLOAD_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFMPEG_TIMEOUT_SECONDS") or "180").strip() or "180")
_CAPABILITY_TIMEOUT = float((os.environ.get("VIDEO_HEADLINE_FFPROBE_TIMEOUT_SECONDS") or "30").strip() or "30")
_UPLOAD_SECRET_HEADER = "X-ACE-Video-Headline-Upload-Secret"


class Builder2FinalPublicationError(Builder2TournamentError):
    def __init__(
        self,
        code: str,
        *,
        stage: str = "publication",
        http_status: int | None = None,
        verification: "FinalVideoPublicationResult | None" = None,
        server_failure_code: str = "",
    ) -> None:
        super().__init__(code)
        self.stage = stage
        self.http_status = http_status
        self.verification = verification
        self.server_failure_code = server_failure_code


@dataclass(frozen=True)
class WebStorageCapabilityProbeResult:
    accepted: bool
    durable_storage_confirmed: bool
    publication_backend_kind: str
    storage_configured: bool
    storage_directory_exists: bool
    storage_writable: bool
    failure_code: str = ""
    http_status: int | None = None

    def to_report_dict(self) -> dict[str, bool | int | str | None]:
        return {
            "storageCapabilityAccepted": self.accepted,
            "webDurableStorageConfirmed": self.durable_storage_confirmed,
            "webPublicationBackendKind": self.publication_backend_kind or None,
            "webStorageConfigured": self.storage_configured,
            "webStorageDirectoryExists": self.storage_directory_exists,
            "webStorageWritable": self.storage_writable,
            "webStorageFailureCode": self.failure_code or None,
            "webStorageHttpStatusCode": self.http_status,
        }


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
    stored_byte_count: int = 0
    web_storage_configured: bool = False
    web_storage_writable: bool = False
    server_failure_code: str = ""
    publisher_kind: str = "builder2_final_video_artifact_upload"

    @property
    def upload_accepted(self) -> bool:
        return self.publication_accepted


def resolve_durable_final_video_publisher_kind() -> str:
    return "builder2_final_video_artifact_upload"


def durable_publication_required() -> bool:
    return bool((os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip())


def _upload_secret() -> str:
    return (os.environ.get("ACE_VIDEO_HEADLINE_UPLOAD_SECRET") or "").strip()


def _resolve_public_base(public_base_url: str) -> str:
    base = (public_base_url or "").strip().rstrip("/")
    if base:
        return base
    from engine.public_base_url import resolve_public_base_url

    resolution = resolve_public_base_url()
    return resolution.value if resolution.configured else ""


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


def _capability_from_payload(payload: Mapping[str, Any], *, http_status: int | None) -> WebStorageCapabilityProbeResult:
    durable = bool(payload.get("durableStorageConfirmed"))
    backend = str(payload.get("publicationBackendKind") or "")
    configured = bool(payload.get("storageConfigured"))
    exists = bool(payload.get("storageDirectoryExists"))
    writable = bool(payload.get("storageWritable"))
    ok = bool(payload.get("ok"))
    failure = str(payload.get("webStorageFailureCode") or payload.get("failureCode") or "")
    accepted = ok and durable and backend == "persistent_disk" and writable
    if not accepted and not failure:
        if not durable:
            failure = "builder2_web_storage_not_persistent"
        elif not writable:
            failure = "builder2_web_storage_not_writable"
        elif not ok:
            failure = "builder2_web_storage_not_configured"
    return WebStorageCapabilityProbeResult(
        accepted=accepted,
        durable_storage_confirmed=durable,
        publication_backend_kind=backend,
        storage_configured=configured,
        storage_directory_exists=exists,
        storage_writable=writable,
        failure_code=failure,
        http_status=http_status,
    )


def probe_builder2_final_video_web_storage_capability(public_base_url: str) -> WebStorageCapabilityProbeResult:
    if not _upload_secret():
        return WebStorageCapabilityProbeResult(
            accepted=False,
            durable_storage_confirmed=False,
            publication_backend_kind="unconfigured",
            storage_configured=False,
            storage_directory_exists=False,
            storage_writable=False,
            failure_code="builder2_final_publication_missing_upload_secret",
        )
    base = _resolve_public_base(public_base_url)
    if not base:
        return WebStorageCapabilityProbeResult(
            accepted=False,
            durable_storage_confirmed=False,
            publication_backend_kind="unconfigured",
            storage_configured=False,
            storage_directory_exists=False,
            storage_writable=False,
            failure_code="builder2_final_publication_missing_public_base_url",
        )
    endpoint = f"{base}/api/builder2-final-video-storage-capability"
    try:
        response = requests.get(endpoint, headers=_auth_headers(), timeout=_CAPABILITY_TIMEOUT)
    except requests.RequestException:
        return WebStorageCapabilityProbeResult(
            accepted=False,
            durable_storage_confirmed=False,
            publication_backend_kind="unconfigured",
            storage_configured=False,
            storage_directory_exists=False,
            storage_writable=False,
            failure_code="builder2_web_storage_capability_unreachable",
            http_status=None,
        )
    payload = _parse_json_response(response)
    result = _capability_from_payload(payload, http_status=int(response.status_code))
    if not response.ok and result.failure_code == "":
        return WebStorageCapabilityProbeResult(
            accepted=False,
            durable_storage_confirmed=result.durable_storage_confirmed,
            publication_backend_kind=result.publication_backend_kind,
            storage_configured=result.storage_configured,
            storage_directory_exists=result.storage_directory_exists,
            storage_writable=result.storage_writable,
            failure_code=str(payload.get("failureCode") or "builder2_web_storage_capability_rejected"),
            http_status=int(response.status_code),
        )
    return result


def _validate_upload_payload(
    payload: Mapping[str, Any],
    *,
    local_byte_count: int,
    token: str,
) -> tuple[str, str, int, int, bool, str, str, bool, bool]:
    server_code = str(payload.get("failureCode") or "")
    if not payload.get("ok"):
        raise Builder2FinalPublicationError(
            server_code or "builder2_final_publication_failed",
            stage="publication",
            server_failure_code=server_code,
        )
    if not payload.get("durableStorageConfirmed"):
        raise Builder2FinalPublicationError(
            server_code or "final_publication_not_durable",
            stage="publication",
            server_failure_code=server_code or "final_publication_not_durable",
        )
    backend = str(payload.get("publicationBackendKind") or "")
    if backend != "persistent_disk":
        raise Builder2FinalPublicationError(
            server_code or "final_publication_not_durable",
            stage="publication",
            server_failure_code=server_code or "builder2_web_storage_not_persistent",
        )
    public_url = str(payload.get("finalPublicUrl") or "").strip()
    if not public_url or classify_url_route_family(public_url) != "api/builder2-final-video":
        raise Builder2FinalPublicationError(
            server_code or "final_publication_verification_failed",
            stage="publication",
            server_failure_code=server_code,
        )
    uploaded = int(payload.get("uploadedByteCount") or 0)
    stored = int(payload.get("storedByteCount") or 0)
    if uploaded != local_byte_count or stored != local_byte_count:
        raise Builder2FinalPublicationError(
            server_code or "final_publication_size_mismatch",
            stage="publication",
            server_failure_code=server_code or "builder2_web_storage_verification_failed",
        )
    fingerprint_ok = bool(payload.get("artifactFingerprintVerified"))
    if not fingerprint_ok:
        raise Builder2FinalPublicationError(
            server_code or "builder2_web_storage_verification_failed",
            stage="publication",
            server_failure_code=server_code or "builder2_web_storage_verification_failed",
        )
    output_token = str(payload.get("outputToken") or token).strip()
    return (
        public_url,
        output_token,
        uploaded,
        stored,
        fingerprint_ok,
        backend,
        server_code,
        bool(payload.get("storageConfigured")),
        bool(payload.get("storageWritable")),
    )


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

    base = _resolve_public_base(public_base_url)
    if not base:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_missing_public_base_url",
            stage="publication",
        )
    if not _upload_secret():
        raise Builder2FinalPublicationError(
            "builder2_final_publication_missing_upload_secret",
            stage="publication",
        )

    local_byte_count = int(local_final_path.stat().st_size)
    if local_byte_count <= 0:
        raise Builder2FinalPublicationError(
            "final_publication_verification_failed",
            stage="publication",
        )

    token = (output_token or uuid.uuid4().hex).strip()
    source_fingerprint = hashlib.sha256(local_final_path.read_bytes()).hexdigest()
    upload_endpoint = f"{base}/api/builder2-final-video-artifact"
    try:
        with open(local_final_path, "rb") as handle:
            upload = requests.post(
                upload_endpoint,
                headers=_auth_headers(),
                files={"file": ("builder2_final.mp4", handle, "video/mp4")},
                data={"token": token, "sourceFingerprint": source_fingerprint},
                timeout=_UPLOAD_TIMEOUT,
            )
    except requests.RequestException as exc:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_failed",
            stage="publication",
        ) from exc

    payload = _parse_json_response(upload)
    if not upload.ok:
        server_code = str(payload.get("failureCode") or "")
        raise Builder2FinalPublicationError(
            server_code or "builder2_final_publication_failed",
            stage="publication",
            http_status=upload.status_code,
            server_failure_code=server_code,
        )

    try:
        (
            public_url,
            output_token_resolved,
            uploaded,
            stored,
            fingerprint_ok,
            backend,
            server_code,
            storage_configured,
            storage_writable,
        ) = _validate_upload_payload(payload, local_byte_count=local_byte_count, token=token)
    except Builder2FinalPublicationError:
        raise
    except Exception as exc:
        raise Builder2FinalPublicationError(
            "builder2_final_publication_failed",
            stage="publication",
            http_status=upload.status_code,
        ) from exc

    verification = verify_published_final_video_artifact(
        public_url,
        expected_byte_count=local_byte_count,
        durable_storage_confirmed=True,
    )
    result = FinalVideoPublicationResult(
        public_url=public_url,
        output_token=output_token_resolved,
        route_family=classify_url_route_family(public_url),
        publication_accepted=verification.post_upload_verification_accepted,
        durable_storage_confirmed=True,
        publication_backend_kind=backend,
        publication_reference_present=True,
        uploaded_byte_count=uploaded,
        stored_byte_count=stored,
        post_upload_verification_attempted=verification.post_upload_verification_attempted,
        post_upload_verification_accepted=verification.post_upload_verification_accepted,
        post_upload_http_status_code=verification.final_url_http_status_code,
        post_upload_content_type=verification.final_url_content_type,
        post_upload_content_length=verification.final_url_content_length,
        artifact_fingerprint_verified=fingerprint_ok and verification.artifact_fingerprint_verified,
        web_storage_configured=storage_configured,
        web_storage_writable=storage_writable,
        server_failure_code=server_code,
    )
    if not verification.post_upload_verification_accepted:
        raise Builder2FinalPublicationError(
            verification.failure_code or "final_publication_verification_failed",
            stage="publication_verification",
            http_status=verification.final_url_http_status_code,
            verification=result,
        )
    return result
