"""
Builder2 final-video publication verification — bounded HTTP checks, no full download.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from engine.builder2_closure_render import classify_url_route_family

_VERIFY_TIMEOUT = 30.0
_ACCEPTED_VIDEO_PREFIXES = ("video/",)
_RANGE_BYTES = "bytes=0-1023"


@dataclass(frozen=True)
class FinalVideoArtifactVerification:
    final_url_accessible: bool
    final_url_http_status_code: Optional[int]
    final_url_content_type: str
    final_url_content_length: Optional[int]
    final_artifact_looks_like_video: bool
    durable_storage_confirmed: bool
    post_upload_verification_attempted: bool
    post_upload_verification_accepted: bool
    artifact_fingerprint_verified: bool
    failure_code: str = ""

    def to_report_dict(self) -> dict[str, bool | int | str | None]:
        return {
            "finalUrlAccessible": self.final_url_accessible,
            "finalUrlHttpStatusCode": self.final_url_http_status_code,
            "finalUrlContentType": self.final_url_content_type or None,
            "finalUrlContentLength": self.final_url_content_length,
            "finalArtifactLooksLikeVideo": self.final_artifact_looks_like_video,
            "durableStorageConfirmed": self.durable_storage_confirmed,
            "postUploadVerificationAttempted": self.post_upload_verification_attempted,
            "postUploadVerificationAccepted": self.post_upload_verification_accepted,
            "artifactFingerprintVerified": self.artifact_fingerprint_verified,
            "finalPublicationVerificationFailureCode": self.failure_code or None,
        }


def _looks_like_json_not_found(body: bytes) -> bool:
    if not body:
        return False
    sample = body[:512].strip()
    if not sample.startswith(b"{"):
        return False
    try:
        payload = json.loads(sample.decode("utf-8", errors="replace"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    error = str(payload.get("error") or "").strip().lower()
    return error == "not_found" or payload.get("ok") is False


def _content_type_is_video(content_type: str) -> bool:
    token = (content_type or "").split(";", 1)[0].strip().lower()
    return any(token.startswith(prefix) for prefix in _ACCEPTED_VIDEO_PREFIXES)


def _size_matches(expected: Optional[int], observed: Optional[int]) -> bool:
    if expected is None or observed is None:
        return observed is None or observed > 0
    return observed == expected


def verify_published_final_video_artifact(
    public_url: str,
    *,
    expected_byte_count: Optional[int] = None,
    durable_storage_confirmed: bool = False,
    timeout: float = _VERIFY_TIMEOUT,
) -> FinalVideoArtifactVerification:
    url = (public_url or "").strip()
    if not url:
        return FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=None,
            final_url_content_type="",
            final_url_content_length=None,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_url_inaccessible",
        )
    if classify_url_route_family(url) not in {"api/builder2-final-video", "api/video-headline"}:
        return FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=None,
            final_url_content_type="",
            final_url_content_length=None,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_verification_failed",
        )

    status_code: Optional[int] = None
    content_type = ""
    content_length: Optional[int] = None
    body_prefix = b""
    try:
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        status_code = int(head.status_code)
        content_type = str(head.headers.get("Content-Type") or "")
        raw_length = head.headers.get("Content-Length")
        if raw_length is not None and str(raw_length).strip().isdigit():
            content_length = int(str(raw_length).strip())
    except requests.RequestException:
        status_code = None

    if status_code not in {200, 206} or content_length is None or content_length <= 0:
        try:
            ranged = requests.get(
                url,
                headers={"Range": _RANGE_BYTES},
                timeout=timeout,
                allow_redirects=True,
            )
            status_code = int(ranged.status_code)
            content_type = str(ranged.headers.get("Content-Type") or content_type)
            raw_length = ranged.headers.get("Content-Length")
            if raw_length is not None and str(raw_length).strip().isdigit():
                content_length = int(str(raw_length).strip())
            body_prefix = ranged.content[:512] if ranged.content else b""
        except requests.RequestException:
            return FinalVideoArtifactVerification(
                final_url_accessible=False,
                final_url_http_status_code=status_code,
                final_url_content_type=content_type,
                final_url_content_length=content_length,
                final_artifact_looks_like_video=False,
                durable_storage_confirmed=durable_storage_confirmed,
                post_upload_verification_attempted=True,
                post_upload_verification_accepted=False,
                artifact_fingerprint_verified=False,
                failure_code="final_publication_url_inaccessible",
            )

    if status_code not in {200, 206}:
        failure = "final_publication_artifact_missing"
        if _looks_like_json_not_found(body_prefix):
            failure = "final_publication_artifact_missing"
        return FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=status_code,
            final_url_content_type=content_type,
            final_url_content_length=content_length,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code=failure,
        )

    if _looks_like_json_not_found(body_prefix):
        return FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=status_code,
            final_url_content_type=content_type,
            final_url_content_length=content_length,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_artifact_missing",
        )

    looks_like_video = _content_type_is_video(content_type)
    if not looks_like_video and body_prefix.startswith(b"\x00\x00\x00"):
        looks_like_video = True
    if not looks_like_video and len(body_prefix) >= 8 and body_prefix[4:8] == b"ftyp":
        looks_like_video = True

    if not looks_like_video:
        return FinalVideoArtifactVerification(
            final_url_accessible=True,
            final_url_http_status_code=status_code,
            final_url_content_type=content_type,
            final_url_content_length=content_length,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_content_type_invalid",
        )

    if content_length is not None and content_length <= 0:
        return FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=status_code,
            final_url_content_type=content_type,
            final_url_content_length=content_length,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_verification_failed",
        )

    if expected_byte_count is not None and content_length is not None and content_length != expected_byte_count:
        return FinalVideoArtifactVerification(
            final_url_accessible=True,
            final_url_http_status_code=status_code,
            final_url_content_type=content_type,
            final_url_content_length=content_length,
            final_artifact_looks_like_video=looks_like_video,
            durable_storage_confirmed=durable_storage_confirmed,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_size_mismatch",
        )

    if not durable_storage_confirmed:
        return FinalVideoArtifactVerification(
            final_url_accessible=True,
            final_url_http_status_code=status_code,
            final_url_content_type=content_type,
            final_url_content_length=content_length,
            final_artifact_looks_like_video=looks_like_video,
            durable_storage_confirmed=False,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=_size_matches(expected_byte_count, content_length),
            failure_code="final_publication_not_durable",
        )

    accepted = looks_like_video and _size_matches(expected_byte_count, content_length)
    return FinalVideoArtifactVerification(
        final_url_accessible=True,
        final_url_http_status_code=status_code,
        final_url_content_type=content_type,
        final_url_content_length=content_length,
        final_artifact_looks_like_video=looks_like_video,
        durable_storage_confirmed=True,
        post_upload_verification_attempted=True,
        post_upload_verification_accepted=accepted,
        artifact_fingerprint_verified=accepted,
        failure_code="" if accepted else "final_publication_verification_failed",
    )


def extract_builder2_final_video_token(public_url: str) -> str:
    path = (urlparse((public_url or "").strip()).path or "").lower()
    for prefix in ("/api/builder2-final-video/", "/builder2-final-video/"):
        if prefix in path:
            token = path.rsplit("/", 1)[-1].split(".", 1)[0]
            return token
    return ""
