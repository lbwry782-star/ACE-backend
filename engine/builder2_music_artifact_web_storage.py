"""
Builder2 music artifact Web Service persistence — disk write authority on Web.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

from engine.builder2_final_video_web_storage import assess_builder2_final_video_web_storage_capability
from engine.builder2_music_artifact_store import (
    classify_music_artifact_backend_kind,
    get_builder2_music_artifact_path,
    write_builder2_music_artifact_bytes,
)
from engine.public_base_url import resolve_public_base_url

logger = logging.getLogger(__name__)

_TOKEN_RE = __import__("re").compile(r"^[a-f0-9]{32}$")


@dataclass(frozen=True)
class MusicArtifactStoreResult:
    ok: bool
    durable_storage_confirmed: bool
    publication_backend_kind: str
    storage_configured: bool
    storage_writable: bool
    uploaded_byte_count: int
    stored_byte_count: int
    artifact_fingerprint_verified: bool
    music_artifact_url: str
    output_token: str
    failure_code: str = ""

    def to_upload_response_dict(self) -> dict[str, bool | int | str | None]:
        return {
            "ok": self.ok,
            "durableStorageConfirmed": self.durable_storage_confirmed,
            "publicationBackendKind": self.publication_backend_kind or None,
            "storageConfigured": self.storage_configured,
            "storageWritable": self.storage_writable,
            "uploadedByteCount": self.uploaded_byte_count,
            "storedByteCount": self.stored_byte_count,
            "artifactFingerprintVerified": self.artifact_fingerprint_verified,
            "musicArtifactUrl": self.music_artifact_url or None,
            "outputToken": self.output_token or None,
            "failureCode": self.failure_code or None,
        }


def _resolve_public_base_url() -> str:
    resolution = resolve_public_base_url()
    return resolution.value.rstrip("/") if resolution.configured else ""


def persist_builder2_music_artifact(
    token: str,
    data: bytes,
    *,
    source_fingerprint: str = "",
) -> MusicArtifactStoreResult:
    t = (token or "").strip()
    uploaded = len(data)
    if not _TOKEN_RE.match(t):
        return MusicArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind="unconfigured",
            storage_configured=False,
            storage_writable=False,
            uploaded_byte_count=uploaded,
            stored_byte_count=0,
            artifact_fingerprint_verified=False,
            music_artifact_url="",
            output_token=t,
            failure_code="builder2_music_artifact_invalid_token",
        )

    capability = assess_builder2_final_video_web_storage_capability()
    backend = classify_music_artifact_backend_kind()
    if not capability.ok or backend != "persistent_disk":
        return MusicArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=backend,
            storage_configured=capability.storage_configured,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=0,
            artifact_fingerprint_verified=False,
            music_artifact_url="",
            output_token=t,
            failure_code=capability.failure_code or "builder2_web_storage_not_persistent",
        )

    if not write_builder2_music_artifact_bytes(t, data):
        return MusicArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=backend,
            storage_configured=True,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=0,
            artifact_fingerprint_verified=False,
            music_artifact_url="",
            output_token=t,
            failure_code="builder2_music_artifact_write_failed",
        )

    stored_path = get_builder2_music_artifact_path(t)
    stored_count = int(stored_path.stat().st_size) if stored_path and stored_path.is_file() else 0
    fingerprint_ok = stored_count == uploaded and uploaded > 0
    if source_fingerprint:
        try:
            digest = hashlib.sha256(stored_path.read_bytes()).hexdigest() if stored_path else ""
            fingerprint_ok = fingerprint_ok and digest == source_fingerprint
        except OSError:
            fingerprint_ok = False

    if stored_count != uploaded:
        return MusicArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=backend,
            storage_configured=True,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=stored_count,
            artifact_fingerprint_verified=False,
            music_artifact_url="",
            output_token=t,
            failure_code="builder2_music_artifact_verification_failed",
        )

    base = _resolve_public_base_url()
    music_url = f"{base}/api/builder2-music-artifact/{t}" if base else ""
    if not music_url:
        return MusicArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=backend,
            storage_configured=True,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=stored_count,
            artifact_fingerprint_verified=fingerprint_ok,
            music_artifact_url="",
            output_token=t,
            failure_code="builder2_web_storage_not_configured",
        )

    logger.info(
        "BUILDER2_MUSIC_ARTIFACT_STORED requestAccepted=true storedByteCount=%s backendKind=%s",
        stored_count,
        backend,
    )
    return MusicArtifactStoreResult(
        ok=True,
        durable_storage_confirmed=True,
        publication_backend_kind=backend,
        storage_configured=True,
        storage_writable=capability.storage_writable,
        uploaded_byte_count=uploaded,
        stored_byte_count=stored_count,
        artifact_fingerprint_verified=fingerprint_ok,
        music_artifact_url=music_url,
        output_token=t,
    )


def log_builder2_music_artifact_served(*, stored_byte_count: int, request_accepted: bool) -> None:
    logger.info(
        "BUILDER2_MUSIC_ARTIFACT_SERVED requestAccepted=%s storedByteCount=%s durableStorageConfirmed=%s",
        request_accepted,
        stored_byte_count,
        request_accepted,
    )
