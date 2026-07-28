"""
Builder2 final-video Web Service storage — capability, persist, and classification.

Runs only in the Web Service process. Workers must not use these helpers to infer
local durability; they call HTTP capability/upload endpoints instead.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.builder2_final_video_store import (
    _TOKEN_RE,
    explicit_builder2_final_storage_configured,
    get_builder2_final_video_path,
    resolve_builder2_final_video_storage_root,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebStorageCapability:
    ok: bool
    durable_storage_confirmed: bool
    publication_backend_kind: str
    storage_configured: bool
    storage_directory_exists: bool
    storage_writable: bool
    final_video_upload_route_available: bool = True
    final_video_serve_route_available: bool = True
    failure_code: str = ""

    def to_report_dict(self) -> dict[str, bool | str | None]:
        return {
            "ok": self.ok,
            "durableStorageConfirmed": self.durable_storage_confirmed,
            "publicationBackendKind": self.publication_backend_kind or None,
            "storageConfigured": self.storage_configured,
            "storageDirectoryExists": self.storage_directory_exists,
            "storageWritable": self.storage_writable,
            "finalVideoUploadRouteAvailable": self.final_video_upload_route_available,
            "finalVideoServeRouteAvailable": self.final_video_serve_route_available,
            "webStorageFailureCode": self.failure_code or None,
        }


@dataclass(frozen=True)
class WebArtifactStoreResult:
    ok: bool
    durable_storage_confirmed: bool
    publication_backend_kind: str
    storage_configured: bool
    storage_writable: bool
    uploaded_byte_count: int
    stored_byte_count: int
    artifact_fingerprint_verified: bool
    final_public_url: str
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
            "finalPublicUrl": self.final_public_url or None,
            "outputToken": self.output_token or None,
            "failureCode": self.failure_code or None,
        }


def _system_temp_roots() -> set[Path]:
    roots: set[Path] = set()
    for candidate in (tempfile.gettempdir(), "/tmp"):
        token = (candidate or "").strip()
        if not token:
            continue
        try:
            roots.add(Path(token).expanduser().resolve())
        except OSError:
            continue
    return roots


def classify_web_publication_backend_kind() -> str:
    if not explicit_builder2_final_storage_configured():
        return "unconfigured"
    root = resolve_builder2_final_video_storage_root()
    try:
        resolved = root.resolve()
    except OSError:
        return "unconfigured"
    for temp_root in _system_temp_roots():
        try:
            if resolved == temp_root or temp_root in resolved.parents:
                return "ephemeral_tmp"
        except OSError:
            continue
    return "persistent_disk"


def _storage_root_is_persistent() -> bool:
    return classify_web_publication_backend_kind() == "persistent_disk"


def _probe_storage_writable(root: Path) -> bool:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".builder2_write_probe"
        probe.write_bytes(b"1")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _resolve_public_base_url() -> str:
    return (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip().rstrip("/")


def assess_builder2_final_video_web_storage_capability() -> WebStorageCapability:
    backend = classify_web_publication_backend_kind()
    configured = explicit_builder2_final_storage_configured()
    root = resolve_builder2_final_video_storage_root()
    directory_exists = False
    writable = False
    failure = ""

    if not configured:
        failure = "builder2_web_storage_not_configured"
    elif backend != "persistent_disk":
        failure = "builder2_web_storage_not_persistent"
    else:
        try:
            root.mkdir(parents=True, exist_ok=True)
            directory_exists = root.is_dir()
        except OSError:
            directory_exists = False
        if not directory_exists:
            failure = "builder2_web_storage_not_writable"
        else:
            writable = _probe_storage_writable(root)
            if not writable:
                failure = "builder2_web_storage_not_writable"

    durable = configured and backend == "persistent_disk" and directory_exists and writable and not failure
    ok = durable
    if durable and not _resolve_public_base_url():
        ok = False
        durable = False
        failure = failure or "builder2_web_storage_not_configured"

    logger.info(
        "BUILDER2_FINAL_VIDEO_STORAGE_CAPABILITY storageConfigured=%s storageDirectoryExists=%s "
        "storageWritable=%s backendKind=%s durableStorageConfirmed=%s requestAccepted=%s",
        configured,
        directory_exists,
        writable,
        backend,
        durable,
        ok,
    )
    return WebStorageCapability(
        ok=ok,
        durable_storage_confirmed=durable,
        publication_backend_kind=backend if configured else "unconfigured",
        storage_configured=configured,
        storage_directory_exists=directory_exists,
        storage_writable=writable,
        failure_code=failure,
    )


def persist_builder2_final_video_artifact(
    token: str,
    data: bytes,
    *,
    source_fingerprint: str = "",
) -> WebArtifactStoreResult:
    t = (token or "").strip()
    if not _TOKEN_RE.match(t):
        return WebArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind="unconfigured",
            storage_configured=False,
            storage_writable=False,
            uploaded_byte_count=len(data),
            stored_byte_count=0,
            artifact_fingerprint_verified=False,
            final_public_url="",
            output_token=t,
            failure_code="builder2_web_storage_write_failed",
        )

    capability = assess_builder2_final_video_web_storage_capability()
    if not capability.ok:
        return WebArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=capability.publication_backend_kind,
            storage_configured=capability.storage_configured,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=len(data),
            stored_byte_count=0,
            artifact_fingerprint_verified=False,
            final_public_url="",
            output_token=t,
            failure_code=capability.failure_code or "builder2_web_storage_not_persistent",
        )

    root = resolve_builder2_final_video_storage_root()
    dest = root / f"{t}.mp4"
    uploaded = len(data)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        dest.write_bytes(data)
    except OSError:
        logger.info(
            "BUILDER2_FINAL_VIDEO_ARTIFACT_STORED requestAccepted=false storedByteCount=0 backendKind=%s",
            capability.publication_backend_kind,
        )
        return WebArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=capability.publication_backend_kind,
            storage_configured=True,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=0,
            artifact_fingerprint_verified=False,
            final_public_url="",
            output_token=t,
            failure_code="builder2_web_storage_write_failed",
        )

    stored_path = get_builder2_final_video_path(t)
    stored_count = int(stored_path.stat().st_size) if stored_path and stored_path.is_file() else 0
    fingerprint_ok = stored_count == uploaded and uploaded > 0
    if source_fingerprint:
        try:
            digest = hashlib.sha256(stored_path.read_bytes()).hexdigest() if stored_path else ""
            fingerprint_ok = fingerprint_ok and digest == source_fingerprint
        except OSError:
            fingerprint_ok = False

    if stored_count != uploaded:
        return WebArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=capability.publication_backend_kind,
            storage_configured=True,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=stored_count,
            artifact_fingerprint_verified=False,
            final_public_url="",
            output_token=t,
            failure_code="builder2_web_storage_verification_failed",
        )

    base = _resolve_public_base_url()
    final_url = f"{base}/api/builder2-final-video/{t}" if base else ""
    if not final_url:
        return WebArtifactStoreResult(
            ok=False,
            durable_storage_confirmed=False,
            publication_backend_kind=capability.publication_backend_kind,
            storage_configured=True,
            storage_writable=capability.storage_writable,
            uploaded_byte_count=uploaded,
            stored_byte_count=stored_count,
            artifact_fingerprint_verified=fingerprint_ok,
            final_public_url="",
            output_token=t,
            failure_code="builder2_web_storage_not_configured",
        )

    logger.info(
        "BUILDER2_FINAL_VIDEO_ARTIFACT_STORED requestAccepted=true storedByteCount=%s backendKind=%s "
        "durableStorageConfirmed=true",
        stored_count,
        capability.publication_backend_kind,
    )
    return WebArtifactStoreResult(
        ok=True,
        durable_storage_confirmed=True,
        publication_backend_kind=capability.publication_backend_kind,
        storage_configured=True,
        storage_writable=capability.storage_writable,
        uploaded_byte_count=uploaded,
        stored_byte_count=stored_count,
        artifact_fingerprint_verified=fingerprint_ok,
        final_public_url=final_url,
        output_token=t,
    )


def log_builder2_final_video_served(*, stored_byte_count: int, request_accepted: bool) -> None:
    logger.info(
        "BUILDER2_FINAL_VIDEO_ARTIFACT_SERVED requestAccepted=%s storedByteCount=%s durableStorageConfirmed=%s",
        request_accepted,
        stored_byte_count,
        request_accepted,
    )
