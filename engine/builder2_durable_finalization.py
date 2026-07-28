"""
Shared Builder2 durable final-video finalization — fresh production and recovery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from engine.builder2_final_video_publication import (
    Builder2FinalPublicationError,
    FinalVideoPublicationResult,
    WebStorageCapabilityProbeResult,
    probe_builder2_final_video_web_storage_capability,
    publish_builder2_final_video,
)


class Builder2DurableFinalizationError(Builder2FinalPublicationError):
    pass


def require_builder2_web_storage_capability(public_base_url: str) -> WebStorageCapabilityProbeResult:
    capability = probe_builder2_final_video_web_storage_capability(public_base_url)
    if not capability.accepted:
        raise Builder2DurableFinalizationError(
            capability.failure_code or "builder2_web_storage_not_persistent",
            stage="publication_capability",
            server_failure_code=capability.failure_code or "",
        )
    return capability


def publish_builder2_durable_final_video(
    local_final_path: Path,
    public_base_url: str,
    *,
    job_id: str = "",
    output_token: str | None = None,
) -> FinalVideoPublicationResult:
    return publish_builder2_final_video(
        local_final_path,
        public_base_url,
        job_id=job_id,
        output_token=output_token,
    )


def apply_builder2_durable_publication_fields(
    state: Dict[str, Any],
    publication: FinalVideoPublicationResult,
) -> str:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    public_url = publication.public_url
    media["finalPublicUrl"] = public_url
    media["finalVideoWithClosureUrl"] = public_url
    media["finalVideoPath"] = public_url
    media["finalPublicationVerificationAccepted"] = publication.post_upload_verification_accepted
    media["finalPublicationDurableStorageConfirmed"] = publication.durable_storage_confirmed
    media["finalPublicationBackendKind"] = publication.publication_backend_kind
    media["finalPublicationReferencePresent"] = publication.publication_reference_present
    media["finalPublicationUploadedByteCount"] = publication.uploaded_byte_count
    media["artifactFingerprintVerified"] = publication.artifact_fingerprint_verified
    media["finalArtifactPublishedAt"] = media.get("finalArtifactPublishedAt")
    media["deliveryArtifactPaths"] = [public_url]
    media["advertisingClosureRendered"] = True
    media["advertisingClosureStatus"] = "completed"
    state["advertisingClosureStatus"] = "completed"
    return public_url


def mark_builder2_finalization_infrastructure_failure(
    state: Dict[str, Any],
    *,
    failure_stage: str,
    failure_code: str,
    failure_class: str = "Builder2FinalPublicationError",
) -> None:
    media = state.setdefault("mediaResume", {})
    if not isinstance(media, dict):
        media = {}
        state["mediaResume"] = media
    state["status"] = "media_finalization_incomplete"
    state["mediaContinuationRequired"] = True
    state["advertisingClosureStatus"] = "failed"
    media["mediaResumeStatus"] = "finalization_failed"
    media["finalizationFailureStage"] = failure_stage[:64]
    media["finalizationFailureCode"] = failure_code[:128]
    media["finalizationFailureClass"] = failure_class[:64]
    media["advertisingClosureStatus"] = "failed"
    media["advertisingClosureRendered"] = False
    media.pop("finalPublicUrl", None)
    media.pop("finalVideoWithClosureUrl", None)
    media.pop("finalVideoPath", None)
