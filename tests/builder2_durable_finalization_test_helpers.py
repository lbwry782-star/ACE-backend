"""Shared mocks for Builder2 durable fresh-production finalization tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from engine.builder2_closure_render import ClosureRenderResult
from engine.builder2_final_video_publication import FinalVideoPublicationResult, WebStorageCapabilityProbeResult
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL


def accepted_web_storage_capability_result() -> WebStorageCapabilityProbeResult:
    return WebStorageCapabilityProbeResult(
        accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind="persistent_disk",
        storage_configured=True,
        storage_directory_exists=True,
        storage_writable=True,
    )


def durable_publication_result(public_url: str = CLOSURE_URL) -> FinalVideoPublicationResult:
    return FinalVideoPublicationResult(
        public_url=public_url,
        output_token="tok" * 8,
        route_family="api/builder2-final-video",
        publication_accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind="persistent_disk",
        publication_reference_present=True,
        uploaded_byte_count=128,
        stored_byte_count=128,
        post_upload_verification_attempted=True,
        post_upload_verification_accepted=True,
        post_upload_http_status_code=200,
        post_upload_content_type="video/mp4",
        post_upload_content_length=128,
        artifact_fingerprint_verified=True,
        web_storage_configured=True,
        web_storage_writable=True,
    )


def mock_closure_render_result(source_video_url: str = "", **kwargs: Any) -> ClosureRenderResult:
    _ = source_video_url
    output_path = kwargs.get("output_path")
    if output_path is not None:
        Path(output_path).write_bytes(b"x" * 128)
    return ClosureRenderResult(
        public_url="",
        local_path=str(output_path or "/tmp/builder2_final.mp4"),
        measured_duration_seconds=12.034,
        output_token="abcd1234567890123456789012345678",
        input_fingerprint="abc",
        closure_ffprobe_calls=1,
    )


def patch_media_pipeline_durable_finalization(public_url: str = CLOSURE_URL):
    publish_mock = MagicMock(return_value=durable_publication_result(public_url))
    return (
        patch(
            "engine.builder2_durable_finalization.require_builder2_web_storage_capability",
            return_value=accepted_web_storage_capability_result(),
        ),
        patch(
            "engine.builder2_closure_render.render_builder2_advertising_closure_endcard",
            side_effect=mock_closure_render_result,
        ),
        patch(
            "engine.builder2_durable_finalization.publish_builder2_durable_final_video",
            publish_mock,
        ),
        publish_mock,
    )
