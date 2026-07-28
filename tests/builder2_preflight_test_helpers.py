"""Shared mocks for Builder2 finalization preflight tests."""
from __future__ import annotations

from unittest.mock import patch

from engine.builder2_final_video_publication import WebStorageCapabilityProbeResult


def accepted_web_storage_capability_result() -> WebStorageCapabilityProbeResult:
    return WebStorageCapabilityProbeResult(
        accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind="persistent_disk",
        storage_configured=True,
        storage_directory_exists=True,
        storage_writable=True,
    )


def patch_accepted_web_storage_capability():
    return patch(
        "engine.builder2_media_finalization_resume.probe_builder2_final_video_web_storage_capability",
        return_value=accepted_web_storage_capability_result(),
    )
