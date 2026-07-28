"""
Builder2 worker/web durable publication split tests — mocks only.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from engine.builder2_final_video_publication import (
    Builder2FinalPublicationError,
    probe_builder2_final_video_web_storage_capability,
    publish_builder2_final_video,
)
from engine.builder2_final_video_store import classify_publication_backend_kind, is_durable_publication_backend
from engine.builder2_final_video_verification import FinalVideoArtifactVerification
from engine.builder2_final_video_web_storage import (
    assess_builder2_final_video_web_storage_capability,
    persist_builder2_final_video_artifact,
)
from engine.builder2_media_finalization_resume import run_finalization_preflight
from tests.test_builder2_media_finalization_failure_inspect import (
    HEADLINE_URL,
    JOB_ID,
    _false_completion_state,
    _job_raw,
)
from tests.test_builder2_media_finalization_eligibility import _production_stranded_state

WORKER_ENV: dict[str, str] = {"ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret"}
WEB_ENV: dict[str, str] = {
    "ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret",
    "BUILDER2_FINAL_VIDEO_STORAGE_DIR": "/var/data/builder2_final",
    "ACE_PUBLIC_BASE_URL": "https://ace.example.com",
}


def _capability_response(*, ok: bool = True) -> MagicMock:
    response = MagicMock(ok=ok, status_code=200 if ok else 503)
    response.json.return_value = {
        "ok": ok,
        "durableStorageConfirmed": ok,
        "publicationBackendKind": "persistent_disk" if ok else "ephemeral_tmp",
        "storageConfigured": ok,
        "storageDirectoryExists": ok,
        "storageWritable": ok,
        "finalVideoUploadRouteAvailable": True,
        "finalVideoServeRouteAvailable": True,
    }
    return response


def _upload_response(*, byte_count: int = 128, token: str = "abcd1234567890123456789012345678") -> MagicMock:
    response = MagicMock(ok=True, status_code=200)
    response.json.return_value = {
        "ok": True,
        "durableStorageConfirmed": True,
        "publicationBackendKind": "persistent_disk",
        "storageConfigured": True,
        "storageWritable": True,
        "uploadedByteCount": byte_count,
        "storedByteCount": byte_count,
        "artifactFingerprintVerified": True,
        "finalPublicUrl": f"https://ace.example.com/api/builder2-final-video/{token}",
        "outputToken": token,
    }
    return response


class TestWorkerDoesNotClassifyLocalStorage(unittest.TestCase):
    @patch.dict(os.environ, WORKER_ENV, clear=True)
    def test_worker_env_classifies_non_durable_locally(self) -> None:
        self.assertFalse(is_durable_publication_backend())
        self.assertEqual(classify_publication_backend_kind(), "ephemeral_tmp")

    @patch("engine.builder2_final_video_publication.verify_published_final_video_artifact")
    @patch("engine.builder2_final_video_publication.requests.post")
    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=lambda k, d=None: WORKER_ENV.get(k, d))
    def test_worker_without_storage_env_still_uploads_when_web_confirms(
        self,
        _env: Any,
        post: Any,
        verify: Any,
    ) -> None:
        post.return_value = _upload_response()
        verify.return_value = FinalVideoArtifactVerification(
            final_url_accessible=True,
            final_url_http_status_code=200,
            final_url_content_type="video/mp4",
            final_url_content_length=128,
            final_artifact_looks_like_video=True,
            durable_storage_confirmed=True,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=True,
            artifact_fingerprint_verified=True,
        )
        with tempfile.TemporaryDirectory() as td:
            local_final = Path(td) / "builder2_final.mp4"
            local_final.write_bytes(b"x" * 128)
            result = publish_builder2_final_video(local_final, "https://ace.example.com")
        self.assertTrue(result.publication_accepted)
        post.assert_called_once()
        self.assertNotIn("/var/data", post.call_args.args[0])

    @patch("engine.builder2_final_video_publication.requests.post")
    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=lambda k, d=None: WORKER_ENV.get(k, d))
    def test_worker_rejects_missing_web_durability_metadata(self, _env: Any, post: Any) -> None:
        response = MagicMock(ok=True, status_code=200)
        response.json.return_value = {"ok": True, "durableStorageConfirmed": False}
        post.return_value = response
        with tempfile.TemporaryDirectory() as td:
            local_final = Path(td) / "builder2_final.mp4"
            local_final.write_bytes(b"x" * 128)
            with self.assertRaises(Builder2FinalPublicationError) as ctx:
                publish_builder2_final_video(local_final, "https://ace.example.com")
        self.assertEqual(ctx.exception.args[0], "final_publication_not_durable")


class TestWebStorageCapability(unittest.TestCase):
    @patch("engine.builder2_final_video_web_storage._system_temp_roots", return_value=set())
    @patch.dict(os.environ, WEB_ENV, clear=True)
    def test_web_classifies_persistent_disk(self, _no_temp: Any) -> None:
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {**WEB_ENV, "BUILDER2_FINAL_VIDEO_STORAGE_DIR": td}, clear=False):
                capability = assess_builder2_final_video_web_storage_capability()
        self.assertTrue(capability.durable_storage_confirmed)
        self.assertEqual(capability.publication_backend_kind, "persistent_disk")

    @patch.dict({"ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret", "ACE_PUBLIC_BASE_URL": "https://ace.example.com"}, clear=True)
    def test_web_rejects_tmp_without_explicit_config(self) -> None:
        capability = assess_builder2_final_video_web_storage_capability()
        self.assertFalse(capability.ok)
        self.assertEqual(capability.failure_code, "builder2_web_storage_not_configured")

    @patch.dict(os.environ, WEB_ENV, clear=True)
    def test_web_rejects_tmp_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td) / "tmp"
            tmp_root.mkdir()
            with patch.dict(
                os.environ,
                {
                    **WEB_ENV,
                    "BUILDER2_FINAL_VIDEO_STORAGE_DIR": str(tmp_root / "ace_builder2_final_video_store"),
                },
                clear=False,
            ):
                with patch("engine.builder2_final_video_web_storage.tempfile.gettempdir", return_value=str(tmp_root)):
                    capability = assess_builder2_final_video_web_storage_capability()
        self.assertFalse(capability.ok)
        self.assertEqual(capability.failure_code, "builder2_web_storage_not_persistent")

    @patch("engine.builder2_final_video_web_storage._system_temp_roots", return_value=set())
    @patch.dict(os.environ, WEB_ENV, clear=True)
    def test_web_upload_returns_durable_metadata(self, _no_temp: Any) -> None:
        token = "abcd1234567890123456789012345678"
        payload = b"x" * 256
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {**WEB_ENV, "BUILDER2_FINAL_VIDEO_STORAGE_DIR": td}, clear=False):
                result = persist_builder2_final_video_artifact(token, payload)
        self.assertTrue(result.ok)
        self.assertTrue(result.durable_storage_confirmed)
        self.assertEqual(result.stored_byte_count, 256)
        self.assertIn("/api/builder2-final-video/", result.final_public_url)

    @patch("engine.builder2_final_video_web_storage._system_temp_roots", return_value=set())
    @patch.dict(os.environ, WEB_ENV, clear=True)
    def test_web_upload_creates_subdirectory(self, _no_temp: Any) -> None:
        token = "abcd1234567890123456789012345678"
        payload = b"x" * 256
        with tempfile.TemporaryDirectory() as td:
            nested = Path(td) / "builder2_final"
            self.assertFalse(nested.exists())
            with patch.dict(os.environ, {**WEB_ENV, "BUILDER2_FINAL_VIDEO_STORAGE_DIR": str(nested)}, clear=False):
                result = persist_builder2_final_video_artifact(token, payload)
            self.assertTrue(result.ok)
            stored = nested / f"{token}.mp4"
            self.assertTrue(stored.is_file())
            self.assertEqual(stored.read_bytes(), payload)


class TestProductionShapedRegression(unittest.TestCase):
    """Worker has no disk env; Web has persistent disk; old worker classification is ignored."""

    @patch("engine.builder2_final_video_publication.verify_published_final_video_artifact")
    @patch("engine.builder2_final_video_publication.requests.post")
    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=lambda k, d=None: WORKER_ENV.get(k, d))
    def test_worker_publishes_via_web_metadata_only(
        self,
        _env: Any,
        post: Any,
        verify: Any,
    ) -> None:
        self.assertFalse(is_durable_publication_backend())
        post.return_value = _upload_response(byte_count=256)
        verify.return_value = FinalVideoArtifactVerification(
            final_url_accessible=True,
            final_url_http_status_code=200,
            final_url_content_type="video/mp4",
            final_url_content_length=256,
            final_artifact_looks_like_video=True,
            durable_storage_confirmed=True,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=True,
            artifact_fingerprint_verified=True,
        )
        with tempfile.TemporaryDirectory() as td:
            local_final = Path(td) / "builder2_final.mp4"
            local_final.write_bytes(b"y" * 256)
            result = publish_builder2_final_video(local_final, "https://ace.example.com")
        self.assertTrue(result.publication_accepted)
        self.assertEqual(result.publication_backend_kind, "persistent_disk")
        post.assert_called_once()

    @patch("engine.builder2_invalid_final_publication_repair.verify_published_final_video_artifact")
    @patch("engine.builder2_invalid_final_publication_repair.redis_configured", return_value=True)
    @patch("engine.builder2_invalid_final_publication_repair.save_tournament_state")
    @patch("engine.builder2_invalid_final_publication_repair.video_job_get_raw")
    @patch("engine.builder2_invalid_final_publication_repair._read_raw")
    def test_recoverable_job_does_not_need_another_repair(
        self,
        read_raw: Any,
        job_get_raw: Any,
        save_state: Any,
        _redis: Any,
        verify: Any,
    ) -> None:
        from engine.builder2_invalid_final_publication_repair import repair_invalid_final_publication_state
        from tests.test_builder2_media_finalization_eligibility import _production_stranded_state

        state = _production_stranded_state()
        media = state["mediaResume"]
        media.pop("finalPublicUrl", None)
        media.pop("finalVideoWithClosureUrl", None)
        media.pop("finalVideoPath", None)
        media["finalizationFailureCode"] = "final_publication_not_durable"
        read_raw.return_value = state
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = repair_invalid_final_publication_state(JOB_ID)
        self.assertTrue(report["repairCompleted"])
        self.assertTrue(report["repairIdempotent"])
        save_state.assert_not_called()


class TestPreflightCapabilityGate(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.probe_builder2_final_video_web_storage_capability")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    def test_failed_capability_skips_ffmpeg(
        self,
        _redis: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        probe: Any,
        pipeline: Any,
    ) -> None:
        from engine.builder2_final_video_publication import WebStorageCapabilityProbeResult

        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        probe.return_value = WebStorageCapabilityProbeResult(
            accepted=False,
            durable_storage_confirmed=False,
            publication_backend_kind="ephemeral_tmp",
            storage_configured=False,
            storage_directory_exists=False,
            storage_writable=False,
            failure_code="builder2_web_storage_not_persistent",
            http_status=503,
        )
        report = run_finalization_preflight(
            job_id=JOB_ID,
            state=_production_stranded_state(),
            job_video_url=HEADLINE_URL,
        )
        pipeline.assert_not_called()
        self.assertEqual(report["storageCapabilityCalls"], 1)
        self.assertFalse(report["storageCapabilityAccepted"])
        self.assertEqual(report["failureStage"], "publication_capability")
        self.assertEqual(report["publicationCalls"], 0)
        self.assertEqual(report["totalFfmpegCalls"], 0)

    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.probe_builder2_final_video_web_storage_capability")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    def test_successful_capability_allows_pipeline(
        self,
        _redis: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        probe: Any,
        pipeline: Any,
    ) -> None:
        from engine.builder2_final_video_publication import WebStorageCapabilityProbeResult

        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        probe.return_value = WebStorageCapabilityProbeResult(
            accepted=True,
            durable_storage_confirmed=True,
            publication_backend_kind="persistent_disk",
            storage_configured=True,
            storage_directory_exists=True,
            storage_writable=True,
        )

        def _ok(**kwargs: Any) -> None:
            kwargs["report"]["ok"] = True
            kwargs["report"]["readyForFinalizationRecovery"] = True

        pipeline.side_effect = _ok
        report = run_finalization_preflight(
            job_id=JOB_ID,
            state=_production_stranded_state(),
            job_video_url=HEADLINE_URL,
        )
        probe.assert_called_once()
        pipeline.assert_called_once()
        self.assertTrue(report["storageCapabilityAccepted"])
        self.assertTrue(report["webDurableStorageConfirmed"])


class TestCapabilityProbeClient(unittest.TestCase):
    @patch("engine.builder2_final_video_publication.requests.get")
    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=lambda k, d=None: WORKER_ENV.get(k, d))
    def test_probe_accepts_persistent_disk_response(self, _env: Any, get: Any) -> None:
        get.return_value = _capability_response()
        result = probe_builder2_final_video_web_storage_capability("https://ace.example.com")
        self.assertTrue(result.accepted)
        self.assertEqual(result.publication_backend_kind, "persistent_disk")
        get.assert_called_once()
        self.assertIn("/api/builder2-final-video-storage-capability", get.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
