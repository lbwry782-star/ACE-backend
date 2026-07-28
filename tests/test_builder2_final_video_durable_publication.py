"""
Builder2 durable final-video publication contract tests — mocks only.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from engine.builder2_final_video_publication import (
    Builder2FinalPublicationError,
    FinalVideoPublicationResult,
    publish_builder2_final_video,
)
from engine.builder2_final_video_store import classify_publication_backend_kind, is_durable_publication_backend
from engine.builder2_final_video_verification import (
    FinalVideoArtifactVerification,
    verify_published_final_video_artifact,
)
from engine.builder2_invalid_final_publication_repair import repair_invalid_final_publication_state
from engine.builder2_media_finalization_contract import (
    assess_false_completion,
    closure_inclusive_artifact_valid,
    validate_builder2_media_completion_contract,
)
from engine.builder2_final_video_artifact_inspect import inspect_builder2_final_video_artifact
from tests.test_builder2_media_finalization_failure_inspect import (
    BROKEN_HEADLINE_FINAL_URL,
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    RAW_RUNWAY,
    _false_completion_state,
    _job_raw,
    verified_final_publication_media_fields,
)


DURABLE_ENV = {
    "ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret",
    "VIDEO_HEADLINE_STORAGE_DIR": "/var/data/ace-videos",
}


def _env_get(key: str, default: Any = None) -> Any:
    return DURABLE_ENV.get(key, default)


def _verified_publication_result(public_url: str = CLOSURE_URL) -> FinalVideoPublicationResult:
    return FinalVideoPublicationResult(
        public_url=public_url,
        output_token="tok" * 8,
        route_family="api/builder2-final-video",
        publication_accepted=True,
        durable_storage_confirmed=True,
        publication_backend_kind="persistent_disk",
        publication_reference_present=True,
        uploaded_byte_count=1028987,
        post_upload_verification_attempted=True,
        post_upload_verification_accepted=True,
        post_upload_http_status_code=200,
        post_upload_content_type="video/mp4",
        post_upload_content_length=1028987,
        artifact_fingerprint_verified=True,
    )


def _completed_false_publication_state() -> dict[str, Any]:
    state = deepcopy(_false_completion_state(with_valid_closure=False))
    media = state["mediaResume"]
    media["finalPublicUrl"] = BROKEN_HEADLINE_FINAL_URL
    media["finalVideoWithClosureUrl"] = BROKEN_HEADLINE_FINAL_URL
    media["finalVideoPath"] = BROKEN_HEADLINE_FINAL_URL
    media["advertisingClosureRendered"] = True
    media["advertisingClosureStatus"] = "completed"
    media["actualFinalVideoDurationSeconds"] = 12.034
    media["headlineReconstructionCompleted"] = True
    media.pop("finalPublicationVerificationAccepted", None)
    media.pop("finalPublicationDurableStorageConfirmed", None)
    return state


class TestStorageClassification(unittest.TestCase):
    @patch.dict(os.environ, DURABLE_ENV, clear=False)
    def test_persistent_disk_when_storage_dir_configured(self) -> None:
        self.assertEqual(classify_publication_backend_kind(), "persistent_disk")
        self.assertTrue(is_durable_publication_backend())

    @patch.dict(os.environ, {}, clear=True)
    def test_tmp_not_durable_without_explicit_config(self) -> None:
        self.assertEqual(classify_publication_backend_kind(), "ephemeral_tmp")
        self.assertFalse(is_durable_publication_backend())


class TestVerification(unittest.TestCase):
    @patch("engine.builder2_final_video_verification.requests.get")
    @patch("engine.builder2_final_video_verification.requests.head")
    def test_accepts_real_mp4_response(self, head: Any, get: Any) -> None:
        head.return_value = MagicMock(status_code=200, headers={"Content-Type": "video/mp4", "Content-Length": "128"})
        result = verify_published_final_video_artifact(
            CLOSURE_URL,
            expected_byte_count=128,
            durable_storage_confirmed=True,
        )
        self.assertTrue(result.post_upload_verification_accepted)
        get.assert_not_called()

    @patch("engine.builder2_final_video_verification.requests.get")
    @patch("engine.builder2_final_video_verification.requests.head")
    def test_rejects_json_not_found(self, head: Any, get: Any) -> None:
        head.return_value = MagicMock(status_code=404, headers={"Content-Type": "application/json"})
        get.return_value = MagicMock(
            status_code=404,
            headers={"Content-Type": "application/json"},
            content=json.dumps({"ok": False, "error": "not_found"}).encode("utf-8"),
        )
        result = verify_published_final_video_artifact(BROKEN_HEADLINE_FINAL_URL)
        self.assertFalse(result.post_upload_verification_accepted)
        self.assertEqual(result.failure_code, "final_publication_artifact_missing")

    @patch("engine.builder2_final_video_verification.requests.get")
    @patch("engine.builder2_final_video_verification.requests.head")
    def test_rejects_content_type_mismatch(self, head: Any, get: Any) -> None:
        head.return_value = MagicMock(status_code=200, headers={"Content-Type": "application/json", "Content-Length": "64"})
        get.return_value = MagicMock(
            status_code=200,
            headers={"Content-Type": "application/json", "Content-Length": "64"},
            content=b'{"ok":false}',
        )
        result = verify_published_final_video_artifact(CLOSURE_URL, durable_storage_confirmed=True)
        self.assertFalse(result.final_artifact_looks_like_video)
        self.assertEqual(result.failure_code, "final_publication_content_type_invalid")

    @patch("engine.builder2_final_video_verification.requests.get")
    @patch("engine.builder2_final_video_verification.requests.head")
    def test_rejects_size_mismatch(self, head: Any, get: Any) -> None:
        head.return_value = MagicMock(status_code=200, headers={"Content-Type": "video/mp4", "Content-Length": "64"})
        result = verify_published_final_video_artifact(
            CLOSURE_URL,
            expected_byte_count=128,
            durable_storage_confirmed=True,
        )
        self.assertEqual(result.failure_code, "final_publication_size_mismatch")


class TestPublication(unittest.TestCase):
    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=_env_get)
    @patch("engine.builder2_final_video_publication.is_durable_publication_backend", return_value=True)
    @patch("engine.builder2_final_video_publication.classify_publication_backend_kind", return_value="persistent_disk")
    @patch("engine.builder2_final_video_publication.verify_published_final_video_artifact")
    @patch("engine.builder2_final_video_publication.requests.post")
    def test_upload_success_without_accessibility_rejected(
        self,
        post: Any,
        verify: Any,
        _backend: Any,
        _durable: Any,
        _env: Any,
    ) -> None:
        post.return_value = MagicMock(ok=True, status_code=200)
        verify.return_value = FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=404,
            final_url_content_type="application/json",
            final_url_content_length=None,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=True,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_artifact_missing",
        )
        with tempfile.TemporaryDirectory() as td:
            local_final = Path(td) / "builder2_final.mp4"
            local_final.write_bytes(b"x" * 128)
            with self.assertRaises(Builder2FinalPublicationError) as ctx:
                publish_builder2_final_video(local_final, "https://ace.example.com")
        self.assertEqual(ctx.exception.args[0], "final_publication_artifact_missing")

    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=_env_get)
    @patch("engine.builder2_final_video_publication.is_durable_publication_backend", return_value=True)
    @patch("engine.builder2_final_video_publication.classify_publication_backend_kind", return_value="persistent_disk")
    @patch("engine.builder2_final_video_publication.verify_published_final_video_artifact")
    @patch("engine.builder2_final_video_publication.requests.post")
    def test_successful_publication_uses_builder2_route(
        self,
        post: Any,
        verify: Any,
        _backend: Any,
        _durable: Any,
        _env: Any,
    ) -> None:
        post.return_value = MagicMock(ok=True, status_code=200)
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
            result = publish_builder2_final_video(
                local_final,
                "https://ace.example.com",
                output_token="abcd1234567890123456789012345678",
            )
        self.assertIn("/api/builder2-final-video/", result.public_url)
        self.assertTrue(result.publication_accepted)
        self.assertTrue(result.post_upload_verification_accepted)
        post.assert_called_once()
        self.assertIn("/api/builder2-final-video-artifact", post.call_args.args[0])

    @patch.dict({"ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret"}, clear=True)
    def test_rejects_ephemeral_backend(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            local_final = Path(td) / "builder2_final.mp4"
            local_final.write_bytes(b"x" * 128)
            with self.assertRaises(Builder2FinalPublicationError) as ctx:
                publish_builder2_final_video(local_final, "https://ace.example.com")
        self.assertEqual(ctx.exception.args[0], "final_publication_not_durable")


class TestCompletionContract(unittest.TestCase):
    def test_headline_route_404_not_published(self) -> None:
        state = _completed_false_publication_state()
        plan = state["winnerDevelopmentPlan"]
        ok, failure, failures = validate_builder2_media_completion_contract(
            state=state,
            plan=plan,
            job_video_url=HEADLINE_URL,
            require_job_video_url_match=False,
        )
        self.assertFalse(ok)
        self.assertIn("final_publication_route_not_durable", failures)

    def test_verified_builder2_final_not_false_completion(self) -> None:
        state = deepcopy(_false_completion_state(with_valid_closure=True))
        plan = state["winnerDevelopmentPlan"]
        false_completion, reasons = assess_false_completion(state=state, plan=plan, job_video_url=CLOSURE_URL)
        self.assertFalse(false_completion)
        self.assertEqual(reasons, [])

    def test_embedded_headline_final_does_not_require_separate_headline_artifact(self) -> None:
        state = deepcopy(_false_completion_state(with_valid_closure=True))
        state["mediaResume"].pop("headlineArtifactUrl", None)
        media = state["mediaResume"]
        ok, _, _ = validate_builder2_media_completion_contract(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=CLOSURE_URL,
            require_job_video_url_match=False,
        )
        self.assertTrue(ok)
        self.assertTrue(
            closure_inclusive_artifact_valid(
                state=state,
                closure_url=media["finalVideoWithClosureUrl"],
                raw_url=RAW_RUNWAY,
                headline_url="",
                job_video_url=CLOSURE_URL,
            )
        )


class TestArtifactInspectAndRepair(unittest.TestCase):
    @patch("engine.builder2_final_video_artifact_inspect.verify_published_final_video_artifact")
    @patch("engine.builder2_final_video_artifact_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_final_video_artifact_inspect.video_job_get_raw")
    @patch("engine.builder2_final_video_artifact_inspect._read_raw")
    def test_inspect_detects_completion_contradiction(
        self,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
        verify: Any,
    ) -> None:
        read_raw.return_value = _completed_false_publication_state()
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        verify.return_value = FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=404,
            final_url_content_type="application/json",
            final_url_content_length=None,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=False,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_artifact_missing",
        )
        report = inspect_builder2_final_video_artifact(JOB_ID)
        self.assertEqual(report["finalUrlHttpStatusCode"], 404)
        self.assertFalse(report["finalUrlAccessible"])
        self.assertTrue(report["persistedCompletionContradictsArtifact"])
        self.assertTrue(report["rawRunwayRecoverable"])
        self.assertEqual(report["recommendedNextAction"], "repair_invalid_final_publication_state")

    @patch("engine.builder2_invalid_final_publication_repair.verify_published_final_video_artifact")
    @patch("engine.builder2_invalid_final_publication_repair.redis_configured", return_value=True)
    @patch("engine.builder2_invalid_final_publication_repair.save_tournament_state")
    @patch("engine.builder2_invalid_final_publication_repair.video_job_get_raw")
    @patch("engine.builder2_invalid_final_publication_repair._read_raw")
    def test_repair_returns_recoverable_state(
        self,
        read_raw: Any,
        job_get_raw: Any,
        save_state: Any,
        _redis: Any,
        verify: Any,
    ) -> None:
        state = _completed_false_publication_state()
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        verify.return_value = FinalVideoArtifactVerification(
            final_url_accessible=False,
            final_url_http_status_code=404,
            final_url_content_type="application/json",
            final_url_content_length=None,
            final_artifact_looks_like_video=False,
            durable_storage_confirmed=False,
            post_upload_verification_attempted=True,
            post_upload_verification_accepted=False,
            artifact_fingerprint_verified=False,
            failure_code="final_publication_artifact_missing",
        )
        report = repair_invalid_final_publication_state(JOB_ID)
        self.assertTrue(report["repairCompleted"])
        saved = save_state.call_args[0][1]
        self.assertEqual(saved["status"], "media_finalization_incomplete")
        self.assertTrue(saved["mediaContinuationRequired"])
        self.assertEqual(saved["mediaResume"]["mediaResumeStatus"], "finalization_failed")
        self.assertEqual(saved["mediaResume"]["finalizationFailureCode"], "final_publication_artifact_missing")
        self.assertEqual(saved["mediaResume"]["brokenFinalPublicationUrl"], BROKEN_HEADLINE_FINAL_URL)
        self.assertNotIn("finalPublicUrl", saved["mediaResume"])
        self.assertEqual(saved["mediaResume"]["rawRunwayVideoUrl"], RAW_RUNWAY)

    @patch("engine.builder2_invalid_final_publication_repair.verify_published_final_video_artifact")
    @patch("engine.builder2_invalid_final_publication_repair.redis_configured", return_value=True)
    @patch("engine.builder2_invalid_final_publication_repair.save_tournament_state")
    @patch("engine.builder2_invalid_final_publication_repair.video_job_get_raw")
    @patch("engine.builder2_invalid_final_publication_repair._read_raw")
    def test_repair_idempotent(
        self,
        read_raw: Any,
        job_get_raw: Any,
        save_state: Any,
        _redis: Any,
        verify: Any,
    ) -> None:
        state = _completed_false_publication_state()
        state["status"] = "media_finalization_incomplete"
        state["mediaContinuationRequired"] = True
        media = state["mediaResume"]
        media["mediaResumeStatus"] = "finalization_failed"
        media["advertisingClosureStatus"] = "failed"
        media["advertisingClosureRendered"] = False
        media.pop("finalPublicUrl", None)
        media.pop("finalVideoWithClosureUrl", None)
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = repair_invalid_final_publication_state(JOB_ID)
        self.assertTrue(report["repairCompleted"])
        self.assertTrue(report["repairIdempotent"])
        save_state.assert_not_called()


class TestAppRoutes(unittest.TestCase):
    def test_builder2_final_video_store_roundtrip(self) -> None:
        from engine.builder2_final_video_store import get_builder2_final_video_path, write_builder2_final_video_bytes

        token = "abcd1234567890123456789012345678"
        payload = b"\x00\x00\x00\x18ftypmp42" + b"x" * 64
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {"BUILDER2_FINAL_VIDEO_STORAGE_DIR": td, "VIDEO_HEADLINE_STORAGE_DIR": td},
            clear=False,
        ):
            self.assertTrue(write_builder2_final_video_bytes(token, payload))
            path = get_builder2_final_video_path(token)
            self.assertIsNotNone(path)
            assert path is not None
            self.assertEqual(path.read_bytes(), payload)

    def test_legacy_headline_route_still_registered(self) -> None:
        from app import app

        rules = {str(rule.rule) for rule in app.url_map.iter_rules()}
        self.assertTrue(any("/api/video-headline/<token>" in rule for rule in rules))


if __name__ == "__main__":
    unittest.main()
