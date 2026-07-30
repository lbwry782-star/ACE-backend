"""
Builder2 final-video staging and publication tests — mocks only.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from engine.builder2_closure_render import Builder2ClosureRenderError, ClosureRenderResult
from engine.builder2_final_local_staging import (
    Builder2FinalLocalStagingError,
    handoff_local_final_artifact,
    is_legacy_headline_store_path,
    prepare_publication_staging,
)
from engine.builder2_final_video_publication import (
    Builder2FinalPublicationError,
    FinalVideoPublicationResult,
    publish_builder2_final_video,
    resolve_durable_final_video_publisher_kind,
)
from tests.test_builder2_final_video_durable_publication import _env_get
from engine.builder2_media_finalization_resume import (
    FinalizationPipelineOutcome,
    _execute_finalization_render_pipeline,
    _initial_report,
    run_finalization_preflight,
    run_one_media_finalization_resume,
)
from engine.video_headline_postprocess import _storage_root
from tests.test_builder2_media_finalization import _valid_closure_result
from tests.test_builder2_media_finalization_failure_inspect import (
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    _false_completion_state,
    _job_raw,
)
from tests.test_builder2_media_finalization_eligibility import _production_stranded_state


def _render_result(*, measured: float = 12.034, local_path: str = "/tmp/builder2_final.mp4") -> ClosureRenderResult:
    return ClosureRenderResult(
        public_url="",
        local_path=local_path,
        measured_duration_seconds=measured,
        output_token="tok" * 8,
        input_fingerprint="abc",
        closure_ffprobe_calls=2,
    )


def _verified_publication_result() -> FinalVideoPublicationResult:
    return FinalVideoPublicationResult(
        public_url=CLOSURE_URL,
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


class TestLocalStaging(unittest.TestCase):
    def test_source_missing_identified(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "nested" / "final.mp4"
            with self.assertRaises(Builder2FinalLocalStagingError) as ctx:
                handoff_local_final_artifact(Path(td) / "missing.mp4", dest)
            self.assertEqual(ctx.exception.args[0], "builder2_final_local_source_missing")

    def test_destination_parent_missing_before_mkdir_simulated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.mp4"
            source.write_bytes(b"video")
            dest = Path(td) / "child" / "final.mp4"
            handoff_local_final_artifact(source, dest)
            self.assertTrue(dest.is_file())

    def test_legacy_headline_store_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.mp4"
            source.write_bytes(b"video")
            root = _storage_root()
            dest = root / "deadbeefdeadbeefdeadbeefdeadbeef.mp4"
            with self.assertRaises(Builder2FinalLocalStagingError) as ctx:
                handoff_local_final_artifact(source, dest)
            self.assertEqual(ctx.exception.args[0], "builder2_final_legacy_headline_store_rejected")

    def test_is_legacy_headline_store_path(self) -> None:
        root = _storage_root()
        self.assertTrue(is_legacy_headline_store_path(root / "abc12345678901234567890123456789012.mp4"))
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(is_legacy_headline_store_path(Path(td) / "final.mp4"))


class TestPublication(unittest.TestCase):
    @patch("engine.builder2_final_video_publication.os.environ.get", side_effect=_env_get)
    @patch("engine.builder2_final_video_publication.verify_published_final_video_artifact")
    @patch("engine.builder2_final_video_publication.requests.post")
    def test_publish_once_after_verified_render(self, post: Any, verify: Any, _env: Any) -> None:
        from engine.builder2_final_video_verification import FinalVideoArtifactVerification
        from tests.test_builder2_final_video_durable_publication import _upload_json_response

        token = "tok12345678901234567890123456789012"
        post.return_value = _upload_json_response(byte_count=128, token=token)
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
                job_id=JOB_ID,
                output_token=token,
            )
        self.assertTrue(result.upload_accepted)
        self.assertEqual(result.publisher_kind, resolve_durable_final_video_publisher_kind())
        self.assertIn("/api/builder2-final-video/", result.public_url)
        post.assert_called_once()

    @patch.dict("os.environ", {"ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret"}, clear=False)
    def test_rejects_legacy_headline_store_local_path(self) -> None:
        root = _storage_root()
        local_final = root / "tok12345678901234567890123456789012.mp4"
        with self.assertRaises(Builder2FinalPublicationError):
            publish_builder2_final_video(local_final, "https://ace.example.com")


class TestRecoveryPipelineRegression(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume._probe_duration", return_value=10.042)
    @patch("engine.builder2_media_finalization_resume.publish_builder2_final_video")
    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    def test_production_shaped_recovery_uses_caller_owned_path_and_publisher(
        self,
        build_config: Any,
        source_decision: Any,
        closure_render: Any,
        publish: Any,
        _probe: Any,
    ) -> None:
        from engine.builder2_final_video_publication import FinalVideoPublicationResult

        state = _production_stranded_state()
        plan = state["winnerDevelopmentPlan"]
        report = _initial_report(job_id=JOB_ID, preflight=False)
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        raw_path = Path(tempfile.gettempdir()) / "raw.mp4"
        raw_path.write_bytes(b"raw")

        def _render(*_args: Any, **kwargs: Any) -> ClosureRenderResult:
            output_path = kwargs["output_path"]
            self.assertEqual(output_path.name, "builder2_final.mp4")
            output_path.write_bytes(b"x" * 1028987)
            return _render_result(measured=13.534, local_path=str(output_path))

        closure_render.side_effect = _render
        publish.return_value = _verified_publication_result()
        source_decision.return_value = MagicMock(
            source_kind="raw_runway",
            closure_input_path=raw_path,
            failure_reason=None,
            local_headline_render_required=False,
            legacy_headline_url=None,
            persisted_headline_url=None,
            raw_runway_diagnostics=MagicMock(download_accepted=True),
        )

        outcome = _execute_finalization_render_pipeline(
            job_id=JOB_ID,
            state=state,
            plan=plan,
            job_video_url=HEADLINE_URL,
            report=report,
            preflight=False,
            public_base_url="https://ace.example.com",
        )
        self.assertIsInstance(outcome, FinalizationPipelineOutcome)
        self.assertEqual(outcome.public_url, CLOSURE_URL)
        self.assertTrue(report["localRenderAccepted"])
        self.assertTrue(report["closureRenderAccepted"])
        self.assertEqual(report["measuredFinalDurationSeconds"], 13.534)
        self.assertTrue(report["finalDurationAccepted"])
        self.assertEqual(report["publicationCalls"], 1)
        publish.assert_called_once()
        closure_render.assert_called_once()
        _, kwargs = closure_render.call_args
        self.assertFalse(kwargs.get("publish", False))

    @patch("engine.builder2_media_finalization_resume._probe_duration", return_value=10.042)
    @patch("engine.builder2_media_finalization_resume.publish_builder2_final_video")
    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    def test_publication_failure_preserves_render_diagnostics(
        self,
        build_config: Any,
        source_decision: Any,
        closure_render: Any,
        publish: Any,
        _probe: Any,
    ) -> None:
        state = _production_stranded_state()
        plan = state["winnerDevelopmentPlan"]
        report = _initial_report(job_id=JOB_ID, preflight=False)
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        raw_path = Path(tempfile.gettempdir()) / "raw2.mp4"
        raw_path.write_bytes(b"raw")

        def _render(*_args: Any, **kwargs: Any) -> ClosureRenderResult:
            kwargs["output_path"].write_bytes(b"x" * 1028987)
            return _render_result(measured=13.534, local_path=str(kwargs["output_path"]))

        closure_render.side_effect = _render
        publish.side_effect = Builder2FinalPublicationError("builder2_final_publication_failed")
        source_decision.return_value = MagicMock(
            source_kind="raw_runway",
            closure_input_path=raw_path,
            failure_reason=None,
            local_headline_render_required=False,
            legacy_headline_url=None,
            persisted_headline_url=None,
            raw_runway_diagnostics=MagicMock(download_accepted=True),
        )

        outcome = _execute_finalization_render_pipeline(
            job_id=JOB_ID,
            state=state,
            plan=plan,
            job_video_url=HEADLINE_URL,
            report=report,
            preflight=False,
            public_base_url="https://ace.example.com",
        )
        self.assertIsNone(outcome)
        self.assertTrue(report["localRenderAccepted"])
        self.assertTrue(report["closureRenderAccepted"])
        self.assertEqual(report["measuredFinalDurationSeconds"], 13.534)
        self.assertTrue(report["finalDurationAccepted"])
        self.assertFalse(report["publicationAccepted"])
        self.assertEqual(report["failureStage"], "publication")
        self.assertEqual(report["publicationCalls"], 0)

    def test_old_headline_store_destination_would_have_failed_without_mkdir(self) -> None:
        root = _storage_root()
        token = "abcd1234567890123456789012345678"
        dest = root / f"{token}.mp4"
        if dest.parent.exists():
            pass
        else:
            self.assertFalse(dest.parent.is_dir())
        source = Path(tempfile.gettempdir()) / "verified_out.mp4"
        source.write_bytes(b"x" * 1028987)
        with self.assertRaises(FileNotFoundError):
            source.replace(dest)
        with self.assertRaises(Builder2FinalLocalStagingError):
            handoff_local_final_artifact(source, dest)


class TestPreflightStaging(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.probe_builder2_final_video_web_storage_capability")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    def test_preflight_exercises_staging_without_publication(
        self,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        probe: Any,
        pipeline: Any,
    ) -> None:
        from engine.builder2_final_video_publication import WebStorageCapabilityProbeResult

        probe.return_value = WebStorageCapabilityProbeResult(
            accepted=True,
            durable_storage_confirmed=True,
            publication_backend_kind="persistent_disk",
            storage_configured=True,
            storage_directory_exists=True,
            storage_writable=True,
        )

        def _ok(**kwargs: Any) -> FinalizationPipelineOutcome:
            kwargs["report"].update(
                {
                    "ok": True,
                    "readyForFinalizationRecovery": True,
                    "localFinalRenderCompleted": True,
                    "publicationStagingPreparationAccepted": True,
                    "publicationCalls": 0,
                }
            )
            return FinalizationPipelineOutcome(
                render_result=_render_result(),
                publication_result=None,
                public_url="",
            )

        pipeline.side_effect = _ok
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {}
        state = _false_completion_state(with_valid_closure=False)
        report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["ok"])
        self.assertEqual(report["publicationCalls"], 0)
        self.assertEqual(report["redisMutations"], 0)


if __name__ == "__main__":
    unittest.main()
