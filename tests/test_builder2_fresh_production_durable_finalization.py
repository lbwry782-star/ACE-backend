"""
Builder2 fresh-production durable finalization regression tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch

from engine.builder2_durable_finalization import publish_builder2_durable_final_video
from engine.builder2_final_video_publication import publish_builder2_final_video
from engine.builder2_media_finalization_contract import validate_builder2_media_completion_contract
from engine.builder2_media_pipeline import execute_builder2_media_pipeline
from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_resume_service import build_builder2_status_payload
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.runway_video import RunwayVideoMVPError, _generate_one_video_mvp_body
from tests.builder2_durable_finalization_test_helpers import patch_media_pipeline_durable_finalization
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL as FINAL_URL
from tests.test_builder2_media_resume import _media_ready_state, _mock_pipeline_deps, _mock_start_image_data_uri

WORKER_ENV = {"ACE_VIDEO_HEADLINE_UPLOAD_SECRET": "secret", "RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}


class TestFreshProductionUsesCanonicalPublisher(unittest.TestCase):
    def test_fresh_publisher_delegates_to_shared_publish_function(self) -> None:
        with patch("engine.builder2_durable_finalization.publish_builder2_final_video") as publish:
            publish.return_value = MagicMock(public_url=FINAL_URL)
            from pathlib import Path

            result = publish_builder2_durable_final_video(Path("/tmp/x.mp4"), "https://ace.example.com")
            publish.assert_called_once()
            self.assertEqual(result.public_url, FINAL_URL)

    @patch("engine.builder2_durable_finalization.require_builder2_web_storage_capability")
    @patch("engine.builder2_closure_render.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_durable_finalization.publish_builder2_durable_final_video")
    def test_fresh_pipeline_calls_capability_before_closure_render(
        self,
        publish: Any,
        closure_render: Any,
        capability: Any,
    ) -> None:
        from tests.builder2_durable_finalization_test_helpers import (
            accepted_web_storage_capability_result,
            durable_publication_result,
            mock_closure_render_result,
        )

        enable_memory_store()
        try:
            capability.return_value = accepted_web_storage_capability_result()
            closure_render.side_effect = mock_closure_render_result
            publish.return_value = durable_publication_result(FINAL_URL)
            state = _media_ready_state(job_id="job-fresh-capability")
            state["mediaResume"] = {
                "startImageArtifact": _mock_start_image_data_uri(),
                "runwayTaskId": "task-1",
                "runwayVideoUrl": "https://runway.example.com/raw.mp4",
                "downloadedVideoPath": "https://runway.example.com/raw.mp4",
                "rawRunwayVideoUrl": "https://runway.example.com/raw.mp4",
            }
            call_order: list[str] = []

            def _capability(*_args: Any, **_kwargs: Any) -> Any:
                call_order.append("capability")
                return accepted_web_storage_capability_result()

            def _render(_source: str, **kwargs: Any) -> Any:
                call_order.append("render")
                return mock_closure_render_result(_source, **kwargs)

            def _publish(*_args: Any, **_kwargs: Any) -> Any:
                call_order.append("publish")
                return durable_publication_result(FINAL_URL)

            capability.side_effect = _capability
            closure_render.side_effect = _render
            publish.side_effect = _publish
            with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda _jid, fn: fn(state)):
                execute_builder2_media_pipeline(
                    job_id="job-fresh-capability",
                    state=state,
                    plan=state["winnerDevelopmentPlan"],
                    public_base_url="https://ace.example.com",
                    product_description="desc",
                    deps=_mock_pipeline_deps(),
                )
            self.assertEqual(call_order[:3], ["capability", "render", "publish"])
            publish.assert_called_once()
        finally:
            disable_memory_store()

    @patch("engine.runway_video.builder2_video_plan_struct_ok_for_runway", return_value=(True, ""))
    @patch("engine.runway_video._env_api_key", return_value="rk-test")
    @patch.dict({**WORKER_ENV, "ACE_PUBLIC_BASE_URL": "https://ace.example.com"}, clear=True)
    @patch("engine.builder2_media_resume.run_one_media_resume")
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_builder2_tournament_enabled", return_value=True)
    def test_generate_mvp_delegates_to_media_resume_for_builder2(
        self,
        _enabled: Any,
        tournament: Any,
        media_resume: Any,
        _api_key: Any,
        _struct: Any,
    ) -> None:
        enable_memory_store()
        try:
            state = _media_ready_state(job_id="job-fresh-delegate")
            plan = state["winnerDevelopmentPlan"]
            plan["planInferenceMode"] = "builder2_tournament_winner_v1"
            tournament.return_value = plan
            media_resume.return_value = {
                "ok": True,
                "finalVideoAvailable": True,
                "jobCompleted": True,
            }
            with patch("engine.builder2_tournament_store.load_tournament_state") as load_state:
                load_state.return_value = {
                    "mediaResume": {
                        "finalPublicUrl": FINAL_URL,
                        "marketingText": "Marketing copy for the ad.",
                    },
                    "winnerDevelopmentPlan": plan,
                }
                with patch("engine.runway_video.video_job_set_phase"):
                    url, marketing, _headline = _generate_one_video_mvp_body(
                        "Product",
                        "A useful product.",
                        public_base_url="https://ace.example.com",
                        job_id="job-fresh-delegate",
                    )
            media_resume.assert_called_once_with(job_id="job-fresh-delegate")
            self.assertIn("/api/builder2-final-video/", url)
            self.assertTrue(marketing)
        finally:
            disable_memory_store()


class TestFreshProductionIntegration(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()
        self.capability_patch, self.closure_patch, self.publish_patch, self.publish_mock = patch_media_pipeline_durable_finalization(
            FINAL_URL
        )
        self.capability_patch.start()
        self.closure_patch.start()
        self.publish_patch.start()

    def tearDown(self) -> None:
        self.publish_patch.stop()
        self.closure_patch.stop()
        self.capability_patch.stop()
        disable_memory_store()

    @patch("engine.builder2_media_resume.save_tournament_state")
    @patch.dict(
        os.environ,
        {
            **WORKER_ENV,
            "ACE_PUBLIC_BASE_URL": "https://ace.example.com",
        },
        clear=True,
    )
    def test_successful_fresh_production_persists_builder2_final_url(
        self,
        _save: Any,
    ) -> None:
        state = _media_ready_state(job_id="job-fresh-success")
        report = run_one_media_resume(
            job_id="job-fresh-success",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertTrue(report["ok"])
        self.publish_mock.assert_called_once()
        saved_state = _save.call_args[0][1]
        final_arg = str(saved_state["mediaResume"]["finalPublicUrl"])
        self.assertIn("/api/builder2-final-video/", final_arg)
        self.assertEqual(saved_state["mediaResume"]["finalVideoWithClosureUrl"], final_arg)
        self.assertTrue(saved_state["mediaResume"]["finalPublicationVerificationAccepted"])

    @patch("engine.builder2_execution_lease.has_active_lease", return_value=False)
    @patch("engine.builder2_resume_service.is_job_queued", return_value=False)
    @patch("engine.builder2_resume_service.has_active_lease", return_value=False)
    @patch("engine.builder2_resume_service.load_tournament_state")
    @patch("engine.builder2_resume_service.resolve_builder2_resume_stage")
    def test_completed_status_payload_exposes_builder2_final_url(
        self,
        resolver: Any,
        load_state: Any,
        _lease: Any,
        _queued: Any,
        _exec_lease: Any,
    ) -> None:
        saved_state = _media_ready_state(job_id="job-fresh-status")
        media = saved_state.setdefault("mediaResume", {})
        media.update(
            {
                "finalPublicUrl": FINAL_URL,
                "finalVideoWithClosureUrl": FINAL_URL,
                "finalPublicationVerificationAccepted": True,
                "finalPublicationDurableStorageConfirmed": True,
                "finalPublicationBackendKind": "persistent_disk",
            }
        )
        load_state.return_value = saved_state
        resolver.return_value = {
            "jobAlreadyCompleted": True,
            "canResume": False,
            "resumeFromStage": "completed",
            "completedStages": [],
            "reusableArtifacts": [],
            "blockedReason": None,
        }
        payload = build_builder2_status_payload(
            "job-fresh-status",
            {
                "status": "done",
                "video_url": FINAL_URL,
                "builder": "builder2",
                "builder2ResumeContractVersion": "builder2_resume_v1",
            },
            tournament_state=saved_state,
        )
        self.assertEqual(payload["videoUrl"], FINAL_URL)
        self.assertEqual(payload["finalVideoUrl"], FINAL_URL)
        self.assertEqual(payload["status"], "done")

    @patch("engine.builder2_media_resume.save_tournament_state")
    @patch.dict(os.environ, {**WORKER_ENV, "ACE_PUBLIC_BASE_URL": "https://ace.example.com"}, clear=True)
    def test_publication_failure_leaves_recoverable_state_without_completion(
        self,
        save_state: Any,
    ) -> None:
        from engine.builder2_final_video_publication import Builder2FinalPublicationError

        self.publish_patch.stop()
        self.publish_patch = patch(
            "engine.builder2_durable_finalization.publish_builder2_durable_final_video",
            side_effect=Builder2FinalPublicationError("final_publication_not_durable"),
        )
        self.publish_patch.start()
        state = _media_ready_state(job_id="job-fresh-pub-fail")
        report = run_one_media_resume(
            job_id="job-fresh-pub-fail",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps(),
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report.get("recoverableFinalizationFailure"))
        saved = save_state.call_args[0][1]
        self.assertEqual(saved["status"], "media_finalization_incomplete")
        self.assertTrue(saved["mediaContinuationRequired"])
        self.assertEqual(saved["mediaResume"]["mediaResumeStatus"], "finalization_failed")
        self.assertIn("rawRunwayVideoUrl", saved["mediaResume"])

    def test_headline_route_cannot_satisfy_completion_contract(self) -> None:
        state = _media_ready_state(job_id="job-headline-only")
        media = state.setdefault("mediaResume", {})
        headline_url = "https://ace.example.com/api/video-headline/abc12345678901234567890123456789012"
        media.update(
            {
                "finalPublicUrl": headline_url,
                "finalVideoWithClosureUrl": headline_url,
                "advertisingClosureRendered": True,
                "advertisingClosureStatus": "completed",
                "actualFinalVideoDurationSeconds": 13.534,
            }
        )
        ok, _, failures = validate_builder2_media_completion_contract(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=headline_url,
        )
        self.assertFalse(ok)
        self.assertIn("final_publication_route_not_durable", failures)

    @patch("engine.runway_video.builder2_video_plan_struct_ok_for_runway", return_value=(True, ""))
    @patch("engine.runway_video._env_api_key", return_value="rk-test")
    @patch("engine.builder2_media_resume.run_one_media_resume")
    @patch("engine.runway_video.run_builder2_tournament")
    @patch("engine.runway_video.resolve_builder2_tournament_enabled", return_value=True)
    @patch.dict({**WORKER_ENV, "ACE_PUBLIC_BASE_URL": "https://ace.example.com"}, clear=True)
    def test_recoverable_finalization_failure_does_not_use_error_terminal(
        self,
        _enabled: Any,
        tournament: Any,
        media_resume: Any,
        _api_key: Any,
        _struct: Any,
    ) -> None:
        enable_memory_store()
        try:
            plan = _media_ready_state(job_id="job-recoverable")["winnerDevelopmentPlan"]
            plan["planInferenceMode"] = "builder2_tournament_winner_v1"
            tournament.return_value = plan
            media_resume.return_value = {"ok": False, "failureReason": "final_publication_not_durable"}
            with patch("engine.builder2_tournament_store.load_tournament_state") as load_state:
                load_state.return_value = {"status": "media_finalization_incomplete"}
                with patch("engine.runway_video.video_job_set_phase"):
                    with self.assertRaises(RunwayVideoMVPError) as ctx:
                        _generate_one_video_mvp_body("Product", "desc", job_id="job-recoverable")
            self.assertEqual(ctx.exception.args[0], "builder2_media_finalization_recoverable")
        finally:
            disable_memory_store()


class TestFreshProductionDoesNotInvokeRecoveryClis(unittest.TestCase):
    @patch("engine.builder2_media_resume.execute_builder2_media_pipeline")
    @patch("engine.builder2_media_resume.collect_media_resume_missing_paths", return_value=[])
    @patch("engine.builder2_media_resume._load_and_normalize_winner")
    @patch.dict(os.environ, {**WORKER_ENV, "ACE_PUBLIC_BASE_URL": "https://ace.example.com"}, clear=True)
    def test_media_resume_does_not_call_recovery_cli_modules(
        self,
        _normalize: Any,
        _missing: Any,
        pipeline: Any,
    ) -> None:
        enable_memory_store()
        try:
            state = _media_ready_state(job_id="job-no-cli")
            _normalize.return_value = state["winnerDevelopmentPlan"]
            pipeline.return_value = (state, MagicMock(ffmpeg_calls=1, runway_submission_calls=1))

            def _mutate(**kwargs: Any) -> tuple[Any, Any]:
                media = kwargs["state"].setdefault("mediaResume", {})
                media["finalPublicUrl"] = FINAL_URL
                media["finalVideoWithClosureUrl"] = FINAL_URL
                media["marketingText"] = "copy"
                media.update(
                    {
                        "finalPublicationVerificationAccepted": True,
                        "finalPublicationDurableStorageConfirmed": True,
                        "finalPublicationBackendKind": "persistent_disk",
                        "finalPublicationReferencePresent": True,
                        "finalPublicationUploadedByteCount": 128,
                        "advertisingClosureRendered": True,
                        "advertisingClosureStatus": "completed",
                        "actualFinalVideoDurationSeconds": 13.534,
                        "headlineOverlaySkipped": True,
                        "sloganRenderedExactlyOnce": True,
                        "advertisingCopyRenderStages": 1,
                        "copyContractVersion": "builder2_single_slogan_v1",
                        "logoPolicyVersion": "builder2_no_logos_v1",
                        "brandNameRenderedAsPlainText": True,
                        "brandGraphicRendered": False,
                        "logoAssetUsed": False,
                        "logoPolicySatisfied": True,
                    }
                )
                kwargs["state"]["status"] = "completed"
                return kwargs["state"], MagicMock(
                    ffmpeg_calls=1,
                    runway_submission_calls=1,
                    runway_polling_calls=1,
                    start_image_calls=1,
                    sync_legacy_start_image_calls=lambda: None,
                )

            pipeline.side_effect = _mutate
            with patch("engine.builder2_media_resume.redis_configured", return_value=False):
                with patch("engine.builder2_invalid_final_publication_repair.repair_invalid_final_publication_state") as repair:
                    with patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume") as recovery:
                        with patch("engine.builder2_media_finalization_resume.run_finalization_preflight") as preflight:
                            report = run_one_media_resume(job_id="job-no-cli", tournament_state=deepcopy(state))
            self.assertTrue(report["ok"])
            repair.assert_not_called()
            recovery.assert_not_called()
            preflight.assert_not_called()
        finally:
            disable_memory_store()


if __name__ == "__main__":
    unittest.main()
