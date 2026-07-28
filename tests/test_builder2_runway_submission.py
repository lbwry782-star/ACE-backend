"""
Runway API URL and Builder2 submission tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import requests

from engine.builder2_media_pipeline import MediaPipelineCounters, MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
from engine.builder2_runway_submission import (
    audit_media_resume_start_image,
    build_builder2_runway_dry_run_report,
    submit_builder2_runway_task,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.runway_api_urls import (
    RunwayUrlConfigurationError,
    build_runway_api_url,
    build_runway_image_to_video_url,
    build_runway_task_poll_url,
    validate_runway_api_url,
)
from engine.runway_prompt_budget import (
    RunwayPromptBudgetError,
    count_utf16_code_units,
    normalize_runway_prompt_to_budget,
)
from tests.builder2_durable_finalization_test_helpers import patch_media_pipeline_durable_finalization
from tests.test_builder2_media_resume import _media_ready_state, _mock_start_image_data_uri


class TestRunwayApiUrls(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_origin_without_v1_produces_single_prefix(self) -> None:
        resolution = build_runway_image_to_video_url(configured_base="https://api.dev.runwayml.com")
        self.assertEqual(resolution.normalizedPath, "/v1/image_to_video")
        self.assertEqual(resolution.absoluteUrl, "https://api.dev.runwayml.com/v1/image_to_video")
        self.assertEqual(resolution.normalizedPath.count("/v1"), 1)

    @patch.dict(os.environ, {"RUNWAY_API_BASE": "https://api.dev.runwayml.com/v1"}, clear=True)
    def test_base_with_v1_suffix_still_single_prefix(self) -> None:
        resolution = build_runway_image_to_video_url()
        self.assertTrue(resolution.configuredBaseHadVersionPrefix)
        self.assertEqual(resolution.normalizedPath, "/v1/image_to_video")
        self.assertNotIn("/v1/v1/", resolution.absoluteUrl)

    @patch.dict(os.environ, {"RUNWAY_API_BASE": "https://api.dev.runwayml.com/v1/"}, clear=True)
    def test_trailing_slash_is_normalized(self) -> None:
        resolution = build_runway_image_to_video_url()
        self.assertEqual(resolution.absoluteUrl, "https://api.dev.runwayml.com/v1/image_to_video")

    @patch.dict(os.environ, {"RUNWAY_API_BASE": "https://api.dev.runwayml.com/v1"}, clear=True)
    def test_double_v1_is_impossible(self) -> None:
        resolution = build_runway_image_to_video_url()
        poll = build_runway_task_poll_url("task-123")
        self.assertNotIn("/v1/v1/", resolution.absoluteUrl)
        self.assertNotIn("/v1/v1/", poll.absoluteUrl)

    @patch.dict(os.environ, {}, clear=True)
    def test_poll_path_template(self) -> None:
        resolution = build_runway_task_poll_url("abc123", configured_base="https://api.dev.runwayml.com")
        self.assertEqual(resolution.normalizedPath, "/v1/tasks/abc123")

    def test_invalid_endpoint_fails_before_http(self) -> None:
        resolution = build_runway_api_url("image_to_video", configured_base="http://api.dev.runwayml.com")
        with self.assertRaises(RunwayUrlConfigurationError):
            validate_runway_api_url(resolution)

    @patch.dict(os.environ, {"RUNWAY_API_BASE": "https://api.dev.runwayml.com/v1"}, clear=True)
    def test_dry_run_detects_single_version_prefix(self) -> None:
        from engine.builder2_tournament_store import disable_memory_store, enable_memory_store

        enable_memory_store()
        try:
            state = _media_ready_state(job_id="job-runway-dry-url")
            report = build_builder2_runway_dry_run_report(
                plan=state["winnerDevelopmentPlan"],
                state=state,
                runway_model="gen4_turbo",
                duration_seconds=7,
                ratio="1280:720",
            )
        finally:
            disable_memory_store()
        self.assertEqual(report["runwayVersionPrefixCount"], 1)
        self.assertTrue(report["runwayEndpointAccepted"])


class TestRunwayPromptBudget(unittest.TestCase):
    def test_utf16_measurement_counts_surrogates(self) -> None:
        self.assertEqual(count_utf16_code_units("a"), 1)
        self.assertEqual(count_utf16_code_units("😀"), 2)

    def test_prompt_validated_after_physics_suffix(self) -> None:
        core = "Core visual action with subject and location. " * 20
        result = normalize_runway_prompt_to_budget(
            core_prompt=core,
            physics_suffix="REALISM: weight, contact, resistance; no frictionless sliding or gliding.",
            maximum_utf16_units=1000,
        )
        self.assertLessEqual(result.utf16Length, 1000)
        self.assertIn("REALISM:", result.promptText)

    def test_over_budget_irreducible_core_fails(self) -> None:
        core = "X" * 1200
        with self.assertRaises(RunwayPromptBudgetError):
            normalize_runway_prompt_to_budget(core_prompt=core, maximum_utf16_units=1000)

    def test_1037_fixture_normalizes_to_budget(self) -> None:
        core = "B" * 850
        result = normalize_runway_prompt_to_budget(core_prompt=core)
        self.assertGreater(result.utf16LengthBefore, 1000)
        self.assertLessEqual(result.utf16Length, 1000)

    def test_concise_visual_policy_present(self) -> None:
        result = normalize_runway_prompt_to_budget(core_prompt="Opening fan scene with motion.")
        self.assertTrue(result.visualPolicyPresent)
        self.assertIn("VISUAL POLICY", result.promptText)


class TestRunwaySubmissionAccounting(unittest.TestCase):
    def setUp(self) -> None:
        from engine.builder2_tournament_store import enable_memory_store

        enable_memory_store()
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.enable_start_image()
        MediaResumeIsolationGuard.enable_runway()
        MediaResumeIsolationGuard.enable_ffmpeg()
        self.capability_patch, self.closure_patch, self.publish_patch, self.publish_mock = (
            patch_media_pipeline_durable_finalization()
        )
        self.capability_patch.start()
        self.closure_patch.start()
        self.publish_patch.start()

    def tearDown(self) -> None:
        from engine.builder2_tournament_store import disable_memory_store

        self.publish_patch.stop()
        self.closure_patch.stop()
        self.capability_patch.stop()
        MediaResumeIsolationGuard.end()
        disable_memory_store()

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "RUNWAY_API_BASE": "https://api.dev.runwayml.com/v1"}, clear=False)
    def test_http_404_counts_submission_not_task(self) -> None:
        state = _media_ready_state(job_id="job-runway-404")
        state["mediaResume"] = {
            "startImageArtifact": _mock_start_image_data_uri(),
            "startImageStatus": "completed",
            "startImageOutputWidth": 1280,
            "startImageOutputHeight": 720,
        }
        session = MagicMock()
        response = MagicMock()
        response.status_code = 404
        response.json.return_value = {"error": "not found"}
        session.post.return_value = response
        with self.assertRaises(Builder2TournamentError) as ctx:
            submit_builder2_runway_task(
                session=session,
                api_key="rk-test",
                plan=state["winnerDevelopmentPlan"],
                runway_model="gen4_turbo",
                duration_seconds=7,
                prompt_image_data_uri=state["mediaResume"]["startImageArtifact"],
            )
        self.assertEqual(str(ctx.exception.args[0]), "builder2_runway_submission_http_error")
        submission = getattr(ctx.exception, "runway_submission_result", None)
        assert submission is not None
        self.assertTrue(submission.request_submitted)
        self.assertFalse(submission.task_created)
        self.assertEqual(submission.http_status, 404)
        called_url = session.post.call_args.args[0]
        self.assertEqual(called_url, "https://api.dev.runwayml.com/v1/image_to_video")

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_existing_start_image_is_reused_without_openai(self) -> None:
        state = _media_ready_state(job_id="job-runway-reuse-image")
        artifact = _mock_start_image_data_uri()
        state["mediaResume"] = {
            "startImageArtifact": artifact,
            "startImageStatus": "completed",
            "startImageOutputWidth": 1280,
            "startImageOutputHeight": 720,
        }
        audit = audit_media_resume_start_image(state)
        self.assertTrue(audit["startImageAvailable"])
        self.assertTrue(audit["startImageReusable"])
        deps = MediaPipelineDeps(
            generate_start_image=MagicMock(return_value="should-not-run"),
            submit_runway_task=lambda **kwargs: "task-created-1",
            poll_runway_task=lambda **kwargs: ("SUCCEEDED", "https://runway/mock.mp4"),
            postprocess_video=lambda **kwargs: kwargs["runway_url"],
            compose_marketing_copy=lambda **kwargs: "copy",
        )
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            _, counters = execute_builder2_media_pipeline(
                job_id="job-runway-reuse-image",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                deps=deps,
            )
        deps.generate_start_image.assert_not_called()
        self.assertTrue(counters.start_image_reused)
        self.assertEqual(counters.start_image_normal_calls, 0)
        self.assertEqual(counters.runway_submission_calls, 1)
        self.assertEqual(counters.runway_task_created_count, 1)


class TestRunwayDryRunParity(unittest.TestCase):
    def setUp(self) -> None:
        from engine.builder2_tournament_store import enable_memory_store

        enable_memory_store()

    def tearDown(self) -> None:
        from engine.builder2_tournament_store import disable_memory_store

        disable_memory_store()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_RESUME_DRY_RUN": "true",
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "ACE_PUBLIC_BASE_URL": "https://example.com",
            "RUNWAY_API_BASE": "https://api.dev.runwayml.com/v1",
        },
        clear=False,
    )
    def test_dry_run_reports_runway_contract(self) -> None:
        state = _media_ready_state(job_id="job-runway-dry-contract")
        state["mediaResume"] = {
            "startImageArtifact": _mock_start_image_data_uri(),
            "startImageStatus": "completed",
            "startImageOutputWidth": 1280,
            "startImageOutputHeight": 720,
        }
        report = run_one_media_resume(job_id="job-runway-dry-contract", tournament_state=deepcopy(state), dry_run=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["runwayCreateEndpointPath"], "/v1/image_to_video")
        self.assertEqual(report["runwayTaskEndpointTemplate"], "/v1/tasks/{taskId}")
        self.assertEqual(report["runwayVersionPrefixCount"], 1)
        self.assertLessEqual(report["runwayPromptUtf16Length"], 1000)


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
