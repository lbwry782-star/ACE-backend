"""
Builder2 start-image geometry and pipeline tests — mocks only.
"""
from __future__ import annotations

import base64
import io
import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_media_pipeline import MediaPipelineCounters, MediaPipelineDeps, execute_builder2_media_pipeline
from engine.builder2_media_resume import run_one_media_resume
from engine.builder2_start_image_geometry import (
    DEFAULT_BUILDER2_START_IMAGE_GENERATION_SIZE,
    DEFAULT_BUILDER2_START_IMAGE_OUTPUT_SIZE,
    Builder2StartImageGeometryError,
    resolve_builder2_start_image_geometry,
)
from engine.builder2_start_image_pipeline import (
    Builder2StartImagePipelineError,
    StartImageCallCounters,
    generate_builder2_start_image,
    transform_builder2_start_image,
    validate_builder2_runway_start_image_artifact,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_winner_downstream import build_builder2_start_frame_image_prompt
from engine.video_start_image import build_ace_start_frame_image_prompt, generate_video_start_image_data_uri
from tests.test_builder2_media_resume import _media_ready_state, _mock_pipeline_deps, _mock_render_advertising_closure


def _png_data_uri(width: int, height: int, *, color: str = "red") -> str:
    from PIL import Image

    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _source_png_bytes(width: int = 1536, height: int = 1024) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), color="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestBuilder2StartImageGeometry(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_separate_generation_and_output_sizes(self) -> None:
        geometry = resolve_builder2_start_image_geometry()
        self.assertEqual(geometry.imageGenerationSize, "1536x1024")
        self.assertEqual(geometry.startImageOutputSize, "1280x720")
        self.assertEqual(geometry.outputAspectRatio, "16:9")
        self.assertEqual(geometry.cropStrategy, "center_crop")
        self.assertTrue(geometry.resizeRequired)

    @patch.dict(os.environ, {"VIDEO_START_IMAGE_SIZE": "1280x720"}, clear=True)
    def test_legacy_video_start_image_size_maps_to_supported_generation(self) -> None:
        geometry = resolve_builder2_start_image_geometry()
        self.assertEqual(geometry.imageGenerationSize, "1536x1024")
        self.assertEqual(geometry.startImageOutputSize, "1280x720")
        self.assertNotEqual(geometry.imageGenerationSize, "1280x720")

    @patch.dict(os.environ, {"BUILDER2_START_IMAGE_GENERATION_SIZE": "800x600"}, clear=True)
    def test_unsupported_generation_size_fails_before_api(self) -> None:
        with self.assertRaises(Builder2StartImageGeometryError) as ctx:
            resolve_builder2_start_image_geometry()
        self.assertEqual(str(ctx.exception), "builder2_start_image_unsupported_generation_size")

    @patch.dict(os.environ, {}, clear=True)
    def test_crop_box_for_landscape_defaults(self) -> None:
        geometry = resolve_builder2_start_image_geometry()
        self.assertEqual(geometry.cropBox, {"left": 0, "top": 80, "right": 1536, "bottom": 944})
        self.assertEqual(geometry.croppedWidth, 1536)
        self.assertEqual(geometry.croppedHeight, 864)
        self.assertEqual(geometry.outputWidth, 1280)
        self.assertEqual(geometry.outputHeight, 720)

    @patch.dict(os.environ, {}, clear=True)
    def test_dry_run_and_actual_share_geometry_object(self) -> None:
        geometry = resolve_builder2_start_image_geometry()
        metadata = geometry.to_safe_metadata()
        self.assertEqual(metadata["imageGenerationSize"], DEFAULT_BUILDER2_START_IMAGE_GENERATION_SIZE)
        self.assertEqual(metadata["startImageOutputSize"], DEFAULT_BUILDER2_START_IMAGE_OUTPUT_SIZE)
        self.assertTrue(metadata["startImageGeometryAccepted"])

    @patch.dict(os.environ, {}, clear=True)
    def test_transform_crops_then_resizes_without_stretching_source(self) -> None:
        geometry = resolve_builder2_start_image_geometry()
        from PIL import Image

        image = Image.new("RGB", (1536, 1024))
        pixels = image.load()
        for y in range(1024):
            for x in range(1536):
                pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        source = buffer.getvalue()
        output_bytes, metadata = transform_builder2_start_image(source, geometry)
        output = Image.open(io.BytesIO(output_bytes))
        self.assertEqual(output.size, (1280, 720))
        self.assertAlmostEqual(output.size[0] / output.size[1], 16 / 9, places=3)
        self.assertEqual(geometry.croppedHeight, 864)
        self.assertNotEqual(geometry.generationHeight, geometry.croppedHeight)


class TestBuilder2StartImagePrompt(unittest.TestCase):
    def test_prompt_includes_safe_area_instruction(self) -> None:
        plan: Dict[str, Any] = {
            "planInferenceMode": "builder2_tournament_winner_v1",
            "productNameResolved": "ACE Product",
            "coreVisualIdea": "maximum connection",
            "openingFrameDescription": "friends hugging warmly",
            "sceneVariations": ["young couple hugging"],
            "structureType": "variation_montage",
            "headlineDecision": {"decision": "omit"},
            "headline": "",
        }
        prompt = build_ace_start_frame_image_prompt(plan)
        self.assertIn("central 16:9 safe area", prompt)
        self.assertIn("crop margins", prompt)

    def test_headline_omit_adds_no_advertising_copy(self) -> None:
        plan: Dict[str, Any] = {
            "productNameResolved": "ACE Product",
            "coreVisualIdea": "maximum connection",
            "openingFrameDescription": "friends hugging warmly",
            "structureType": "variation_montage",
            "headlineDecision": {"decision": "omit"},
            "headlineText": "Buy ACE Now",
            "headline": "Buy ACE Now",
        }
        prompt = build_builder2_start_frame_image_prompt(plan, duration_seconds=7)
        self.assertNotIn("Buy ACE Now", prompt)
        self.assertIn("No text", prompt)


class TestBuilder2StartImageGenerationAccounting(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("engine.builder2_start_image_pipeline.openai_retry.openai_call_with_retry")
    def test_http_400_counts_submitted_call_not_generated_image(self, openai_call: Any) -> None:
        class FakeApiError(Exception):
            status_code = 400

        openai_call.side_effect = FakeApiError("Invalid size '1280x720'")
        plan = {"planInferenceMode": "builder2_tournament_winner_v1", "coreVisualIdea": "scene", "openingFrameDescription": "opening"}
        with self.assertRaises(Builder2StartImagePipelineError) as ctx:
            generate_builder2_start_image(plan)
        result = ctx.exception.result
        assert result is not None
        self.assertTrue(result.api_submitted)
        self.assertEqual(result.counters.startImageNormalCalls, 1)
        self.assertEqual(result.counters.startImageGeneratedCount, 0)
        self.assertEqual(ctx.exception.failure_stage, "start_image_generation")
        self.assertEqual(str(ctx.exception.args[0]), "builder2_media_start_image_api_rejected")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("engine.builder2_start_image_pipeline.openai_retry.openai_call_with_retry")
    def test_successful_generation_persists_transformed_metadata(self, openai_call: Any) -> None:
        geometry = resolve_builder2_start_image_geometry()
        source_b64 = base64.b64encode(_source_png_bytes()).decode("ascii")
        response = MagicMock()
        response.data = [MagicMock(b64_json=source_b64)]
        openai_call.return_value = response
        plan = {"planInferenceMode": "builder2_tournament_winner_v1", "coreVisualIdea": "scene", "openingFrameDescription": "opening"}
        result = generate_builder2_start_image(plan)
        self.assertEqual(result.counters.startImageNormalCalls, 1)
        self.assertEqual(result.counters.startImageGeneratedCount, 1)
        self.assertIsNotNone(result.data_uri)
        assert result.data_uri is not None
        validate_builder2_runway_start_image_artifact(result.data_uri, geometry)
        self.assertEqual(result.metadata["startImageGenerationSize"], "1536x1024")
        self.assertEqual(result.metadata["startImageOutputWidth"], 1280)
        self.assertEqual(result.metadata["startImageOutputHeight"], 720)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "VIDEO_START_IMAGE_SIZE": "1280x720"}, clear=True)
    @patch("engine.builder2_start_image_pipeline.OpenAI")
    def test_openai_never_receives_1280x720(self, openai_cls: Any) -> None:
        client = MagicMock()
        openai_cls.return_value = client
        client.images.generate.side_effect = AssertionError("should not be called")
        with patch("engine.builder2_start_image_pipeline.openai_retry.openai_call_with_retry") as openai_call:
            openai_call.side_effect = AssertionError("unexpected call")
            plan = {"planInferenceMode": "builder2_tournament_winner_v1", "coreVisualIdea": "scene", "openingFrameDescription": "opening"}
            geometry = resolve_builder2_start_image_geometry()
            self.assertEqual(geometry.imageGenerationSize, "1536x1024")


class TestBuilder2StartImageMediaPipeline(unittest.TestCase):
    def setUp(self) -> None:
        from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
        from engine.builder2_tournament_store import enable_memory_store

        enable_memory_store()
        MediaResumeIsolationGuard.begin()
        MediaResumeIsolationGuard.enable_start_image()
        MediaResumeIsolationGuard.enable_runway()
        MediaResumeIsolationGuard.enable_ffmpeg()
        self.render_patch = patch(
            "engine.builder2_advertising_closure_pipeline.render_advertising_closure_for_state",
            side_effect=_mock_render_advertising_closure,
        )
        self.render_patch.start()

    def tearDown(self) -> None:
        from engine.builder2_media_resume_guard import MediaResumeIsolationGuard
        from engine.builder2_tournament_store import disable_memory_store

        self.render_patch.stop()
        MediaResumeIsolationGuard.end()
        disable_memory_store()

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_existing_valid_artifact_is_reused(self) -> None:
        state = _media_ready_state(job_id="job-start-image-reuse")
        artifact = _png_data_uri(1280, 720)
        state["mediaResume"] = {"startImageArtifact": artifact, "startImageStatus": "completed"}
        deps = _mock_pipeline_deps()
        deps.generate_start_image = unittest.mock.Mock(return_value="should-not-run")
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            _, counters = execute_builder2_media_pipeline(
                job_id="job-start-image-reuse",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                deps=deps,
            )
        deps.generate_start_image.assert_not_called()
        self.assertEqual(counters.start_image_normal_calls, 0)
        self.assertEqual(counters.start_image_generated_count, 0)

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("engine.builder2_start_image_pipeline.generate_builder2_start_image")
    def test_runway_not_submitted_after_start_image_failure(self, generate_mock: Any) -> None:
        state = _media_ready_state(job_id="job-start-image-fail")
        generate_mock.side_effect = Builder2StartImagePipelineError(
            "builder2_media_start_image_api_rejected",
            failure_stage="start_image_generation",
            result=type(
                "R",
                (),
                {
                    "counters": StartImageCallCounters(startImageNormalCalls=1),
                    "api_submitted": True,
                    "api_status": 400,
                    "api_error_category": "FakeApiError",
                    "submitted_size": "1536x1024",
                    "model_name": "gpt-image-1.5",
                },
            )(),
        )
        submit_mock = unittest.mock.Mock(return_value="should-not-run")
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            with patch("engine.builder2_media_pipeline._default_submit_runway_task", submit_mock):
                with self.assertRaises(Builder2TournamentError):
                    execute_builder2_media_pipeline(
                        job_id="job-start-image-fail",
                        state=state,
                        plan=state["winnerDevelopmentPlan"],
                        public_base_url="https://example.com",
                        product_description="desc",
                        deps=None,
                    )
        submit_mock.assert_not_called()
        self.assertNotIn("startImageArtifact", state.get("mediaResume", {}))

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test"}, clear=False)
    def test_runway_receives_only_output_sized_artifact(self) -> None:
        state = _media_ready_state(job_id="job-runway-size")
        captured: Dict[str, Any] = {}

        def _submit(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "task-size-1"

        deps = _mock_pipeline_deps()
        deps.generate_start_image = lambda plan: _png_data_uri(1280, 720)
        deps.submit_runway_task = _submit
        with patch("engine.builder2_media_pipeline.patch_tournament_state", side_effect=lambda job_id, fn: fn(state)):
            _, counters = execute_builder2_media_pipeline(
                job_id="job-runway-size",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                public_base_url="https://example.com",
                product_description="desc",
                deps=deps,
            )
        validate_builder2_runway_start_image_artifact(captured["prompt_image_data_uri"])
        self.assertEqual(counters.start_image_normal_calls, 1)


class TestBuilder2StartImageDryRun(unittest.TestCase):
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
        },
        clear=False,
    )
    def test_dry_run_reports_geometry(self) -> None:
        state = _media_ready_state(job_id="job-dry-geometry")
        report = run_one_media_resume(job_id="job-dry-geometry", tournament_state=deepcopy(state), dry_run=True)
        self.assertTrue(report["ok"])
        geometry = report["startImageGeometry"]
        self.assertEqual(geometry["imageGenerationSize"], "1536x1024")
        self.assertEqual(geometry["startImageOutputSize"], "1280x720")
        self.assertEqual(geometry["cropBox"], {"left": 0, "top": 80, "right": 1536, "bottom": 944})
        self.assertTrue(geometry["startImageGeometryAccepted"])

    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_RESUME_DRY_RUN": "true",
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "ACE_PUBLIC_BASE_URL": "https://example.com",
            "BUILDER2_START_IMAGE_GENERATION_SIZE": "1280x720",
        },
        clear=False,
    )
    def test_dry_run_fails_on_unsupported_generation_size(self) -> None:
        state = _media_ready_state(job_id="job-dry-bad-size")
        report = run_one_media_resume(job_id="job-dry-bad-size", tournament_state=deepcopy(state), dry_run=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureReason"], "builder2_start_image_unsupported_generation_size")
        self.assertEqual(report["failureStage"], "start_image_configuration")


class TestBuilder2StartImageIsolation(unittest.TestCase):
    def setUp(self) -> None:
        from engine.builder2_tournament_store import enable_memory_store

        enable_memory_store()

    def tearDown(self) -> None:
        from engine.builder2_tournament_store import disable_memory_store

        disable_memory_store()

    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_model_env_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "rk-test", "OPENAI_API_KEY": "sk-test", "ACE_PUBLIC_BASE_URL": "https://example.com"}, clear=False)
    def test_zero_reasoning_calls_on_media_resume(self) -> None:
        state = _media_ready_state(job_id="job-zero-reasoning-start-image")
        report = run_one_media_resume(
            job_id="job-zero-reasoning-start-image",
            tournament_state=deepcopy(state),
            dry_run=False,
            pipeline_deps=_mock_pipeline_deps_with_valid_image(),
        )
        self.assertEqual(report["strategyCalls"], 0)
        self.assertEqual(report["creatorCalls"], 0)
        self.assertEqual(report["judgeCalls"], 0)
        self.assertEqual(report["winnerCalls"], 0)


def _mock_pipeline_deps_with_valid_image() -> MediaPipelineDeps:
    deps = _mock_pipeline_deps()
    deps.generate_start_image = lambda plan: _png_data_uri(1280, 720)
    return deps


if __name__ == "__main__":
    unittest.main()
