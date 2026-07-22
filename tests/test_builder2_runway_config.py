"""
Builder2 Runway video model routing and generation path tests.

Run: python -m unittest tests.test_builder2_runway_config -v
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder1_planning_profile import quality_model
from engine.builder2_runway_config import (
    DEFAULT_BUILDER2_RUNWAY_VIDEO_MODEL,
    DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS,
    Builder2RunwayConfigError,
    builder2_runway_generation_mode,
    builder2_runway_requires_start_image,
    resolve_builder2_runway_video_model,
    resolve_builder2_video_duration_seconds,
)
from engine.runway_video import RunwayVideoMVPError, _create_image_to_video_task, _create_text_to_video_task
from engine.video_start_image import build_ace_start_frame_image_prompt


class TestBuilder2RunwayModelConfig(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_default_model_is_gen4_turbo(self) -> None:
        self.assertEqual(resolve_builder2_runway_video_model(), "gen4_turbo")
        self.assertEqual(DEFAULT_BUILDER2_RUNWAY_VIDEO_MODEL, "gen4_turbo")

    @patch.dict(os.environ, {"BUILDER2_RUNWAY_VIDEO_MODEL": "gen4.5"}, clear=True)
    def test_gen4_5_env_override(self) -> None:
        self.assertEqual(resolve_builder2_runway_video_model(), "gen4.5")

    @patch.dict(os.environ, {"BUILDER2_RUNWAY_VIDEO_MODEL": "gen4_turbo"}, clear=True)
    def test_gen4_turbo_requires_start_image(self) -> None:
        self.assertTrue(builder2_runway_requires_start_image("gen4_turbo"))
        self.assertEqual(builder2_runway_generation_mode("gen4_turbo"), "image_to_video")

    @patch.dict(os.environ, {"BUILDER2_RUNWAY_VIDEO_MODEL": "gen4.5"}, clear=True)
    def test_gen4_5_does_not_require_start_image(self) -> None:
        self.assertFalse(builder2_runway_requires_start_image("gen4.5"))
        self.assertEqual(builder2_runway_generation_mode("gen4.5"), "text_to_video")

    @patch.dict(os.environ, {"BUILDER2_RUNWAY_VIDEO_MODEL": "veo3"}, clear=True)
    def test_invalid_model_raises(self) -> None:
        with self.assertRaises(Builder2RunwayConfigError):
            resolve_builder2_runway_video_model()

    @patch.dict(
        os.environ,
        {
            "BUILDER2_RUNWAY_VIDEO_MODEL": "gen4_turbo",
            "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
            "BUILDER1_PLANNING_PROFILE": "QUALITY",
        },
        clear=True,
    )
    def test_builder1_env_does_not_change_builder2_runway_model(self) -> None:
        self.assertEqual(resolve_builder2_runway_video_model(), "gen4_turbo")
        self.assertEqual(quality_model(), "gpt-5.6-sol")


class TestBuilder2StartImagePrompt(unittest.TestCase):
    def test_prompt_uses_active_plan_fields(self) -> None:
        plan: Dict[str, Any] = {
            "productNameResolved": "ACE Widget",
            "coreVisualIdea": "maximum connection",
            "openingFrameDescription": "friends hugging warmly",
            "sceneVariations": ["young couple hugging", "parent and child hugging"],
            "videoPrompt": "A 7-second montage of hug moments. No text.",
        }
        prompt = build_ace_start_frame_image_prompt(plan)
        self.assertIn("maximum connection", prompt)
        self.assertIn("friends hugging warmly", prompt)
        self.assertIn("ACE Widget", prompt)
        self.assertIn("No text", prompt)
        self.assertNotIn("objectA", prompt)


class TestBuilder2RunwayTaskCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.session = MagicMock()
        self.response = MagicMock()
        self.response.status_code = 200
        self.response.json.return_value = {"id": "task-123"}
        self.response.content = b"{}"
        self.response.text = "{}"
        self.session.post.return_value = self.response

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "test-key"}, clear=False)
    @patch("engine.runway_video._headers", return_value={"Authorization": "Bearer test"})
    def test_gen4_5_text_to_video_payload(self, _headers_mock) -> None:
        task_id = _create_text_to_video_task(
            self.session,
            "https://api.dev.runwayml.com",
            "gen4.5",
            "A simple cinematic scene. No text.",
            duration_seconds=resolve_builder2_video_duration_seconds(),
        )
        self.assertEqual(task_id, "task-123")
        call_args = self.session.post.call_args
        self.assertEqual(call_args[0][0], "https://api.dev.runwayml.com/v1/text_to_video")
        body = call_args[1]["json"]
        self.assertEqual(body["model"], "gen4.5")
        self.assertEqual(body["ratio"], "1280:720")
        self.assertEqual(body["duration"], DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS)
        self.assertNotIn("promptImage", body)

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "test-key"}, clear=False)
    @patch("engine.runway_video._headers", return_value={"Authorization": "Bearer test"})
    def test_gen4_turbo_image_to_video_payload(self, _headers_mock) -> None:
        data_uri = "data:image/png;base64,abc123"
        task_id = _create_image_to_video_task(
            self.session,
            "https://api.dev.runwayml.com",
            "gen4_turbo",
            "Animate the opening hug moment. No text.",
            data_uri,
            duration_seconds=resolve_builder2_video_duration_seconds(),
        )
        self.assertEqual(task_id, "task-123")
        call_args = self.session.post.call_args
        self.assertEqual(call_args[0][0], "https://api.dev.runwayml.com/v1/image_to_video")
        body = call_args[1]["json"]
        self.assertEqual(body["model"], "gen4_turbo")
        self.assertEqual(body["promptImage"], data_uri)
        self.assertEqual(body["promptText"], "Animate the opening hug moment. No text.")
        self.assertEqual(body["ratio"], "1280:720")
        self.assertEqual(body["duration"], DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS)

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "test-key"}, clear=False)
    @patch("engine.runway_video._headers", return_value={"Authorization": "Bearer test"})
    def test_gen4_turbo_missing_prompt_image_fails(self, _headers_mock) -> None:
        with self.assertRaises(RunwayVideoMVPError) as ctx:
            _create_image_to_video_task(
                self.session,
                "https://api.dev.runwayml.com",
                "gen4_turbo",
                "prompt",
                "",
                duration_seconds=7,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_start_image_generation_failed")
        self.session.post.assert_not_called()

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "test-key"}, clear=False)
    @patch("engine.runway_video._headers", return_value={"Authorization": "Bearer test"})
    def test_invalid_model_for_text_path_fails(self, _headers_mock) -> None:
        with self.assertRaises(RunwayVideoMVPError) as ctx:
            _create_text_to_video_task(
                self.session,
                "https://api.dev.runwayml.com",
                "gen4_turbo",
                "prompt",
                duration_seconds=7,
            )
        self.assertEqual(ctx.exception.args[0], "unsupported_runway_model")
        self.session.post.assert_not_called()


class TestBuilder2RunwayGenerationRouting(unittest.TestCase):
    PLAN: Dict[str, Any] = {
        "productNameResolved": "Product",
        "headlineText": "Product headline",
        "headlineDecision": "include_product_name",
        "advertisingPromise": "headline",
        "headline": "headline",
        "headlineCoreKeyword": "headline",
        "coreVisualIdea": "connection",
        "sceneVariations": ["friends hugging", "couple hugging"],
        "sceneConcept": "friends hugging | couple hugging",
        "videoPrompt": "Montage of hugs. No text.",
        "videoPromptCore": "Montage of hugs. No text.",
        "openingFrameDescription": "friends hugging",
        "language": "en",
    }

    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_RUNWAY_VIDEO_MODEL": "gen4_turbo",
            "BUILDER2_TOURNAMENT_ENABLED": "false",
        },
        clear=False,
    )
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value=None)
    @patch("engine.runway_video.fetch_video_plan_o3", return_value=(PLAN, ""))
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-x")
    @patch("engine.runway_video._create_text_to_video_task", return_value="task-y")
    def test_start_image_failure_does_not_fallback_to_gen4_5(
        self,
        text_task_mock,
        image_task_mock,
        _product_name,
        _plan,
        _start_image,
        _redis_name,
    ) -> None:
        from engine.runway_video import _generate_one_video_mvp_body

        with self.assertRaises(RunwayVideoMVPError) as ctx:
            _generate_one_video_mvp_body("Product", "A useful product.", job_id="job-1")
        self.assertEqual(ctx.exception.args[0], "builder2_start_image_generation_failed")
        _start_image.assert_called_once()
        image_task_mock.assert_not_called()
        text_task_mock.assert_not_called()

    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch("engine.runway_video.video_job_set_phase")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_TOURNAMENT_ENABLED": "false",
        },
        clear=False,
    )
    @patch("engine.runway_video.generate_video_start_image_data_uri", return_value="data:image/png;base64,x")
    @patch("engine.runway_video.fetch_video_plan_o3", return_value=(PLAN, ""))
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    @patch("engine.runway_video._sleep_poll_interval")
    @patch("engine.runway_video._poll_get_task_once")
    @patch("engine.runway_video._create_image_to_video_task", return_value="task-x")
    @patch("engine.runway_video._create_text_to_video_task")
    def test_gen4_turbo_uses_image_to_video_only(
        self,
        text_task_mock,
        image_task_mock,
        poll_mock,
        _sleep_mock,
        _product_name,
        _plan,
        _start_image,
        _phase,
        _redis_name,
    ) -> None:
        poll_mock.return_value = {"status": "SUCCEEDED", "output": ["https://runway/video.mp4"]}
        from engine.runway_video import _generate_one_video_mvp_body

        with patch.dict(
            os.environ,
            {"BUILDER2_RUNWAY_VIDEO_MODEL": "gen4_turbo", "BUILDER2_TOURNAMENT_ENABLED": "false"},
            clear=False,
        ):
            with patch("engine.runway_video.postprocess_video_headline", return_value="https://final/video.mp4"):
                with patch("engine.runway_video._fallback_packaging_marketing_copy", return_value="copy"):
                    with patch("engine.runway_video.record_ad_promise_generation_success"):
                        _generate_one_video_mvp_body("Product", "A useful product.", job_id="job-2")
        _start_image.assert_called_once()
        image_task_mock.assert_called_once()
        text_task_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
