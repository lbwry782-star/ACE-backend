"""
Builder2 video duration configuration tests (BUILDER2_VIDEO_DURATION_SECONDS).

Run: python -m unittest tests.test_builder2_video_duration -v
"""
from __future__ import annotations

import inspect
import os
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder1_planning_profile import quality_model
from engine.builder2_runway_config import (
    DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS,
    Builder2RunwayConfigError,
    resolve_builder2_video_duration_seconds,
)
from engine import video_planning
from engine.runway_video import RunwayVideoMVPError, _create_image_to_video_task, _create_text_to_video_task
from engine.video_headline_postprocess import postprocess_video_headline
from engine.video_start_image import build_ace_start_frame_image_prompt


class TestBuilder2VideoDurationConfig(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_default_duration_is_seven(self) -> None:
        self.assertEqual(resolve_builder2_video_duration_seconds(), 7)
        self.assertEqual(DEFAULT_BUILDER2_VIDEO_DURATION_SECONDS, 7)

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "2"}, clear=True)
    def test_minimum_duration_two_accepted(self) -> None:
        self.assertEqual(resolve_builder2_video_duration_seconds(), 2)

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "10"}, clear=True)
    def test_maximum_duration_ten_accepted(self) -> None:
        self.assertEqual(resolve_builder2_video_duration_seconds(), 10)

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "1"}, clear=True)
    def test_duration_one_rejected(self) -> None:
        with self.assertRaises(Builder2RunwayConfigError) as ctx:
            resolve_builder2_video_duration_seconds()
        self.assertIn("builder2_invalid_video_duration", str(ctx.exception))

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "11"}, clear=True)
    def test_duration_eleven_rejected(self) -> None:
        with self.assertRaises(Builder2RunwayConfigError) as ctx:
            resolve_builder2_video_duration_seconds()
        self.assertIn("builder2_invalid_video_duration", str(ctx.exception))

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "abc"}, clear=True)
    def test_non_numeric_duration_rejected(self) -> None:
        with self.assertRaises(Builder2RunwayConfigError) as ctx:
            resolve_builder2_video_duration_seconds()
        self.assertIn("builder2_invalid_video_duration", str(ctx.exception))

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "7.5"}, clear=True)
    def test_float_duration_rejected(self) -> None:
        with self.assertRaises(Builder2RunwayConfigError):
            resolve_builder2_video_duration_seconds()

    @patch.dict(
        os.environ,
        {
            "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
            "BUILDER1_PLANNING_PROFILE": "QUALITY",
            "VIDEO_DURATION": "5",
        },
        clear=True,
    )
    def test_builder1_env_does_not_change_builder2_duration(self) -> None:
        self.assertEqual(resolve_builder2_video_duration_seconds(), 7)
        self.assertEqual(quality_model(), "gpt-5.6-sol")


class TestBuilder2VideoDurationRunwayPayload(unittest.TestCase):
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
    def test_gen4_5_text_to_video_duration_is_integer_seven(self, _headers_mock) -> None:
        _create_text_to_video_task(
            self.session,
            "https://api.dev.runwayml.com",
            "gen4.5",
            "A simple cinematic scene. No text.",
            duration_seconds=7,
        )
        body = self.session.post.call_args[1]["json"]
        self.assertEqual(body["duration"], 7)
        self.assertIsInstance(body["duration"], int)

    @patch.dict(os.environ, {"RUNWAY_API_KEY": "test-key"}, clear=False)
    @patch("engine.runway_video._headers", return_value={"Authorization": "Bearer test"})
    def test_gen4_turbo_image_to_video_duration_is_integer_seven(self, _headers_mock) -> None:
        _create_image_to_video_task(
            self.session,
            "https://api.dev.runwayml.com",
            "gen4_turbo",
            "Animate the opening hug moment. No text.",
            "data:image/png;base64,abc123",
            duration_seconds=7,
        )
        body = self.session.post.call_args[1]["json"]
        self.assertEqual(body["duration"], 7)
        self.assertIsInstance(body["duration"], int)


class TestBuilder2VideoDurationPlanning(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_planning_prompt_includes_dynamic_duration(self) -> None:
        block = video_planning._planner_duration_instruction_block()
        self.assertIn("7 seconds", block)
        self.assertIn("TOTAL video length", block)

    @patch.dict(os.environ, {"BUILDER2_VIDEO_DURATION_SECONDS": "8"}, clear=True)
    def test_planning_prompt_uses_env_duration(self) -> None:
        instructions = video_planning._build_video_planner_instructions("he")
        self.assertIn("8-second montage", instructions)
        keys = video_planning._json_keys_block()
        self.assertIn("8-second montage", keys)

    @patch.dict(os.environ, {}, clear=True)
    def test_active_planning_has_no_hardcoded_five_second_instruction(self) -> None:
        flow = video_planning._planner_keyword_scene_flow_block()
        instructions = video_planning._build_video_planner_instructions("en")
        repair = video_planning._build_scene_plan_repair_input(
            base_attempt_input="base",
            product_name="P",
            product_description="desc",
            previous_plan={},
            reason="test",
        )
        motion, tag = video_planning._runway_variation_montage_camera_focus()
        for text in (flow, instructions, repair, motion, video_planning._json_keys_block()):
            self.assertNotIn("5-second", text.lower())
            self.assertNotIn("5 seconds", text.lower())
            self.assertNotIn("within 5 seconds", text.lower())
        self.assertEqual(tag, "variation_montage_7s")

    @patch.dict(os.environ, {}, clear=True)
    def test_runway_prompt_builder_uses_dynamic_duration(self) -> None:
        plan: Dict[str, Any] = {
            "coreVisualIdea": "connection",
            "sceneVariations": ["friends hugging", "couple hugging"],
            "videoPrompt": "Montage of hugs. No text.",
            "language": "en",
        }
        prompt = video_planning.build_runway_prompt_from_plan(plan)
        self.assertIn("7-second", prompt)
        self.assertNotIn("5-second", prompt.lower())


class TestBuilder2VideoDurationJobFailure(unittest.TestCase):
    @patch("engine.runway_video.video_job_set_resolved_product_name")
    @patch.dict(
        os.environ,
        {
            "RUNWAY_API_KEY": "rk-test",
            "OPENAI_API_KEY": "sk-test",
            "BUILDER2_VIDEO_DURATION_SECONDS": "1",
        },
        clear=False,
    )
    @patch("engine.runway_video.fetch_video_plan_o3")
    @patch("engine.runway_video.resolve_video_product_name", return_value=("user", "Product"))
    @patch("engine.runway_video._create_text_to_video_task")
    @patch("engine.runway_video._create_image_to_video_task")
    def test_invalid_duration_fails_before_runway(
        self,
        image_task_mock,
        text_task_mock,
        _product_name,
        plan_mock,
        _redis_name,
    ) -> None:
        from engine.runway_video import _generate_one_video_mvp_body

        with self.assertRaises(RunwayVideoMVPError) as ctx:
            _generate_one_video_mvp_body("Product", "A useful product.", job_id="job-dur")
        self.assertEqual(ctx.exception.args[0], "builder2_invalid_video_duration")
        plan_mock.assert_not_called()
        image_task_mock.assert_not_called()
        text_task_mock.assert_not_called()


class TestBuilder2VideoDurationStartImage(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_start_image_prompt_uses_dynamic_duration(self) -> None:
        plan: Dict[str, Any] = {
            "productNameResolved": "ACE Widget",
            "coreVisualIdea": "maximum connection",
            "openingFrameDescription": "friends hugging warmly",
            "sceneVariations": ["young couple hugging"],
            "videoPrompt": "Montage of hugs. No text.",
        }
        prompt = build_ace_start_frame_image_prompt(plan)
        self.assertIn("7-second", prompt)
        self.assertIn("over 7 seconds", prompt)
        self.assertNotIn("5-second", prompt.lower())


class TestBuilder2VideoDurationHeadlineOverlay(unittest.TestCase):
    def test_overlay_timing_uses_ffprobe_not_five_second_assumption(self) -> None:
        source = inspect.getsource(postprocess_video_headline)
        self.assertIn("_ffprobe_duration_seconds", source)
        self.assertIn("duration_sec - overlay_s", source)
        self.assertNotIn("duration_sec == 5", source)
        self.assertNotIn("duration_sec = 5", source)


if __name__ == "__main__":
    unittest.main()
