"""
Builder2 reasoning model routing — isolated from Builder1 and global OpenAI vars.

Run: python -m unittest tests.test_builder2_reasoning_config -v
"""
from __future__ import annotations

import ast
import inspect
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.builder1_planning_profile import quality_model, resolve_stage_model
from engine.builder2_reasoning_config import (
    DEFAULT_BUILDER2_REASONING_EFFORT,
    DEFAULT_BUILDER2_REASONING_MODE,
    DEFAULT_BUILDER2_REASONING_MODEL,
    build_builder2_reasoning_payload,
    log_builder2_model_selected,
    resolve_builder2_reasoning_effort,
    resolve_builder2_reasoning_mode,
    resolve_builder2_reasoning_model,
)
from engine.openai_reasoning import normalize_legacy_text_model
from engine import runway_video, video_headline, video_planning

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"


class TestBuilder2ReasoningDefaults(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_model_defaults_to_gpt_5_6_sol(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_model(), "gpt-5.6-sol")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_mode_defaults_to_standard(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_mode(), "standard")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_effort_defaults_to_medium(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_effort(), "medium")

    @patch.dict(os.environ, {}, clear=True)
    def test_reasoning_payload_defaults(self) -> None:
        self.assertEqual(
            build_builder2_reasoning_payload(),
            {"mode": "standard", "effort": "medium"},
        )

    @patch.dict(os.environ, {"BUILDER2_REASONING_MODEL": "o3-pro"}, clear=True)
    def test_o3_pro_legacy_maps_to_gpt_5_6_sol(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_model(), "gpt-5.6-sol")

    @patch.dict(os.environ, {"BUILDER2_REASONING_EFFORT": "bogus"}, clear=True)
    def test_invalid_effort_falls_back_to_medium(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_effort(), "medium")

    @patch.dict(os.environ, {"BUILDER2_REASONING_MODE": "bogus"}, clear=True)
    def test_invalid_mode_falls_back_to_standard(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_mode(), "standard")


class TestBuilder2IsolationFromGlobalAndBuilder1(unittest.TestCase):
    GLOBAL_BUILDER1_POLLUTION = {
        "OPENAI_TEXT_MODEL": "o4-mini",
        "OPENAI_REASONING_EFFORT": "low",
        "OPENAI_REASONING_MODE": "pro",
        "VIDEO_PLANNER_MODEL": "o3-pro",
        "VIDEO_PLANNER_REASONING_EFFORT": "high",
        "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
        "BUILDER1_PLANNING_PROFILE": "QUALITY",
        "BUILDER2_REASONING_MODEL": "",
        "BUILDER2_REASONING_MODE": "",
        "BUILDER2_REASONING_EFFORT": "",
    }

    @patch.dict(os.environ, GLOBAL_BUILDER1_POLLUTION, clear=True)
    def test_builder2_does_not_read_openai_text_model(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_model(), "gpt-5.6-sol")

    @patch.dict(os.environ, GLOBAL_BUILDER1_POLLUTION, clear=True)
    def test_builder2_does_not_read_openai_reasoning_effort(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_effort(), "medium")

    @patch.dict(os.environ, GLOBAL_BUILDER1_POLLUTION, clear=True)
    def test_builder2_does_not_read_video_planner_vars(self) -> None:
        self.assertEqual(video_planning._text_model(), "gpt-5.6-sol")
        self.assertEqual(
            video_planning._video_plan_reasoning_payload(),
            {"mode": "standard", "effort": "medium"},
        )

    @patch.dict(
        os.environ,
        {
            **GLOBAL_BUILDER1_POLLUTION,
            "BUILDER2_REASONING_MODEL": "gpt-5.6-sol",
            "BUILDER2_REASONING_EFFORT": "medium",
        },
        clear=True,
    )
    def test_builder2_env_vars_take_precedence(self) -> None:
        self.assertEqual(resolve_builder2_reasoning_model(), "gpt-5.6-sol")
        self.assertEqual(resolve_builder2_reasoning_effort(), "medium")

    @patch.dict(
        os.environ,
        {
            "BUILDER2_REASONING_MODEL": "gpt-5.6-sol",
            "BUILDER2_REASONING_EFFORT": "high",
            "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
            "BUILDER1_PLANNING_PROFILE": "QUALITY",
        },
        clear=True,
    )
    def test_builder1_unchanged_when_builder2_env_changes(self) -> None:
        self.assertEqual(quality_model(), "gpt-5.6-sol")
        self.assertEqual(resolve_stage_model("strategy_stage"), "gpt-5.6-sol")
        self.assertEqual(resolve_builder2_reasoning_effort(), "high")


class TestBuilder2ActiveRoutes(unittest.TestCase):
    BUILDER2_PRODUCTION_ENV = {
        "BUILDER2_REASONING_MODEL": "gpt-5.6-sol",
        "BUILDER2_REASONING_MODE": "standard",
        "BUILDER2_REASONING_EFFORT": "medium",
    }

    @patch.dict(os.environ, BUILDER2_PRODUCTION_ENV, clear=True)
    def test_video_planning_model_route(self) -> None:
        self.assertEqual(video_planning._text_model(), "gpt-5.6-sol")

    @patch.dict(os.environ, BUILDER2_PRODUCTION_ENV, clear=True)
    def test_video_planning_reasoning_route(self) -> None:
        self.assertEqual(
            video_planning._video_plan_reasoning_payload(),
            {"mode": "standard", "effort": "medium"},
        )

    @patch.dict(os.environ, BUILDER2_PRODUCTION_ENV, clear=True)
    def test_video_planning_does_not_fallback_to_o3_pro(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_TEXT_MODEL": "o3-pro", "BUILDER2_REASONING_MODEL": ""},
            clear=True,
        ):
            self.assertEqual(video_planning._text_model(), "gpt-5.6-sol")
            self.assertNotEqual(video_planning._text_model(), "o3-pro")

    @patch.dict(os.environ, BUILDER2_PRODUCTION_ENV, clear=True)
    def test_legacy_o3_pro_env_maps_to_gpt_5_6_sol_not_pro_03(self) -> None:
        with patch.dict(os.environ, {"BUILDER2_REASONING_MODEL": "o3-pro"}, clear=True):
            self.assertEqual(resolve_builder2_reasoning_model(), "gpt-5.6-sol")
            self.assertEqual(normalize_legacy_text_model("o3-pro"), "gpt-5.6-sol")


class TestBuilder2CallSites(unittest.TestCase):
    def test_builder2_responses_create_call_count_unchanged(self) -> None:
        expected = {
            "video_planning.py": 1,
            "video_headline.py": 1,
            "runway_video.py": 1,
        }
        for filename, count in expected.items():
            source = (ENGINE / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            calls = 0
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "create":
                    value = func.value
                    if (
                        isinstance(value, ast.Attribute)
                        and value.attr == "responses"
                    ):
                        calls += 1
            with self.subTest(filename=filename):
                self.assertEqual(calls, count)

    def test_builder2_modules_use_builder2_config_not_global_resolver(self) -> None:
        for module in (video_planning, video_headline, runway_video):
            source = inspect.getsource(module)
            with self.subTest(module=module.__name__):
                self.assertNotIn("resolve_openai_reasoning_model", source)
                self.assertIn("builder2_reasoning_config", source)

    def test_video_planning_tournament_call_order_unchanged(self) -> None:
        source = inspect.getsource(video_planning._fetch_video_plan_o3_sync)
        normal_idx = source.index('base_call_type="normal"')
        repair_idx = source.index('base_call_type="repair"')
        fallback_idx = source.index("_build_keyword_scene_fallback_plan")
        self.assertLess(normal_idx, repair_idx)
        self.assertLess(repair_idx, fallback_idx)

    def test_runway_video_model_selection_unchanged(self) -> None:
        source = (ENGINE / "runway_video.py").read_text(encoding="utf-8")
        self.assertIn("RUNWAY_VIDEO_MODEL_SELECTED", source)
        self.assertNotIn("BUILDER2_REASONING_MODEL", source.split("def _runway")[0] if "def _runway" in source else source[:5000])


class TestBuilder2Logging(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "BUILDER2_REASONING_MODEL": "gpt-5.6-sol",
            "BUILDER2_REASONING_MODE": "standard",
            "BUILDER2_REASONING_EFFORT": "medium",
        },
        clear=True,
    )
    @patch("engine.builder2_reasoning_config.logger")
    def test_log_builder2_model_selected_fields(self, mock_logger) -> None:
        log_builder2_model_selected(role="video_planning", call_type="repair", attempt=2)
        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        self.assertIn("BUILDER2_MODEL_SELECTED", message)
        self.assertIn("role=video_planning", message)
        self.assertIn("model=gpt-5.6-sol", message)
        self.assertIn("reasoningMode=standard", message)
        self.assertIn("reasoningEffort=medium", message)
        self.assertIn("callType=repair", message)
        self.assertIn("attempt=2", message)


class TestBuilder2ReasoningRoles(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER2_REASONING_EFFORT": "medium"}, clear=True)
    def test_normal_repair_retry_use_medium_effort(self) -> None:
        for call_type in ("normal", "repair", "retry"):
            with self.subTest(call_type=call_type):
                self.assertEqual(
                    build_builder2_reasoning_payload()["effort"],
                    "medium",
                )

    @patch.dict(os.environ, {}, clear=True)
    def test_builder2_ignores_global_low_effort_when_unset(self) -> None:
        with patch.dict(os.environ, {"OPENAI_REASONING_EFFORT": "low"}, clear=False):
            self.assertEqual(resolve_builder2_reasoning_effort(), "medium")


if __name__ == "__main__":
    unittest.main()
