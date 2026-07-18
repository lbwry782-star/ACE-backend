"""
Builder1 planning profile and latency routing tests.

Run: python -m unittest tests.test_builder1_planning_routing -v
"""
from __future__ import annotations

import copy
import inspect
import os
import unittest
from typing import Any, List
from unittest.mock import patch

from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planning_profile import (
    PlanningProfile,
    execution_optimization_active,
    model_supports_reasoning_effort,
    resolve_planning_profile,
    resolve_stage_model,
    resolve_stage_reasoning_effort,
)
from engine.builder1_planner import plan_builder1
from tests.test_builder1_staged_planning import _full_final_responses

BRIEF = "Reinforced shell product for daily carry"


class TestPlanningProfiles(unittest.TestCase):
    def setUp(self) -> None:
        self._env = patch.dict(
            os.environ,
            {
                "BUILDER1_PLANNING_MODEL": "o3-pro",
                "BUILDER1_EXECUTION_MODEL": "gpt-4.1",
                "BUILDER1_PLANNING_PROFILE": "BALANCED",
            },
            clear=False,
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()

    def test_quality_profile_uses_quality_model(self) -> None:
        with patch.dict(os.environ, {"BUILDER1_PLANNING_PROFILE": "QUALITY"}, clear=False):
            self.assertEqual(resolve_stage_model("strategy_stage"), "o3-pro")
            self.assertEqual(resolve_stage_model("graphic_system"), "o3-pro")

    def test_balanced_profile_routes_execution_stages(self) -> None:
        self.assertEqual(resolve_stage_model("strategy_stage"), "o3-pro")
        self.assertEqual(resolve_stage_model("product_name_resolution"), "gpt-4.1")
        self.assertEqual(resolve_stage_model("graphic_system"), "gpt-4.1")
        self.assertEqual(resolve_stage_model("series_ads"), "gpt-4.1")

    def test_fast_profile_is_opt_in(self) -> None:
        self.assertEqual(resolve_planning_profile(), PlanningProfile.BALANCED)
        with patch.dict(os.environ, {"BUILDER1_PLANNING_PROFILE": "FAST"}, clear=False):
            self.assertEqual(resolve_planning_profile(), PlanningProfile.FAST)
            self.assertEqual(resolve_stage_model("slogan_stage"), "gpt-4.1")

    def test_unsupported_execution_model_falls_back_when_unset(self) -> None:
        with patch.dict(os.environ, {"BUILDER1_EXECUTION_MODEL": ""}, clear=False):
            self.assertFalse(execution_optimization_active())
            self.assertEqual(resolve_stage_model("series_ads"), "o3-pro")

    def test_unsupported_reasoning_effort_is_not_sent_for_non_reasoning_model(self) -> None:
        self.assertIsNone(resolve_stage_reasoning_effort("graphic_system", "gpt-4.1"))

    def test_reasoning_effort_supported_for_o3(self) -> None:
        self.assertTrue(model_supports_reasoning_effort("o3-pro"))
        self.assertEqual(resolve_stage_reasoning_effort("strategy_stage", "o3-pro"), "low")


class TestStageOrderAndCallCounts(unittest.TestCase):
    def test_stage_order_unchanged(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module.run_builder1_campaign_pipeline)
        ordered_tokens = [
            "run_strategy_stage",
            "run_slogan_stage",
            "run_conceptual_stage",
            '_run_stage(\n        "brand_physical"',
            '_run_stage(\n        "graphic_system"',
            "_run_series_stage_with_integrity",
        ]
        indices = [source.index(token) for token in ordered_tokens]
        self.assertEqual(indices, sorted(indices))

    def test_supplied_name_planning_six_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(
            len(
                [
                    s
                    for s in stages
                    if s
                    in {
                        "strategy_stage",
                        "slogan_stage",
                        "conceptual_stage",
                        "brand_physical",
                        "graphic_system",
                        "series_ads",
                    }
                ]
            ),
            NORMAL_PLANNING_CALLS_WITH_NAME,
        )

    def test_blank_name_planning_seven_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(
            len(
                [
                    s
                    for s in stages
                    if s
                    in {
                        "product_name_resolution",
                        "strategy_stage",
                        "slogan_stage",
                        "conceptual_stage",
                        "brand_physical",
                        "graphic_system",
                        "series_ads",
                    }
                ]
            ),
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
        )


class TestProductionShapedRegression(unittest.TestCase):
    def test_production_shaped_balanced_flow(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BUILDER1_PLANNING_PROFILE": "BALANCED",
                "BUILDER1_PLANNING_MODEL": "o3-pro",
                "BUILDER1_EXECUTION_MODEL": "gpt-4.1",
            },
            clear=False,
        ):
            stages: List[str] = []
            models: dict[str, str] = {}

            def model_caller(system: str, user: str, stage: str | None = None) -> object:
                if stage:
                    stages.append(stage)
                    models[stage] = resolve_stage_model(stage)
                return copy.deepcopy(_full_final_responses(2).get(system, {}))

            plan = plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
            self.assertEqual(plan.product_visibility_policy, "FORBIDDEN")
            self.assertTrue(plan.transferred_object)
            self.assertEqual(stages[:6], [
                "strategy_stage",
                "slogan_stage",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
            ])
            self.assertEqual(models["strategy_stage"], "o3-pro")
            self.assertEqual(models["series_ads"], "gpt-4.1")
            prompt = __import__(
                "engine.builder1_visual_prompt", fromlist=["build_visual_prompt"]
            ).build_visual_prompt(plan, plan.ads[0])
            self.assertIn("ADVERTISED PRODUCT: not depicted", prompt)
            self.assertIn("PACKAGING: not depicted", prompt)
            self.assertIn(plan.transferred_object, prompt)


if __name__ == "__main__":
    unittest.main()
