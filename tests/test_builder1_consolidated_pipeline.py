"""
Builder1 consolidated planning pipeline tests.

Run: python -m unittest tests.test_builder1_consolidated_pipeline -v
"""
from __future__ import annotations

import copy
import inspect
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_campaign_integrity import validate_builder1_campaign_integrity
from engine.builder1_planning_contract import (
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_PRODUCT_NAME_RESOLUTION_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planner import plan_builder1
from tests.test_builder1_staged_planning import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    _full_final_responses,
)

BRIEF = "Reinforced shell product for daily carry"

PLANNING_STAGE_SET = {
    "product_name_resolution",
    "strategy_slogan_stage",
    "conceptual_stage",
    "brand_physical",
    "graphic_system",
    "series_ads",
}


class TestConsolidatedProductionFlow(unittest.TestCase):
    def test_blank_name_uses_seven_planning_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan = plan_builder1(
            product_name="",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(plan.ad_count, 2)
        self.assertIn("product_name_resolution", stages)
        self.assertIn("strategy_slogan_stage", stages)
        self.assertIn("conceptual_stage", stages)
        self.assertEqual(stages.count("strategy_slogan_stage"), 1)
        self.assertEqual(stages.count("slogan_stage"), 0)
        self.assertEqual(stages.count("strategy_stage"), 0)
        self.assertEqual(len([s for s in stages if s in PLANNING_STAGE_SET]), NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)

    def test_supplied_name_uses_five_planning_calls(self) -> None:
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
        self.assertNotIn("product_name_resolution", stages)
        self.assertEqual(len([s for s in stages if s in PLANNING_STAGE_SET]), NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_no_final_judge_in_active_pipeline(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module)
        self.assertNotIn("judge_builder1_strategy", source)
        self.assertNotIn("run_strategy_stage", source)
        self.assertNotIn("run_slogan_stage", source)

    def test_integrity_validation_is_deterministic(self) -> None:
        source = inspect.getsource(validate_builder1_campaign_integrity)
        self.assertNotIn("model_caller", source)
        self.assertNotIn("client.responses.create", source)


class TestBuilderBoundary(unittest.TestCase):
    def test_pipeline_has_no_creator_or_judge_roles(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module).lower()
        self.assertNotIn("creator report", source)
        self.assertNotIn("tournament manager", source)
        self.assertNotIn("judge_builder1_strategy", source)


if __name__ == "__main__":
    unittest.main()
