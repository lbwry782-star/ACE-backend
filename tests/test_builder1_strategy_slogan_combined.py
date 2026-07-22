"""
Builder1 combined strategy_slogan_stage tests.

Run: python -m unittest tests.test_builder1_strategy_slogan_combined -v
"""
from __future__ import annotations

import copy
import json
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_consolidated_stages import (
    process_strategy_slogan_stage_response,
    run_strategy_slogan_stage,
)
from engine.builder1_planning_contract import STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
    Builder1PlanningMetrics,
    set_planning_metrics,
)
from engine.builder1_planning_model import (
    STRATEGY_SLOGAN_STAGE_JSON_SCHEMA,
    prepare_strict_json_schema,
)
from engine.builder1_planning_profile import resolve_stage_model
from engine.builder1_planner import _run_stage, plan_builder1
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_strategy_slogan_final import FINAL_SLOGAN_ID, FINAL_STRATEGY_ID
from tests.test_builder1_staged_planning import (
    _full_final_responses,
    _strategy_slogan_stage_payload,
)

BRIEF = "Reinforced shell product for daily carry"


class TestCombinedStageContract(unittest.TestCase):
    def test_strict_schema_requires_strategy_and_slogan_objects(self) -> None:
        prepared = prepare_strict_json_schema(STRATEGY_SLOGAN_STAGE_JSON_SCHEMA)
        self.assertEqual(set(prepared["required"]), {"strategy", "slogan"})
        self.assertIn("strategicProblem", prepared["properties"]["strategy"]["properties"])
        self.assertIn("brandSlogan", prepared["properties"]["slogan"]["properties"])
        self.assertNotIn("candidates", prepared["properties"]["strategy"]["properties"])

    def test_parser_validates_strategy_before_slogan(self) -> None:
        payload = _strategy_slogan_stage_payload()
        del payload["strategy"]["relativeAdvantage"]
        with self.assertRaises(StageParseError) as ctx:
            process_strategy_slogan_stage_response(
                payload,
                product_name="TestBrand",
                product_name_resolved="TestBrand",
                product_description=BRIEF,
                detected_language="en",
                model_caller=lambda *_a, **_k: {},
                run_stage=_run_stage,
            )
        self.assertTrue(any("strategy:" in reason for reason in ctx.exception.reasons))

    def test_combined_output_has_single_final_path(self) -> None:
        result = process_strategy_slogan_stage_response(
            _strategy_slogan_stage_payload(),
            product_name="TestBrand",
            product_name_resolved="TestBrand",
            product_description=BRIEF,
            detected_language="en",
            model_caller=lambda *_a, **_k: {},
            run_stage=_run_stage,
        )
        self.assertEqual(len(result), 7)
        self.assertEqual(result[1].id, FINAL_STRATEGY_ID)
        self.assertEqual(result[5].id, FINAL_SLOGAN_ID)
        self.assertEqual(len(result[2]), 1)
        self.assertEqual(len(result[6]), 1)


class TestCombinedCallCounts(unittest.TestCase):
    def test_supplied_name_makes_five_planning_calls(self) -> None:
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
        self.assertEqual(stages.count("strategy_slogan_stage"), 1)
        self.assertEqual(stages.count("slogan_stage"), 0)
        self.assertEqual(stages.count("strategy_stage"), 0)
        self.assertNotIn("strategy_candidate_repair", stages)
        counted = len(
            [
                s
                for s in stages
                if s
                in {
                    "strategy_slogan_stage",
                    "conceptual_stage",
                    "brand_physical",
                    "graphic_system",
                    "series_ads",
                }
            ]
        )
        self.assertEqual(counted, NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_makes_six_planning_calls(self) -> None:
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
        self.assertIn("product_name_resolution", stages)
        self.assertEqual(stages.count("strategy_slogan_stage"), 1)
        counted = len(
            [
                s
                for s in stages
                if s
                in {
                    "product_name_resolution",
                    "strategy_slogan_stage",
                    "conceptual_stage",
                    "brand_physical",
                    "graphic_system",
                    "series_ads",
                }
            ]
        )
        self.assertEqual(counted, NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)


class TestCombinedModelRouting(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "BUILDER1_PLANNING_PROFILE": "QUALITY",
            "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
            "OPENAI_REASONING_EFFORT": "low",
        },
        clear=False,
    )
    def test_quality_profile_uses_sol_for_all_stages(self) -> None:
        self.assertEqual(resolve_stage_model("strategy_slogan_stage"), "gpt-5.6-sol")
        self.assertEqual(resolve_stage_model("conceptual_stage"), "gpt-5.6-sol")


class TestCombinedMetrics(unittest.TestCase):
    def test_strategy_candidate_repair_metric_stays_zero_for_new_jobs(self) -> None:
        metrics = Builder1PlanningMetrics()
        token = set_planning_metrics(metrics)
        try:
            metrics.record_model_call("strategy_slogan_stage")
            metrics.record_model_call("conceptual_stage")
            self.assertEqual(metrics.strategy_slogan_stage_calls, 1)
            self.assertEqual(metrics.strategy_candidate_repair_calls, 0)
            self.assertEqual(metrics.slogan_only_repair_calls, 0)
        finally:
            from engine.builder1_planning_metrics import reset_planning_metrics

            reset_planning_metrics(token)


class TestCombinedRegression(unittest.TestCase):
    def test_single_path_payload_has_no_candidate_arrays(self) -> None:
        payload = _strategy_slogan_stage_payload()
        self.assertNotIn("candidates", payload["strategy"])
        self.assertNotIn("candidates", payload["slogan"])
        self.assertNotIn("selectedCandidateId", payload["strategy"])
        self.assertNotIn("selectedCandidateId", payload["slogan"])

    def test_stage_order_unchanged(self) -> None:
        import inspect
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module.run_builder1_campaign_pipeline)
        self.assertLess(source.index("run_strategy_slogan_stage"), source.index("run_conceptual_stage"))

    def test_compare_contract_script_uses_final_payload(self) -> None:
        import scripts.compare_strategy_slogan_contract as compare_module

        self.assertTrue(hasattr(compare_module, "main"))


if __name__ == "__main__":
    unittest.main()
