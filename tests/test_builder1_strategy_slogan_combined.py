"""
Builder1 combined strategy_slogan_stage tests.

Run: python -m unittest tests.test_builder1_strategy_slogan_combined -v
"""
from __future__ import annotations

import copy
import json
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
from engine.builder1_planning_profile import PlanningProfile, resolve_stage_model
from engine.builder1_planner import _run_stage, plan_builder1
from engine.builder1_staged_parsers import StageParseError
from tests.test_builder1_staged_planning import (
    _full_final_responses,
    _strategy_slogan_stage_payload,
    _strategy_stage_payload,
)

BRIEF = "Reinforced shell product for daily carry"


class TestCombinedStageContract(unittest.TestCase):
    def test_strict_schema_requires_strategy_and_slogan_objects(self) -> None:
        prepared = prepare_strict_json_schema(STRATEGY_SLOGAN_STAGE_JSON_SCHEMA)
        self.assertEqual(set(prepared["required"]), {"strategy", "slogan"})
        self.assertIn("candidates", prepared["properties"]["strategy"]["properties"])
        self.assertIn("candidates", prepared["properties"]["slogan"]["properties"])

    def test_parser_validates_strategy_before_slogan(self) -> None:
        payload = _strategy_slogan_stage_payload()
        del payload["strategy"]["evaluations"]
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

    def test_combined_output_has_separate_sections(self) -> None:
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
        self.assertEqual(result[1].id, "S01")
        self.assertEqual(result[5].id, "L01")


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
    @patch.dict("os.environ", {"BUILDER1_PLANNING_PROFILE": "QUALITY"}, clear=False)
    def test_quality_profile_uses_quality_model(self) -> None:
        self.assertEqual(resolve_stage_model("strategy_slogan_stage"), "o3-pro")


class TestCombinedMetrics(unittest.TestCase):
    def test_strategy_candidate_repair_counted_separately(self) -> None:
        metrics = Builder1PlanningMetrics()
        token = set_planning_metrics(metrics)
        try:
            metrics.record_model_call("strategy_slogan_stage")
            metrics.record_model_call("strategy_candidate_repair")
            metrics.record_model_call("slogan_only_repair")
            metrics.record_stage_retry("strategy_slogan_stage")
            self.assertEqual(metrics.strategy_slogan_stage_calls, 1)
            self.assertEqual(metrics.strategy_candidate_repair_calls, 1)
            self.assertEqual(metrics.slogan_only_repair_calls, 1)
            self.assertEqual(metrics.stage_retry_calls, 1)
            self.assertEqual(metrics.total_planning_model_calls, 3)
        finally:
            from engine.builder1_planning_metrics import reset_planning_metrics

            reset_planning_metrics(token)

    def test_exceptional_calls_exceed_normal_expected(self) -> None:
        metrics = Builder1PlanningMetrics()
        metrics.record_model_call("strategy_slogan_stage")
        metrics.record_model_call("strategy_candidate_repair")
        metrics.record_model_call("conceptual_stage")
        metrics.record_model_call("brand_physical")
        metrics.record_model_call("graphic_system")
        metrics.record_model_call("series_ads")
        self.assertEqual(metrics.total_planning_model_calls, 6)
        self.assertGreater(metrics.total_planning_model_calls, NORMAL_PLANNING_CALLS_WITH_NAME)


class TestCombinedRegression(unittest.TestCase):
    def test_candidate_counts_preserved(self) -> None:
        payload = _strategy_slogan_stage_payload()
        self.assertEqual(len(payload["strategy"]["candidates"]), 12)
        self.assertEqual(len(payload["slogan"]["candidates"]), 6)

    def test_conceptual_stage_remains_separate_call(self) -> None:
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
        self.assertEqual(stages.count("conceptual_stage"), 1)
        self.assertLess(stages.index("strategy_slogan_stage"), stages.index("conceptual_stage"))

    def test_comparison_harness_passes(self) -> None:
        import io
        from contextlib import redirect_stdout

        import scripts.compare_strategy_slogan_contract as compare_module

        with redirect_stdout(io.StringIO()):
            self.assertEqual(compare_module.main(), 0)


class TestSloganOnlyRepair(unittest.TestCase):
    def test_slogan_only_repair_preserves_frozen_strategy(self) -> None:
        calls: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            calls.append(stage or "")
            if stage == "slogan_only_repair":
                return _strategy_slogan_stage_payload()["slogan"]
            return _strategy_slogan_stage_payload()

        payload = _strategy_slogan_stage_payload()
        payload["slogan"]["selectedCandidateId"] = "L99"
        result = process_strategy_slogan_stage_response(
            payload,
            product_name="TestBrand",
            product_name_resolved="TestBrand",
            product_description=BRIEF,
            detected_language="en",
            model_caller=model_caller,
            run_stage=_run_stage,
        )
        self.assertIn("slogan_only_repair", calls)
        self.assertEqual(result[1].id, "S01")
        self.assertEqual(result[5].id, "L01")


if __name__ == "__main__":
    unittest.main()
