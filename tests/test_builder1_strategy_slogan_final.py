"""
Builder1 single-path strategy/slogan final-result tests.

Run: python -m unittest tests.test_builder1_strategy_slogan_final -v
"""
from __future__ import annotations

import copy
import os
import unittest
from typing import Any, List
from unittest.mock import patch

from engine.builder1_consolidated_stages import process_strategy_slogan_stage_response
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_NAME,
    get_planning_metrics,
)
from engine.builder1_planning_model import STRATEGY_SLOGAN_STAGE_JSON_SCHEMA, prepare_strict_json_schema
from engine.builder1_planner import _run_stage, plan_builder1
from engine.builder1_strategy_slogan_final import (
    FINAL_SLOGAN_ID,
    FINAL_STRATEGY_ID,
    normalize_stored_strategy_slogan_payload,
    parse_slogan_final_section,
    parse_strategy_final_section,
)
from tests.test_builder1_staged_planning import (
    _full_final_responses,
    _legacy_strategy_slogan_candidate_payload,
    _strategy_slogan_stage_payload,
)

BRIEF = "Reinforced shell product for daily carry"


class TestFinalStageSchema(unittest.TestCase):
    def test_schema_requires_single_strategy_and_slogan(self) -> None:
        prepared = prepare_strict_json_schema(STRATEGY_SLOGAN_STAGE_JSON_SCHEMA)
        strategy_props = prepared["properties"]["strategy"]["properties"]
        slogan_props = prepared["properties"]["slogan"]["properties"]
        self.assertIn("strategicProblem", strategy_props)
        self.assertIn("relativeAdvantage", strategy_props)
        self.assertIn("brandSlogan", slogan_props)
        self.assertNotIn("candidates", strategy_props)
        self.assertNotIn("candidates", slogan_props)
        self.assertNotIn("selectedCandidateId", strategy_props)
        self.assertNotIn("selectedCandidateId", slogan_props)

    def test_valid_payload_parses_final_path(self) -> None:
        payload = _strategy_slogan_stage_payload()
        strategy_selection, strategy = parse_strategy_final_section(
            payload["strategy"],
            product_description=BRIEF,
        )
        slogan_selection, slogan = parse_slogan_final_section(
            payload["slogan"],
            relative_advantage=strategy.relative_advantage,
            product_description=BRIEF,
            detected_language="en",
        )
        self.assertEqual(strategy_selection.selected_candidate_id, FINAL_STRATEGY_ID)
        self.assertEqual(slogan_selection.selected_candidate_id, FINAL_SLOGAN_ID)
        self.assertTrue(strategy.strategic_problem)
        self.assertTrue(slogan.brand_slogan)

    def test_candidate_arrays_rejected_for_new_payloads(self) -> None:
        payload = _legacy_strategy_slogan_candidate_payload()
        with self.assertRaises(Exception):
            parse_strategy_final_section(payload["strategy"], product_description=BRIEF)


class TestNoStrategyCandidateRepair(unittest.TestCase):
    def test_valid_first_stage_proceeds_without_repair(self) -> None:
        stages: List[str] = []
        captured: dict[str, Any] = {}
        import engine.builder1_planning_metrics as metrics_module

        real_reset = metrics_module.reset_planning_metrics

        def capture_reset(token) -> None:
            metrics = get_planning_metrics()
            if metrics is not None:
                captured["metrics"] = metrics
            real_reset(token)

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        with patch("engine.builder1_planner.reset_planning_metrics", side_effect=capture_reset):
            plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertNotIn("strategy_candidate_repair", stages)
        self.assertNotIn("slogan_only_repair", stages)
        metrics = captured.get("metrics")
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics.strategy_candidate_repair_calls, 0)
        self.assertEqual(metrics.slogan_only_repair_calls, 0)
        self.assertEqual(
            len([s for s in stages if s in {
                "strategy_slogan_stage",
                "conceptual_stage",
                "brand_physical",
                "graphic_system",
                "series_ads",
            }]),
            NORMAL_PLANNING_CALLS_WITH_NAME,
        )

    def test_invalid_json_can_use_bounded_stage_repair(self) -> None:
        calls: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                calls.append(stage)
            if stage == "strategy_slogan_stage" and calls.count("strategy_slogan_stage") == 1:
                return {"strategy": {}, "slogan": {}}
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description=BRIEF,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        self.assertEqual(calls.count("strategy_slogan_stage"), 2)
        self.assertNotIn("strategy_candidate_repair", calls)


class TestBackwardCompatibility(unittest.TestCase):
    def test_legacy_candidate_payload_normalizes_for_read(self) -> None:
        payload = _legacy_strategy_slogan_candidate_payload()
        normalized = normalize_stored_strategy_slogan_payload(payload)
        self.assertNotIn("candidates", normalized["strategy"])
        self.assertNotIn("candidates", normalized["slogan"])
        strategy_selection, strategy = parse_strategy_final_section(
            normalized["strategy"],
            product_description=BRIEF,
            allow_legacy_candidates=True,
        )
        self.assertEqual(strategy.id, FINAL_STRATEGY_ID)
        self.assertTrue(strategy_selection.selection_reason)

    def test_new_payload_does_not_store_candidate_arrays(self) -> None:
        result = process_strategy_slogan_stage_response(
            _strategy_slogan_stage_payload(),
            product_name="TestBrand",
            product_name_resolved="TestBrand",
            product_description=BRIEF,
            detected_language="en",
            model_caller=lambda *_a, **_k: {},
            run_stage=_run_stage,
        )
        self.assertEqual(len(result[2]), 1)
        self.assertEqual(len(result[6]), 1)


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_module_still_present(self) -> None:
        import engine.builder2_zip as builder2_module

        self.assertTrue(hasattr(builder2_module, "build_builder2_video_zip_bytes"))


if __name__ == "__main__":
    unittest.main()
