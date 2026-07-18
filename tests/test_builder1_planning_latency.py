"""
Builder1 planning latency and bounded retry tests.

Run: python -m unittest tests.test_builder1_planning_latency -v
"""
from __future__ import annotations

import copy
import logging
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_creative_methodology import (
    deterministic_methodology_checks,
    earliest_methodology_repair_stage,
    is_foundational_strategic_rejection,
)
from engine.builder1_planning_model import (
    STAGE_JSON_SCHEMAS,
    STRICT_SCHEMA_STAGES,
)
from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_staged_parsers import (
    StageParseError,
    StrategyCandidateReview,
    parse_strategy_selection,
)
from engine.builder1_strategy_judge import BUILDER1_STRATEGY_JUDGE_SYSTEM_PROMPT
from engine.builder1_strategy_scan import (
    ensure_strategy_scan_from_raw,
    parse_strategy_candidate_replacements,
)
from engine.builder1_strategy_selection import (
    StrategySelectionExhausted,
    run_strategy_selection_with_gate,
    validate_selected_strategy_gate,
)
from engine.builder1_strategy_judge import StrategyJudgeResult
from engine.builder1_planning_contract import STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM
from tests.test_builder1_staged_planning import (
    STAGE_STRATEGY_SCAN_SYSTEM,
    STAGE_STRATEGY_SELECT_SYSTEM,
    _brand_physical,
    _early_stage_responses,
    _full_final_responses,
    _graphic,
    _series_ads,
    _strategy_scan_payload,
    _strategy_selection_payload,
)

BRIEF = "Reinforced shell product for daily carry"


class TestStrategyCandidateRepairSchema(unittest.TestCase):
    def test_strategy_candidate_repair_has_dedicated_schema(self) -> None:
        self.assertIn("strategy_candidate_repair", STRICT_SCHEMA_STAGES)
        schema = STAGE_JSON_SCHEMAS["strategy_candidate_repair"]
        self.assertEqual(schema["required"], ["replacements"])

    def test_replacements_array_required(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_strategy_candidate_replacements({}, allowed_ids=["S01"])
        self.assertIn("strategy_candidate_repair_missing_replacements", ctx.exception.reasons)


class TestStrategyRepairBoundedBehavior(unittest.TestCase):
    def test_malformed_repair_retries_without_full_rescan(self) -> None:
        payload = _strategy_scan_payload()
        payload["candidates"][0]["simpleStrategicAction"] = "Needs workshop"
        calls: List[str] = []

        def caller(system: str, user: str, **kwargs: Any) -> object:
            calls.append(system)
            if len(calls) == 1:
                return {"unexpected": "shape"}
            fixed = copy.deepcopy(_strategy_scan_payload()["candidates"][0])
            fixed["simpleStrategicAction"] = None
            return {"replacements": [fixed]}

        result = ensure_strategy_scan_from_raw(
            payload,
            product_name="TestBrand",
            product_description=BRIEF,
            model_caller=caller,
        )
        self.assertEqual(len(result), 12)
        self.assertEqual(calls.count(STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM), 2)

    def test_repair_failure_does_not_imply_full_scan(self) -> None:
        from engine.builder1_planner import _run_strategy_scan_stage

        broken_scan = _strategy_scan_payload()
        broken_scan["candidates"][0]["simpleStrategicAction"] = "Needs workshop"
        scan_calls = 0

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            nonlocal scan_calls
            if stage == "strategy_scan" or system == STAGE_STRATEGY_SCAN_SYSTEM:
                scan_calls += 1
                return broken_scan
            return {"unexpected": "shape"}

        with self.assertRaises(Builder1PlannerError) as ctx:
            _run_strategy_scan_stage(
                model_caller,
                product_name="TestBrand",
                product_description=BRIEF,
                detected_language="en",
                lens_order=["economic"],
                exploration_seed="seed",
            )
        self.assertIn("strategy_candidate_repair_failed", str(ctx.exception))
        self.assertEqual(scan_calls, 1)


class TestStrategySelectionReview(unittest.TestCase):
    def test_material_investment_candidate_ineligible_in_review(self) -> None:
        payload = _strategy_selection_payload(
            selected_id="S02",
            candidate_ids=["S01", "S02"],
        )
        payload["candidateReviews"][0]["eligible"] = True
        payload["candidateReviews"][1]["eligible"] = False
        payload["candidateReviews"][1]["requiresMaterialInvestment"] = True
        payload["candidateReviews"][1]["rejectionCodes"] = ["material_client_investment_required"]
        payload["selectedCandidateId"] = "S01"
        with self.assertRaises(StageParseError):
            parse_strategy_selection(
                payload,
                [],
                eligible_ids={"S01", "S02"},
            )

    def test_selected_strategy_gate_runs_before_slogan_in_pipeline_source(self) -> None:
        from engine.builder1_planning_pipeline import run_builder1_campaign_pipeline
        import inspect

        source = inspect.getsource(run_builder1_campaign_pipeline)
        gate_idx = source.index("run_strategy_selection_with_gate")
        slogan_idx = source.index('"slogan_scan"')
        self.assertLess(gate_idx, slogan_idx)


class TestFinalJudgeRouting(unittest.TestCase):
    def test_slogan_only_failure_routes_to_slogan_not_restart(self) -> None:
        codes = ["slogan_not_derived_from_advantage"]
        self.assertFalse(is_foundational_strategic_rejection(codes))
        self.assertEqual(earliest_methodology_repair_stage(codes), "slogan_scan")

    def test_mixed_failure_routes_to_earliest_stage(self) -> None:
        codes = [
            "slogan_not_derived_from_advantage",
            "material_client_investment_required",
        ]
        self.assertTrue(is_foundational_strategic_rejection(codes))
        self.assertEqual(earliest_methodology_repair_stage(codes), "strategy_scan")

    def test_final_judge_no_lexical_overlap(self) -> None:
        plan = {
            "relativeAdvantage": "Survives daily drops",
            "brandSlogan": "Built To Last",
            "sloganDerivation": "Distills durability into a spoken challenge",
            "conceptualGeneratorAction": "Show impact survival",
            "physicalGenerator": "Reinforced shell",
            "graphicGenerator": {"device": "impact frame"},
            "ads": [{"index": 1, "sceneDescription": "Drop test", "headline": "", "marketingText": "x"}],
        }
        self.assertNotIn("slogan_not_derived_from_advantage", deterministic_methodology_checks(plan))


class TestProductionShapedLatencyRegression(unittest.TestCase):
    def test_production_shaped_flow_reaches_planning_ok(self) -> None:
        broken_scan = _strategy_scan_payload()
        broken_scan["candidates"][0]["briefSupport"] = "Surveys show 90% prefer reinforced shells"
        broken_scan["candidates"][6]["briefSupport"] = "Research shows 87% prefer reinforced shells"
        repair_calls = 0

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            nonlocal repair_calls
            if system == STAGE_STRATEGY_CANDIDATE_REPAIR_SYSTEM:
                repair_calls += 1
                if repair_calls == 1:
                    return {"replacements": []}
                return {
                    "replacements": [
                        {
                            **_strategy_scan_payload()["candidates"][0],
                            "briefSupport": "Follows from brief reinforced shell mention",
                        },
                        {
                            **_strategy_scan_payload()["candidates"][6],
                            "briefSupport": "Follows from brief reinforced shell mention",
                        },
                    ]
                }
            responses = _full_final_responses(2)
            responses[STAGE_STRATEGY_SCAN_SYSTEM] = broken_scan
            return copy.deepcopy(responses.get(system, {"pass": True, "rejectionReasonCodes": []}))

        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            return_value=StrategyJudgeResult(True, []),
        ):
            plan = plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertEqual(plan.ad_count, 2)
        self.assertEqual(repair_calls, 2)


class TestLoggingAndMetrics(unittest.TestCase):
    def test_stage_duration_log_emitted(self) -> None:
        with self.assertLogs("engine.builder1_planner", level="INFO") as captured:
            from engine.builder1_planner import _run_stage

            _run_stage(
                "product_name_resolution",
                lambda *_a, **_k: {"productNameResolved": "TestBrand"},
                "system",
                "user",
                lambda raw: str(raw["productNameResolved"]),
            )
        self.assertTrue(any("BUILDER1_STAGE_DURATION" in line for line in captured.output))

    def test_planning_call_summary_emitted(self) -> None:
        with patch(
            "engine.builder1_planning_pipeline.judge_builder1_strategy",
            return_value=StrategyJudgeResult(True, []),
        ):
            with self.assertLogs("engine.builder1_planning_metrics", level="INFO") as captured:
                plan_builder1(
                    product_name="CarryShell",
                    product_description=BRIEF,
                    format_value="portrait",
                    model_caller=lambda system, user, stage=None: _full_final_responses(2).get(
                        system,
                        {"pass": True, "rejectionReasonCodes": []},
                    ),
                    ad_count=2,
                    campaign_id="cid-latency",
                    job_id="job-latency",
                )
        self.assertTrue(any("BUILDER1_PLANNING_CALL_SUMMARY" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
