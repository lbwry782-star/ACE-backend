"""
Builder1 production fix regression tests (routing, metrics, retry, preflight).

Run: python -m unittest tests.test_builder1_production_fixes -v
"""
from __future__ import annotations

import copy
import io
import logging
import os
import unittest
from typing import Any, List
from unittest.mock import patch

from engine.builder1_campaign_store import (
    clear_memory_store_for_tests,
    create_campaign_session,
    get_campaign_session,
    mark_image_retry_required,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_failure_classification import PlanProductVisibilityConflictError
from engine.builder1_image_prompt_preflight import run_image_prompt_preflight
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
    get_planning_metrics,
)
from engine.builder1_planning_profile import (
    execution_optimization_active,
    log_builder1_planning_profile_config,
    resolve_stage_model,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    build_no_product_strict_correction,
    derive_product_visibility_policy,
)
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_staged_planning import _full_final_responses

BRIEF = "Reinforced shell product for daily carry"


class TestBalancedRoutingAndConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._env = patch.dict(
            os.environ,
            {
                "BUILDER1_PLANNING_PROFILE": "BALANCED",
                "BUILDER1_QUALITY_MODEL": "gpt-5.6-sol",
                "BUILDER1_EXECUTION_MODEL": "gpt-4.1",
            },
            clear=False,
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()

    def test_core_stages_keep_quality_model(self) -> None:
        for stage in ("strategy_slogan_stage", "conceptual_stage", "brand_physical"):
            self.assertEqual(resolve_stage_model(stage), "gpt-5.6-sol")

    def test_missing_execution_model_logs_inactive_optimization(self) -> None:
        with patch.dict(os.environ, {"BUILDER1_EXECUTION_MODEL": ""}, clear=False):
            self.assertFalse(execution_optimization_active())
            self.assertEqual(resolve_stage_model("series_ads"), "gpt-5.6-sol")
            with self.assertLogs("engine.builder1_planning_profile", level="WARNING") as logs:
                import engine.builder1_planning_profile as profile_module

                profile_module._execution_model_warning_logged = False
                profile_module._config_logged = False
                log_builder1_planning_profile_config()
            joined = "\n".join(logs.output)
            self.assertIn("executionOptimizationActive=false", joined)

    def test_fast_profile_remains_opt_in(self) -> None:
        with patch.dict(os.environ, {"BUILDER1_PLANNING_PROFILE": "FAST"}, clear=False):
            self.assertEqual(resolve_stage_model("slogan_stage"), "gpt-4.1")


class TestPlanningCallMetrics(unittest.TestCase):
    def _run_and_capture_metrics(self, **kwargs: Any):
        captured: dict[str, Any] = {}
        real_reset = __import__(
            "engine.builder1_planning_metrics", fromlist=["reset_planning_metrics"]
        ).reset_planning_metrics

        def capture_reset(token) -> None:
            metrics = get_planning_metrics()
            if metrics is not None:
                captured["metrics"] = metrics
            real_reset(token)

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        with patch("engine.builder1_planner.reset_planning_metrics", side_effect=capture_reset):
            plan_builder1(model_caller=model_caller, format_value="portrait", ad_count=2, **kwargs)
        return captured.get("metrics")

    def test_supplied_name_total_planning_calls_six(self) -> None:
        metrics = self._run_and_capture_metrics(product_name="CarryShell", product_description=BRIEF)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertFalse(metrics.product_name_call_used)
        self.assertEqual(metrics.total_planning_model_calls, NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_total_planning_calls_seven(self) -> None:
        metrics = self._run_and_capture_metrics(product_name="", product_description=BRIEF)
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertTrue(metrics.product_name_call_used)
        self.assertEqual(metrics.product_name_stage_calls, 1)
        self.assertEqual(metrics.total_planning_model_calls, NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME)


class TestVisibilityTiming(unittest.TestCase):
    def test_visibility_policy_before_brand_physical(self) -> None:
        events: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage == "brand_physical":
                events.append("brand_physical")
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        with patch(
            "engine.builder1_product_visibility.log_builder1_product_visibility_policy",
            side_effect=lambda **kwargs: events.append("policy_logged"),
        ):
            plan_builder1(
                product_name="CarryShell",
                product_description=BRIEF,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertIn("policy_logged", events)
        self.assertLess(events.index("policy_logged"), events.index("brand_physical"))

    def test_default_policy_is_forbidden(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="",
            product_description=BRIEF,
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.FORBIDDEN)


class TestImagePreflightAndCorrection(unittest.TestCase):
    def test_preflight_blocks_missing_transferred_object(self) -> None:
        from tests.test_builder1_product_visibility import TestImagePromptVisibility

        plan = TestImagePromptVisibility()._plan()
        plan.transferred_object = ""
        plan.physical_generator = ""
        ad = plan.ads[0]
        with self.assertRaises(PlanProductVisibilityConflictError):
            run_image_prompt_preflight(
                series_plan=plan,
                ad_plan=ad,
                prompt=build_visual_prompt(plan, ad),
            )

    def test_no_product_strict_correction_profile(self) -> None:
        block = build_no_product_strict_correction(
            transferred_object="Rubber ball",
            transferred_object_action="Bounces upward",
        )
        self.assertIn("NO_PRODUCT_STRICT", block)
        self.assertIn("Rubber ball", block)

    def test_same_ad_correction_uses_no_product_strict(self) -> None:
        from tests.test_builder1_product_visibility import TestImagePromptVisibility

        prompts: List[str] = []

        def caller(prompt: str, _fmt: str) -> bytes:
            prompts.append(prompt)
            return b"img"

        def reviewer(**_kwargs: Any):
            from engine.builder1_compliance_adjudication import ComplianceEvidenceItem
            from engine.builder1_compliance_product_grounding import ComplianceProductMatch
            from engine.builder1_image_compliance import ImageComplianceResult, finalize_compliance_result

            if len(prompts) == 1:
                return finalize_compliance_result(
                    reviewer_pass=False,
                    candidate_violations=["product_visible_without_explicit_request"],
                    evidence_items=[
                        ComplianceEvidenceItem(
                            code="product_visible_without_explicit_request",
                            confidence="high",
                        )
                    ],
                    overall_confidence="high",
                    series_plan=plan,
                    product_match=ComplianceProductMatch(
                        advertised_product_present=True,
                        product_match_basis="explicit_product_shape",
                        matched_visual_element="TestBrand product unit",
                        relationship_to_advertised_product="actual_product",
                        product_match_explanation="Visible product unit matches advertised product.",
                    ),
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        plan = TestImagePromptVisibility()._plan()
        with self.assertLogs("engine.builder1_image_retry", level="INFO") as logs:
            generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertIn("NO_PRODUCT_STRICT", prompts[1])
        self.assertIn("GLOBAL IMAGE CONSTRAINTS", prompts[1])
        joined = "\n".join(logs.output)
        self.assertIn("BUILDER1_IMAGE_RETRY_CORRECTION", joined)


class TestCampaignPreservationOnImageFailure(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def tearDown(self) -> None:
        clear_memory_store_for_tests()

    def test_image_failure_preserves_campaign_plan(self) -> None:
        from tests.test_builder1_series import _base_campaign, _parse

        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="retry-cmp", plan=plan, target_ad_count=2)
        session = mark_image_retry_required(
            "retry-cmp",
            failed_ad_index=1,
            violations=["product_visible_without_explicit_request"],
        )
        self.assertEqual(session.status, "image_retry_required")
        self.assertEqual(session.generated_count, 0)
        self.assertEqual(session.failed_ad_index, 1)
        self.assertTrue(session.planning_complete)
        self.assertEqual(session.plan.brand_slogan, plan.brand_slogan)

    def test_retry_endpoint_contract_fields(self) -> None:
        from app import _builder1_image_compliance_error_response

        from tests.test_builder1_series import _base_campaign, _parse

        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="retry-api", plan=plan, target_ad_count=2)
        session = get_campaign_session("retry-api")
        out = _builder1_image_compliance_error_response(
            campaign_id="retry-api",
            ad_index=1,
            session=session,
            error_code="image_compliance_failed",
            violations=["product_visible_without_explicit_request"],
        )
        self.assertTrue(out["retryable"])
        self.assertTrue(out["planningComplete"])
        self.assertEqual(out["retryAdIndex"], 1)
        self.assertEqual(out["generatedCount"], 0)


class TestRetryWithoutReplanning(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def tearDown(self) -> None:
        clear_memory_store_for_tests()

    def test_retry_image_route_does_not_call_planner(self) -> None:
        from tests.test_builder1_series import _base_campaign, _parse

        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="retry-route", plan=plan, target_ad_count=2)
        mark_image_retry_required(
            "retry-route",
            failed_ad_index=1,
            violations=["product_visible_without_explicit_request"],
        )
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            with patch("app._builder1_executor") as mock_executor:
                mock_executor.submit.side_effect = lambda fn, *args, **kwargs: fn(*args, **kwargs)
                with patch("app._builder1_run_next_job") as mock_run:
                    from app import app

                    client = app.test_client()
                    resp = client.post(
                        "/api/builder1-retry-image",
                        json={"campaignId": "retry-route", "retryAdIndex": 1},
                    )
                    self.assertEqual(resp.status_code, 202)
                    mock_plan.assert_not_called()
                    mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
