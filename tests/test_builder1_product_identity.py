"""
Builder1 product identity guard, failure classification, and physical repair tests.

Run: python -m unittest tests.test_builder1_product_identity -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, List
from unittest.mock import patch

from engine.builder1_campaign_store import clear_memory_store_for_tests, create_campaign_session
from engine.builder1_failure_classification import (
    Builder1FailureAction,
    Builder1FailureClass,
    PlanProductVisibilityConflictError,
    classify_compliance_failure,
    validate_forbidden_plan_visibility,
)
from engine.builder1_final_stages import BrandPhysicalOutput, parse_brand_physical_output
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_image_prompt_preflight import run_image_prompt_preflight
from engine.builder1_physical_repair import repair_builder1_campaign_from_physical
from engine.builder1_product_identity_guard import detect_product_identity_conflicts
from engine.builder1_product_visibility import ProductVisibilityPolicy
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _brand_physical, _full_final_responses


class TestProductIdentityGuard(unittest.TestCase):
    def test_shoe_product_as_generator_is_rejected(self) -> None:
        reasons = detect_product_identity_conflicts(
            product_name="Stride",
            product_description="Lightweight running shoe for daily training",
            physical_generator="running shoe",
            transferred_object="running shoe",
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertTrue(any("shoe" in reason or "running shoe" in reason for reason in reasons))

    def test_false_boolean_cannot_hide_shoe_identity(self) -> None:
        payload = _brand_physical()
        payload["physicalGenerator"] = "running shoe"
        payload["transferredObject"] = "running shoe"
        payload["physicalGeneratorIsProduct"] = False
        payload["physicalGeneratorIsAdvertisedProduct"] = False
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(
                payload,
                product_description="Lightweight running shoe for daily training",
                product_name_resolved="Stride",
                visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
            )
        self.assertIn("physical_generator_product_identity", str(ctx.exception))

    def test_external_rubber_ball_passes(self) -> None:
        reasons = detect_product_identity_conflicts(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            physical_generator="Rubber ball family",
            transferred_object="Rubber ball family",
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertEqual(reasons, [])

    def test_broad_world_object_not_over_rejected(self) -> None:
        reasons = detect_product_identity_conflicts(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            physical_generator="Steel anvil",
            transferred_object="Steel anvil",
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertEqual(reasons, [])


class TestFailureClassification(unittest.TestCase):
    def test_accidental_product_visible_is_image_execution(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        failure_class, action, _ = classify_compliance_failure(
            violations=["product_visible_without_explicit_request"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.IMAGE_EXECUTION)
        self.assertEqual(action, Builder1FailureAction.REGENERATE_IMAGE)

    def test_product_used_as_generator_is_plan_contradiction(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        failure_class, action, _ = classify_compliance_failure(
            violations=["product_used_as_physical_generator"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.PLAN_CONTRADICTION)
        self.assertEqual(action, Builder1FailureAction.REPAIR_FROM_PHYSICAL)

    def test_contradictory_plan_blocks_preflight(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        plan.product_description = "Lightweight running shoe for daily training"
        with self.assertRaises(PlanProductVisibilityConflictError):
            run_image_prompt_preflight(
                series_plan=plan,
                ad_plan=plan.ads[0],
                prompt="MAIN VISUAL: running shoe",
            )


class TestContradictoryFixtureTrace(unittest.TestCase):
    """Demonstrates the production contradiction path in a deterministic fixture."""

    def test_contradictory_shoe_plan_fails_integrity_and_preflight(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.product_description = "Lightweight running shoe for daily training"
        plan.product_name_resolved = "Stride"
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        plan.transferred_object_action = "Flexes on landing"
        reasons = validate_forbidden_plan_visibility(plan)
        self.assertTrue(any("physical_generator_product_identity" in r for r in reasons))
        with self.assertRaises(PlanProductVisibilityConflictError):
            run_image_prompt_preflight(
                series_plan=plan,
                ad_plan=plan.ads[0],
                prompt="MAIN VISUAL: running shoe",
            )


class TestPhysicalRepairPreservesUpstream(unittest.TestCase):
    def setUp(self) -> None:
        clear_memory_store_for_tests()

    def tearDown(self) -> None:
        clear_memory_store_for_tests()

    def test_repair_reruns_physical_graphic_series_only(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.product_description = "Lightweight running shoe for daily training"
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        original_slogan = plan.brand_slogan
        original_advantage = plan.relative_advantage
        original_concept = plan.conceptual_generator
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        repaired = repair_builder1_campaign_from_physical(plan, model_caller=model_caller)
        self.assertEqual(repaired.brand_slogan, original_slogan)
        self.assertEqual(repaired.relative_advantage, original_advantage)
        self.assertEqual(repaired.conceptual_generator, original_concept)
        self.assertNotEqual(repaired.transferred_object.lower(), "running shoe")
        self.assertEqual(
            stages,
            ["brand_physical", "graphic_system", "series_ads"],
        )
        self.assertNotIn("strategy_stage", stages)
        self.assertNotIn("slogan_stage", stages)
        self.assertNotIn("conceptual_stage", stages)


class TestImageExecutionVsPlanContradiction(unittest.TestCase):
    def test_valid_plan_regenerates_image_only(self) -> None:
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            from engine.builder1_image_compliance import ImageComplianceResult

            if calls["gen"] == 1:
                return ImageComplianceResult(
                    passed=False,
                    violations=["product_visible_without_explicit_request"],
                    confidence="high",
                )
            return ImageComplianceResult(passed=True, violations=[], confidence="high")

        plan = _parse(_base_campaign(2), 2)
        generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 2)

    def test_plan_contradiction_does_not_regenerate_twice(self) -> None:
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            from engine.builder1_image_compliance import ImageComplianceResult

            return ImageComplianceResult(
                passed=False,
                violations=["product_used_as_physical_generator"],
                confidence="high",
            )

        plan = _parse(_base_campaign(2), 2)
        from engine.builder1_failure_classification import PlanContradictionComplianceError

        with self.assertRaises(PlanContradictionComplianceError):
            generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 1)


class TestPublicRetryContract(unittest.TestCase):
    def test_image_only_retry_response_fields(self) -> None:
        from app import _builder1_image_compliance_error_response

        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="public-image", plan=plan, target_ad_count=2)
        from engine.builder1_campaign_store import get_campaign_session

        session = get_campaign_session("public-image")
        out = _builder1_image_compliance_error_response(
            campaign_id="public-image",
            ad_index=1,
            session=session,
            error_code="image_compliance_failed",
            retry_mode="image_only",
        )
        self.assertEqual(out["retryMode"], "image_only")
        self.assertNotIn("prompt", out)

    def test_physical_repair_response_fields(self) -> None:
        from app import _builder1_retry_error_response, BUILDER1_PUBLIC_PHYSICAL_REPAIR_MESSAGE

        plan = _parse(_base_campaign(2), 2)
        create_campaign_session(campaign_id="public-repair", plan=plan, target_ad_count=2)
        from engine.builder1_campaign_store import get_campaign_session, mark_physical_repair_required

        session = mark_physical_repair_required("public-repair", failed_ad_index=1, violations=["product_used_as_physical_generator"])
        out = _builder1_retry_error_response(
            campaign_id="public-repair",
            ad_index=1,
            session=session,
            error_code="physical_plan_conflict",
            retry_mode="repair_from_physical",
            user_message=BUILDER1_PUBLIC_PHYSICAL_REPAIR_MESSAGE,
        )
        self.assertEqual(out["retryMode"], "repair_from_physical")
        self.assertEqual(out["preservedThroughStage"], "conceptual_stage")


if __name__ == "__main__":
    unittest.main()
