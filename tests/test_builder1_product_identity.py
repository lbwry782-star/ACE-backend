"""
Builder1 product identity guard, failure classification, and physical repair tests.

Run: python -m unittest tests.test_builder1_product_identity -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, List

from engine.builder1_campaign_store import clear_memory_store_for_tests, create_campaign_session
from engine.builder1_failure_classification import (
    Builder1FailureAction,
    Builder1FailureClass,
    PlanProductVisibilityConflictError,
    classify_compliance_failure,
    validate_forbidden_plan_visibility,
)
from engine.builder1_image_generator import generate_builder1_ad_image
from engine.builder1_image_prompt_preflight import run_image_prompt_preflight, validate_forbidden_visual_prompt_text
from engine.builder1_physical_repair import repair_builder1_campaign_from_physical
from engine.builder1_product_identity_guard import (
    COPY_AND_TYPOGRAPHY_FIELD_KEYS,
    VISUAL_SUBJECT_AD_FIELDS,
    detect_ad_visual_subject_identity_conflicts,
    detect_product_identity_conflicts,
)
from engine.builder1_product_visibility import ProductVisibilityPolicy
from engine.builder1_visual_prompt import build_visual_prompt
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
        self.assertTrue(any("physical_generator_matches_advertised_product" in reason for reason in reasons))

    def test_false_boolean_cannot_hide_shoe_identity(self) -> None:
        payload = _brand_physical()
        payload["physicalGenerator"] = "running shoe"
        payload["transferredObject"] = "running shoe"
        payload["physicalGeneratorIsProduct"] = False
        payload["physicalGeneratorIsAdvertisedProduct"] = False
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output = __import__(
                "engine.builder1_final_stages", fromlist=["parse_brand_physical_output"]
            ).parse_brand_physical_output
            parse_brand_physical_output(
                payload,
                product_description="Lightweight running shoe for daily training",
                product_name_resolved="Stride",
                visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
            )
        self.assertIn("physical_generator_matches_advertised_product", str(ctx.exception))

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


class TestTypographyFalsePositives(unittest.TestCase):
    def _hebrew_plan(self):
        plan = _parse(_base_campaign(2), 2)
        plan.product_name_resolved = "צעד צעד"
        plan.product_description = "נעלי ריצה קלות לשימוש יומיומי"
        plan.physical_generator = "Compact folding staircase"
        plan.transferred_object = "Compact folding staircase"
        plan.transferred_object_action = "Folds into one compact step"
        plan.product_visibility_policy = "FORBIDDEN"
        plan.ads[0].visual_execution = "Display 'צעד צעד' as plain typography at bottom left"
        plan.ads[0].physical_execution = "Staircase variant showing compact fold"
        plan.ads[0].scene_description = "External staircase carries the visual idea"
        plan.ads[0].headline = "כל דרך מתחילה בצעד"
        internals = dict(plan.planning_internals or {})
        ad_internals = dict(internals.get("adInternals") or {})
        ad_internals[1] = {
            **(ad_internals.get(1) or {}),
            "sloganConnection": "Execution supports the brand slogan through the transferred staircase",
            "productVisible": False,
            "productIsPhysicalGenerator": False,
            "productIsMainVisual": False,
        }
        internals["adInternals"] = ad_internals
        plan.planning_internals = internals
        return plan

    def test_product_name_in_typography_passes_integrity(self) -> None:
        plan = self._hebrew_plan()
        reasons = validate_forbidden_plan_visibility(plan)
        self.assertFalse(any("product_name" in reason for reason in reasons))
        self.assertNotIn("ad_execution_product_name_match", reasons)

    def test_product_name_in_marketing_text_does_not_fail(self) -> None:
        plan = self._hebrew_plan()
        plan.ads[0].marketing_text = (plan.ads[0].marketing_text + " צעד צעד").strip()
        reasons = validate_forbidden_plan_visibility(plan)
        self.assertNotIn("ad_execution_product_name_match", reasons)

    def test_ordinary_word_product_name_in_explanatory_copy_passes(self) -> None:
        plan = self._hebrew_plan()
        plan.product_name_resolved = "קרוב"
        plan.ads[0].visual_execution = "Explain how the execution supports the fixed slogan"
        reasons = detect_ad_visual_subject_identity_conflicts(
            ad=plan.ads[0],
            product_description=plan.product_description,
        )
        self.assertEqual(reasons, [])

    def test_copy_fields_are_excluded_from_visual_scan(self) -> None:
        self.assertIn("sloganConnection", COPY_AND_TYPOGRAPHY_FIELD_KEYS)
        self.assertIn("headline", COPY_AND_TYPOGRAPHY_FIELD_KEYS)
        self.assertNotIn("physical_execution", COPY_AND_TYPOGRAPHY_FIELD_KEYS)

    def test_visual_fields_are_inspected(self) -> None:
        self.assertIn("physical_execution", VISUAL_SUBJECT_AD_FIELDS)
        self.assertIn("scene_description", VISUAL_SUBJECT_AD_FIELDS)

    def test_production_shaped_campaign_reaches_image_prompt(self) -> None:
        plan = self._hebrew_plan()
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("צעד צעד", prompt)
        self.assertIn("ADVERTISED PRODUCT: not depicted", prompt)
        result = validate_forbidden_visual_prompt_text(prompt, series_plan=plan)
        self.assertTrue(result.ok)


class TestRealVisualConflicts(unittest.TestCase):
    def test_product_category_in_scene_description_fails(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.product_description = "Lightweight running shoe for daily training"
        plan.physical_generator = "Rubber ball family"
        plan.transferred_object = "Rubber ball family"
        plan.ads[0].scene_description = "A running shoe dominates the frame"
        reasons = validate_forbidden_plan_visibility(plan)
        self.assertTrue(any(r.startswith("main_visual_matches_advertised_product") for r in reasons))

    def test_product_in_physical_generator_fails(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.product_description = "Lightweight running shoe for daily training"
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        reasons = validate_forbidden_plan_visibility(plan)
        self.assertTrue(any("physical_generator_matches_advertised_product" in r for r in reasons))


class TestFailureClassification(unittest.TestCase):
    def test_accidental_product_visible_is_image_execution(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        failure_class, action, _, _evidence = classify_compliance_failure(
            violations=["product_visible_without_explicit_request"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.IMAGE_EXECUTION)
        self.assertEqual(action, Builder1FailureAction.REGENERATE_IMAGE)

    def test_product_used_as_generator_without_structured_conflict_is_image_execution(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        failure_class, action, _, evidence = classify_compliance_failure(
            violations=["product_used_as_physical_generator"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.IMAGE_EXECUTION)
        self.assertEqual(action, Builder1FailureAction.REGENERATE_IMAGE)
        self.assertFalse(evidence["structuredPlanConflict"])

    def test_structured_product_as_generator_is_plan_contradiction(self) -> None:
        plan = _parse(_base_campaign(2), 2)
        plan.physical_generator = "running shoe"
        plan.transferred_object = "running shoe"
        plan.product_description = "Lightweight running shoe for daily training"
        failure_class, action, _, evidence = classify_compliance_failure(
            violations=["product_used_as_physical_generator"],
            series_plan=plan,
        )
        self.assertEqual(failure_class, Builder1FailureClass.PLAN_CONTRADICTION)
        self.assertEqual(action, Builder1FailureAction.REPAIR_FROM_PHYSICAL)
        self.assertTrue(evidence["structuredPlanConflict"])

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


class TestProductionShapedRegression(unittest.TestCase):
    def test_generated_name_typography_only_passes_visibility_integrity(self) -> None:
        plan = TestTypographyFalsePositives()._hebrew_plan()
        visibility_reasons = validate_forbidden_plan_visibility(plan)
        self.assertNotIn("ad_execution_product_name_match", visibility_reasons)
        self.assertFalse(any("product_name" in r for r in visibility_reasons))

    def test_negative_main_visual_conflict(self) -> None:
        plan = TestTypographyFalsePositives()._hebrew_plan()
        plan.product_description = "Lightweight running shoe for daily training"
        plan.ads[0].scene_description = "Hero running shoe in the center"
        reasons = validate_forbidden_plan_visibility(plan)
        self.assertTrue(any(r.startswith("main_visual_matches_advertised_product") for r in reasons))


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
        self.assertEqual(stages, ["brand_physical", "graphic_system", "series_ads"])


class TestImageExecutionVsPlanContradiction(unittest.TestCase):
    def test_valid_plan_regenerates_image_only(self) -> None:
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            from engine.builder1_compliance_adjudication import ComplianceEvidenceItem
            from engine.builder1_compliance_product_grounding import ComplianceProductMatch
            from engine.builder1_image_compliance import ImageComplianceResult, finalize_compliance_result

            plan = _parse(_base_campaign(2), 2)
            if calls["gen"] == 1:
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

        plan = _parse(_base_campaign(2), 2)
        generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 2)

    def test_pixel_product_generator_is_advisory_without_structured_conflict(self) -> None:
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        def reviewer(**_kwargs: Any):
            from engine.builder1_image_compliance import ImageComplianceResult

            return ImageComplianceResult(
                passed=False,
                violations=["product_used_as_physical_generator"],
                raw_violations=["product_used_as_physical_generator"],
                confidence="high",
            )

        plan = _parse(_base_campaign(2), 2)
        generate_builder1_ad_image(plan, 1, caller, compliance_reviewer=reviewer)
        self.assertEqual(calls["gen"], 1)

    def test_structured_plan_conflict_preflight_does_not_generate(self) -> None:
        calls = {"gen": 0}

        def caller(_prompt: str, _fmt: str) -> bytes:
            calls["gen"] += 1
            return b"img"

        plan = _parse(_base_campaign(2), 2)
        internals = dict(plan.planning_internals or {})
        ad_internals = dict(internals.get("adInternals") or {})
        ad_internals[1] = {**(ad_internals.get(1) or {}), "productIsPhysicalGenerator": True}
        internals["adInternals"] = ad_internals
        plan.planning_internals = internals

        with self.assertRaises(PlanProductVisibilityConflictError):
            generate_builder1_ad_image(plan, 1, caller)
        self.assertEqual(calls["gen"], 0)


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


if __name__ == "__main__":
    unittest.main()
