"""
Builder1 product visibility policy tests (CREATIVE_DECISION default).

Run: python -m unittest tests.test_builder1_product_visibility -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder1_creative_methodology import deterministic_methodology_checks
from engine.builder1_final_stages import parse_brand_physical_output, parse_series_ads_output
from engine.builder1_image_compliance import (
    IMAGE_COMPLIANCE_VIOLATION_CODES,
    finalize_compliance_result,
    parse_image_compliance_response,
)
from engine.builder1_image_generator import VISIBILITY_VIOLATION_CODES
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    ProductVisibilitySource,
    VisualExecutionRoute,
    apply_creative_visibility_fields,
    build_policy_aware_global_image_constraints,
    build_product_visibility_image_block,
    derive_product_visibility_policy,
    enforce_series_ad_visibility_fields,
    explicit_product_visibility_forbidden,
    explicit_product_visibility_requested,
    infer_visual_execution_route,
    plan_approves_product_as_main_visual,
    plan_approves_product_as_physical_generator,
    plan_approves_product_visibility,
    policy_is_legacy_secondary_only,
    policy_prohibits_product_depiction,
    policy_requires_product_depiction,
    resolve_product_visibility_policy,
    visual_route_for_plan,
)
from engine.builder1_planning_metrics import NORMAL_PLANNING_CALLS_WITH_NAME
from engine.builder1_visual_prompt import build_visual_prompt
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _brand_physical, _series_ads

BRIEF = "Reinforced shell product for daily carry"


def forbidden_test_plan() -> Builder1SeriesPlan:
    data = copy.deepcopy(_base_campaign(2))
    data["productVisibilityPolicy"] = "FORBIDDEN"
    return _parse(data, 2)


def _plan(*, policy: str = "CREATIVE_DECISION", route: str = "ANALOGY_LED") -> Builder1SeriesPlan:
    data = copy.deepcopy(_base_campaign(2))
    data["productVisibilityPolicy"] = policy
    plan = _parse(data, 2)
    plan.planning_internals = dict(plan.planning_internals or {})
    plan.planning_internals["visualExecutionRoute"] = route
    plan.planning_internals["productEvidenceRequired"] = route == "PRODUCT_INTEGRATED_ANALOGY"
    if route == "PRODUCT_LED":
        plan.planning_internals["adInternals"] = {
            1: {
                "productVisible": True,
                "productIsMainVisual": True,
                "productIsPhysicalGenerator": True,
            }
        }
    elif route == "PRODUCT_INTEGRATED_ANALOGY":
        plan.planning_internals["adInternals"] = {1: {"productVisible": True}}
    return plan


class TestPolicyDerivation(unittest.TestCase):
    def test_default_policy_is_creative_decision(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=BRIEF,
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.CREATIVE_DECISION)
        self.assertEqual(decision.source, ProductVisibilitySource.DEFAULT)

    def test_resolve_empty_defaults_to_creative_decision(self) -> None:
        self.assertEqual(
            resolve_product_visibility_policy(None),
            ProductVisibilityPolicy.CREATIVE_DECISION,
        )

    def test_resolve_honors_legacy_forbidden(self) -> None:
        self.assertEqual(
            resolve_product_visibility_policy("FORBIDDEN"),
            ProductVisibilityPolicy.FORBIDDEN,
        )

    def test_explicit_show_request_requires_product_visibility(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=f"{BRIEF}. Please show the product in the ad.",
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED)
        self.assertEqual(decision.source, ProductVisibilitySource.EXPLICIT_USER_REQUEST)

    def test_explicit_hide_request_forbids_product(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=f"{BRIEF}. Do not show the product in the ad.",
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.FORBIDDEN)
        self.assertEqual(decision.source, ProductVisibilitySource.EXPLICIT_USER_REQUEST)

    def test_hide_wins_when_both_show_and_hide_present(self) -> None:
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=f"{BRIEF}. Show the product. Do not show the product.",
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.FORBIDDEN)

    def test_resolve_honors_legacy_secondary_exception(self) -> None:
        self.assertEqual(
            resolve_product_visibility_policy("SECONDARY_EXPLICIT_EXCEPTION"),
            ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION,
        )
        self.assertTrue(policy_is_legacy_secondary_only(ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION))

    def test_explicit_hide_detection(self) -> None:
        self.assertTrue(
            explicit_product_visibility_forbidden(
                product_name="",
                product_description=f"{BRIEF}. Product must not be visible.",
            )
        )

    def test_product_description_alone_does_not_allow_visibility(self) -> None:
        self.assertFalse(
            explicit_product_visibility_requested(
                product_name="",
                product_description=BRIEF,
            )
        )

    def test_product_name_alone_does_not_allow_visibility(self) -> None:
        self.assertFalse(
            explicit_product_visibility_requested(
                product_name="CarryShell",
                product_description=BRIEF,
            )
        )


class TestVisualExecutionRoutes(unittest.TestCase):
    def test_analogy_led_is_default_route(self) -> None:
        self.assertEqual(
            infer_visual_execution_route(),
            VisualExecutionRoute.ANALOGY_LED,
        )

    def test_product_led_when_generator_is_product(self) -> None:
        self.assertEqual(
            infer_visual_execution_route(physical_generator_is_product=True),
            VisualExecutionRoute.PRODUCT_LED,
        )

    def test_integrated_when_evidence_required(self) -> None:
        self.assertEqual(
            infer_visual_execution_route(product_evidence_required=True),
            VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY,
        )

    def test_visual_route_for_plan_reads_internals(self) -> None:
        plan = _plan(route="PRODUCT_LED")
        self.assertEqual(visual_route_for_plan(plan), VisualExecutionRoute.PRODUCT_LED)


class TestSeriesVisibilityEnforcement(unittest.TestCase):
    def test_forbidden_forces_all_visibility_false(self) -> None:
        ads = enforce_series_ad_visibility_fields(
            [{"productVisible": True, "productIsMainVisual": True}],
            policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertFalse(ads[0]["productVisible"])
        self.assertFalse(ads[0]["productIsMainVisual"])

    def test_creative_product_led_sets_visibility_true(self) -> None:
        bp = _brand_physical()
        bp["physicalGeneratorIsProduct"] = True
        bp["campaignRationale"] = "Product form demonstrates reinforced shell"
        ads = apply_creative_visibility_fields([{}], brand_physical=bp)
        self.assertTrue(ads[0]["productVisible"])
        self.assertTrue(ads[0]["productIsMainVisual"])
        self.assertTrue(ads[0]["productIsPhysicalGenerator"])

    def test_creative_analogy_led_hides_product_by_default(self) -> None:
        ads = apply_creative_visibility_fields([{}], brand_physical=_brand_physical())
        self.assertFalse(ads[0]["productVisible"])

    def test_required_analogy_led_still_shows_product(self) -> None:
        ads = enforce_series_ad_visibility_fields(
            [{}],
            policy=ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
            brand_physical=_brand_physical(),
        )
        self.assertTrue(ads[0]["productVisible"])
        self.assertFalse(ads[0]["productIsMainVisual"])

    def test_required_product_led_sets_main_visual(self) -> None:
        bp = _brand_physical()
        bp["physicalGeneratorIsProduct"] = True
        bp["campaignRationale"] = "Shell geometry demonstrates reinforced durability"
        ads = enforce_series_ad_visibility_fields(
            [{}],
            policy=ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
            brand_physical=bp,
        )
        self.assertTrue(ads[0]["productVisible"])
        self.assertTrue(ads[0]["productIsMainVisual"])
        self.assertTrue(ads[0]["productIsPhysicalGenerator"])

    def test_legacy_secondary_forces_secondary_only(self) -> None:
        ads = enforce_series_ad_visibility_fields(
            [{"productIsMainVisual": True}],
            policy=ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION,
        )
        self.assertTrue(ads[0]["productVisible"])
        self.assertFalse(ads[0]["productIsMainVisual"])

    def test_model_cannot_enable_visibility_under_forbidden(self) -> None:
        payload = _series_ads(2)
        payload["ads"][0]["productVisibilityRequired"] = True
        result = parse_series_ads_output(
            payload,
            expected_ad_count=2,
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertFalse(result.ads[0]["productVisible"])


class TestBrandPhysicalInvariants(unittest.TestCase):
    def test_forbidden_rejects_product_as_generator(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload, visibility_policy="FORBIDDEN")
        self.assertIn("physical_generator_is_product", str(ctx.exception))

    def test_creative_allows_product_with_rationale(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        payload["campaignRationale"] = "Product shell geometry is the proof"
        result = parse_brand_physical_output(payload, visibility_policy="CREATIVE_DECISION")
        self.assertTrue(result.physical_generator_is_product)

    def test_creative_rejects_product_without_mechanism(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        payload["campaignRationale"] = ""
        payload["physicalGeneratorCampaignRole"] = ""
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload, visibility_policy="CREATIVE_DECISION")
        self.assertIn("product_led_missing_creative_mechanism", str(ctx.exception))

    def test_packaging_always_rejected(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsPackaging"] = True
        with self.assertRaises(Exception) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_generator_is_packaging", str(ctx.exception))


class TestVisualPrompts(unittest.TestCase):
    def test_forbidden_prompt_excludes_product(self) -> None:
        plan = _plan(policy="FORBIDDEN")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("ADVERTISED PRODUCT: not depicted", prompt)

    def test_creative_analogy_prompt_uses_transferred_object(self) -> None:
        plan = _plan(policy="CREATIVE_DECISION", route="ANALOGY_LED")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn(plan.transferred_object, prompt)
        self.assertIn("ANALOGY-LED", prompt)

    def test_product_led_prompt_uses_product_description(self) -> None:
        plan = _plan(policy="PRODUCT_VISIBILITY_REQUIRED", route="PRODUCT_LED")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("PRODUCT-LED", prompt)
        self.assertIn(plan.product_description, prompt)

    def test_required_analogy_does_not_mandate_transferred_object_only(self) -> None:
        plan = _plan(policy="PRODUCT_VISIBILITY_REQUIRED", route="ANALOGY_LED")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("must appear in the image", prompt)
        self.assertIn(plan.transferred_object, prompt)

    def test_legacy_secondary_keeps_transferred_main_visual(self) -> None:
        plan = _plan(policy="SECONDARY_EXPLICIT_EXCEPTION", route="ANALOGY_LED")
        prompt = build_visual_prompt(plan, plan.ads[0])
        self.assertIn("small secondary unbranded element", prompt)
        self.assertIn(plan.transferred_object, prompt)


class TestPlanApprovalHelpers(unittest.TestCase):
    def test_plan_approves_visibility_for_product_led(self) -> None:
        plan = _plan(policy="PRODUCT_VISIBILITY_REQUIRED", route="PRODUCT_LED")
        self.assertTrue(plan_approves_product_visibility(plan))
        self.assertTrue(plan_approves_product_as_main_visual(plan))
        self.assertTrue(plan_approves_product_as_physical_generator(plan))

    def test_required_policy_always_approves_visibility(self) -> None:
        plan = _plan(policy="PRODUCT_VISIBILITY_REQUIRED", route="ANALOGY_LED")
        self.assertTrue(policy_requires_product_depiction(ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED))
        self.assertTrue(plan_approves_product_visibility(plan))

    def test_plan_does_not_approve_for_analogy_led(self) -> None:
        plan = _plan(route="ANALOGY_LED")
        self.assertFalse(plan_approves_product_visibility(plan))


class TestComplianceAdjudication(unittest.TestCase):
    def test_forbidden_rejects_unauthorized_product(self) -> None:
        plan = _plan(policy="FORBIDDEN")
        result = parse_image_compliance_response(
            {
                "pass": False,
                "violations": ["product_visible_without_explicit_request"],
                "confidence": "high",
            },
            series_plan=plan,
        )
        self.assertFalse(result.passed)

    def test_creative_without_plan_approval_fails_on_high_confidence(self) -> None:
        plan = _plan(policy="CREATIVE_DECISION", route="ANALOGY_LED")
        result = finalize_compliance_result(
            reviewer_pass=False,
            candidate_violations=["product_visible_without_explicit_request"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=plan,
            ad_index=1,
        )
        self.assertFalse(result.passed)

    def test_creative_with_plan_approval_allows_product(self) -> None:
        plan = _plan(policy="CREATIVE_DECISION", route="PRODUCT_LED")
        result = finalize_compliance_result(
            reviewer_pass=True,
            candidate_violations=["product_visible_without_explicit_request"],
            evidence_items=[],
            overall_confidence="high",
            series_plan=plan,
            ad_index=1,
        )
        self.assertTrue(result.passed)

    def test_violation_codes_registered(self) -> None:
        for code in (
            "product_visible_without_explicit_request",
            "packaging_visible_without_explicit_request",
            "product_used_as_physical_generator",
            "product_used_as_main_visual",
        ):
            self.assertIn(code, IMAGE_COMPLIANCE_VIOLATION_CODES)
            self.assertIn(code, VISIBILITY_VIOLATION_CODES)


class TestImageBlocksAndConstraints(unittest.TestCase):
    def test_forbidden_image_block_prohibits_product(self) -> None:
        block = build_product_visibility_image_block(
            policy=ProductVisibilityPolicy.FORBIDDEN,
            transferred_object="Rubber ball family",
            transferred_object_action="Bounces",
            product_name="CarryShell",
        )
        self.assertIn("FORBIDDEN", block)
        self.assertIn("Do not depict the advertised product", block)

    def test_product_led_global_constraints(self) -> None:
        block = build_policy_aware_global_image_constraints(
            policy=ProductVisibilityPolicy.CREATIVE_DECISION,
            visual_route=VisualExecutionRoute.PRODUCT_LED,
        )
        self.assertIn("PRODUCT-LED", block)

    def test_secondary_exception_block_legacy_only(self) -> None:
        block = build_product_visibility_image_block(
            policy=ProductVisibilityPolicy.SECONDARY_EXPLICIT_EXCEPTION,
            transferred_object="Rubber ball family",
            transferred_object_action="Bounces",
            product_name="CarryShell",
        )
        self.assertIn("LEGACY SECONDARY", block)

    def test_required_show_allows_product_led_block(self) -> None:
        block = build_product_visibility_image_block(
            policy=ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
            transferred_object="Rubber ball family",
            transferred_object_action="Bounces",
            product_name="CarryShell",
            visual_route=VisualExecutionRoute.PRODUCT_LED,
        )
        self.assertIn("PRODUCT-LED", block)


class TestMethodologyIntegration(unittest.TestCase):
    def test_product_led_without_mechanism_rejected(self) -> None:
        plan_dict: Dict[str, Any] = copy.deepcopy(_base_campaign(2))
        plan_dict["productVisibilityPolicy"] = "CREATIVE_DECISION"
        plan_dict["planningInternals"] = {
            "adInternals": {
                "1": {
                    "productIsMainVisual": True,
                    "productIsPhysicalGenerator": True,
                }
            }
        }
        reasons = deterministic_methodology_checks(plan_dict)
        self.assertIn("product_led_without_creative_mechanism", reasons)

    def test_forbidden_still_blocks_unauthorized_visibility(self) -> None:
        plan_dict = copy.deepcopy(_base_campaign(2))
        plan_dict["productVisibilityPolicy"] = "FORBIDDEN"
        plan_dict["ads"][0]["productVisible"] = True
        reasons = deterministic_methodology_checks(plan_dict)
        self.assertIn("unauthorized_product_visibility", reasons)


class TestCallCountUnchanged(unittest.TestCase):
    def test_supplied_name_planning_call_count_constant(self) -> None:
        self.assertEqual(NORMAL_PLANNING_CALLS_WITH_NAME, 5)


class TestPolicyHelpers(unittest.TestCase):
    def test_policy_prohibits_only_forbidden(self) -> None:
        self.assertTrue(policy_prohibits_product_depiction(ProductVisibilityPolicy.FORBIDDEN))
        self.assertFalse(policy_prohibits_product_depiction(ProductVisibilityPolicy.CREATIVE_DECISION))


if __name__ == "__main__":
    unittest.main()
