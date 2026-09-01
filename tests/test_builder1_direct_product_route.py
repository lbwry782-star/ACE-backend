"""
Builder1 SIMPLE PRODUCT / DIRECT ADVANTAGE PRIORITY tests.

Run: python -m unittest tests.test_builder1_direct_product_route -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict

from engine.builder1_direct_product_route import (
    BUILDER1_SIMPLE_PRODUCT_DIRECT_ADVANTAGE_PRIORITY,
    AdditionalTranslationCost,
    RecommendedVisualRoute,
    direct_product_route_viable,
    parse_direct_product_route_assessment,
    resolve_visual_execution_route,
    validate_direct_product_route_consistency,
)
from engine.builder1_final_stages import parse_brand_physical_output
from engine.builder1_planning_contract import (
    BUILDER1_POPULAR_ANALOGY_FIRST,
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_SCAN_SYSTEM,
)
from engine.builder1_product_shot_methodology import (
    BUILDER1_PRODUCT_EVIDENCE_EXCEPTION,
    BUILDER1_PUBLIC_SIMPLICITY,
)
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    VisualExecutionRoute,
    infer_visual_execution_route,
    visual_route_for_plan,
)
from engine.builder1_staged_parsers import StageParseError
from tests.builder1_test_helpers import (
    direct_product_route_assessment,
    direct_product_route_assessment_integrated,
    direct_product_route_assessment_product_led,
)
from tests.test_builder1_series import _base_campaign, _parse
from tests.test_builder1_staged_planning import _brand_physical, _physical_candidates, _physical_evaluations


SHOE_BRIEF = "חנות נעליים. כל סוגי הנעליים כולל כפכפים וסנדלים."
SHOE_ADVANTAGE = "כל סוגי הנעליים במקום אחד במקום חיפוש מפוצל"


def _food_tray_analogy_payload() -> Dict[str, Any]:
    """Production-shaped unjustified food-tray analogy for a shoe store."""
    transferred = "מגש הגשה עם מחיצות נשלפת ומנות מזון שונות"
    payload = _brand_physical()
    payload.update(
        {
            "productNameResolved": "צעד צעד",
            "physicalGenerator": transferred,
            "transferredObject": transferred,
            "transferredObjectAction": "מסגרת המחיצות נשלפת ומשאירה קבוצות מזון שונות באותו מגש",
            "whyClearerThanShowingProduct": "מגש עם מחיצות נעלמות מראה ריכוז סוגים שונים",
            "campaignRationale": "Food tray partition removal parallels category breadth",
            "directProductRouteAssessment": direct_product_route_assessment(
                readable=True,
                advantage_direct=True,
                mechanism_available=True,
                mechanism_summary="Several distinct shoe types arranged together in one store display without cross-domain translation.",
                unique_gain=False,
                unique_gain_text="Different foods share one tray like different shoes in one store.",
                translation_cost="MEANINGFUL",
                recommended_route="ANALOGY_LED",
                route_reason="Food tray shows variety in one container.",
            ),
        }
    )
    return payload


class TestDirectProductRouteAssessmentParse(unittest.TestCase):
    def test_parse_valid_assessment(self) -> None:
        raw = direct_product_route_assessment_product_led()
        assessment, reasons = parse_direct_product_route_assessment(raw)
        self.assertEqual(reasons, [])
        self.assertIsNotNone(assessment)
        assert assessment is not None
        self.assertEqual(assessment.recommended_route, RecommendedVisualRoute.PRODUCT_LED)

    def test_missing_assessment_rejected_on_brand_physical(self) -> None:
        payload = _brand_physical()
        payload.pop("directProductRouteAssessment", None)
        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_route_assessment_missing", ctx.exception.reasons)


class TestRouteConsistency(unittest.TestCase):
    def test_simple_product_direct_advantage_rejects_unjustified_analogy(self) -> None:
        assessment, _ = parse_direct_product_route_assessment(
            direct_product_route_assessment(
                readable=True,
                advantage_direct=True,
                mechanism_available=True,
                mechanism_summary="Shoe types grouped in one display",
                unique_gain=False,
                unique_gain_text="",
                recommended_route="ANALOGY_LED",
                route_reason="Food tray parallels shoe variety",
            )
        )
        assert assessment is not None
        self.assertTrue(direct_product_route_viable(assessment))
        reasons = validate_direct_product_route_consistency(
            assessment,
            physical_generator_is_product=False,
            physical_generator_is_packaging=False,
            product_evidence_required=False,
            visibility_policy=ProductVisibilityPolicy.CREATIVE_DECISION,
        )
        self.assertIn("physical_analogy_without_unique_gain", reasons)

    def test_product_led_valid_when_mechanism_exists(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        payload["worksWithoutProductVisible"] = False
        payload["physicalGenerator"] = "Mixed footwear display"
        payload["transferredObject"] = "Mixed footwear display"
        payload["campaignRationale"] = "Grouped footwear types demonstrate one-store breadth"
        payload["directProductRouteAssessment"] = direct_product_route_assessment_product_led(
            mechanism_summary="Distinct sandals, sneakers, and flip-flops share one visible store arrangement."
        )
        result = parse_brand_physical_output(
            payload,
            product_description=SHOE_BRIEF,
            product_name_resolved="צעד צעד",
            visibility_policy="CREATIVE_DECISION",
        )
        self.assertTrue(result.physical_generator_is_product)

    def test_generic_packshot_allows_analogy_with_unique_gain(self) -> None:
        assessment, _ = parse_direct_product_route_assessment(
            direct_product_route_assessment(
                readable=True,
                advantage_direct=True,
                mechanism_available=False,
                mechanism_summary="",
                unique_gain=True,
                unique_gain_text="Partition removal makes invisible breadth visible through a causal transformation shoes alone cannot show in one frame.",
                recommended_route="ANALOGY_LED",
                route_reason="Only the partition-removal mechanism proves unified variety causally.",
            )
        )
        assert assessment is not None
        reasons = validate_direct_product_route_consistency(
            assessment,
            physical_generator_is_product=False,
            physical_generator_is_packaging=False,
            product_evidence_required=False,
            visibility_policy=ProductVisibilityPolicy.CREATIVE_DECISION,
        )
        self.assertEqual(reasons, [])

    def test_abstract_advantage_allows_analogy(self) -> None:
        assessment, _ = parse_direct_product_route_assessment(
            direct_product_route_assessment(
                readable=True,
                advantage_direct=False,
                mechanism_available=False,
                mechanism_summary="",
                unique_gain=True,
                unique_gain_text="Invisible software reliability becomes visible through external failure/success contrast.",
                recommended_route="ANALOGY_LED",
                route_reason="Advantage is not physically demonstrable with the product alone.",
            )
        )
        assert assessment is not None
        reasons = validate_direct_product_route_consistency(
            assessment,
            physical_generator_is_product=False,
            physical_generator_is_packaging=False,
            product_evidence_required=False,
            visibility_policy=ProductVisibilityPolicy.CREATIVE_DECISION,
        )
        self.assertEqual(reasons, [])

    def test_product_integrated_analogy_valid(self) -> None:
        payload = _brand_physical()
        payload["productEvidenceRequired"] = True
        payload["productEvidenceReason"] = "Product must appear while external mechanism carries the idea"
        payload["directProductRouteAssessment"] = direct_product_route_assessment_integrated()
        result = parse_brand_physical_output(payload, visibility_policy="CREATIVE_DECISION")
        self.assertTrue(result.product_evidence_required)

    def test_forbidden_preserves_analogy_only(self) -> None:
        assessment, _ = parse_direct_product_route_assessment(
            direct_product_route_assessment(
                readable=True,
                advantage_direct=True,
                mechanism_available=True,
                mechanism_summary="Shoes could show breadth directly",
                unique_gain=False,
                unique_gain_text="",
                recommended_route="PRODUCT_LED",
                route_reason="Direct route would win under creative decision",
            )
        )
        assert assessment is not None
        reasons = validate_direct_product_route_consistency(
            assessment,
            physical_generator_is_product=False,
            physical_generator_is_packaging=False,
            product_evidence_required=False,
            visibility_policy=ProductVisibilityPolicy.FORBIDDEN,
        )
        self.assertIn("physical_route_assessment_inconsistent", reasons)

    def test_forbidden_analogy_assessment_passes(self) -> None:
        payload = _brand_physical()
        payload["directProductRouteAssessment"] = direct_product_route_assessment(
            readable=True,
            advantage_direct=True,
            mechanism_available=True,
            mechanism_summary="Direct route exists but policy forbids product depiction",
            unique_gain=False,
            unique_gain_text="",
            recommended_route="ANALOGY_LED",
            route_reason="Policy forbids product depiction; external object required.",
        )
        result = parse_brand_physical_output(payload, visibility_policy="FORBIDDEN")
        self.assertFalse(result.physical_generator_is_product)


class TestResolveVisualExecutionRoute(unittest.TestCase):
    def test_product_led_from_assessment(self) -> None:
        assessment, _ = parse_direct_product_route_assessment(direct_product_route_assessment_product_led())
        assert assessment is not None
        route = resolve_visual_execution_route(
            physical_generator_is_product=True,
            direct_product_route_assessment=assessment,
        )
        self.assertEqual(route, VisualExecutionRoute.PRODUCT_LED)

    def test_legacy_fallback_without_assessment(self) -> None:
        route = infer_visual_execution_route(
            physical_generator_is_product=False,
            product_evidence_required=False,
            direct_product_route_assessment=None,
        )
        self.assertEqual(route, VisualExecutionRoute.ANALOGY_LED)


class TestTsaadTsaadRegression(unittest.TestCase):
    def test_food_tray_cannot_win_without_unique_gain(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_output(
                _food_tray_analogy_payload(),
                product_description=SHOE_BRIEF,
                product_name_resolved="צעד צעד",
                visibility_policy="CREATIVE_DECISION",
            )
        reasons = ctx.exception.reasons
        self.assertTrue(
            any(code in reasons for code in ("physical_analogy_without_unique_gain", "physical_unjustified_external_analogy")),
            reasons,
        )

    def test_shoes_are_eligible_direct_category_material(self) -> None:
        payload = _brand_physical()
        payload["physicalGeneratorIsProduct"] = True
        payload["worksWithoutProductVisible"] = False
        payload["physicalGenerator"] = "Sandals, sneakers, and flip-flops grouped together"
        payload["transferredObject"] = "Sandals, sneakers, and flip-flops grouped together"
        payload["campaignRationale"] = "Footwear breadth shown through distinct category members sharing one store space"
        payload["directProductRouteAssessment"] = direct_product_route_assessment_product_led(
            mechanism_summary="Distinct shoe categories visibly share one store arrangement."
        )
        result = parse_brand_physical_output(
            payload,
            product_description=SHOE_BRIEF,
            product_name_resolved="צעד צעד",
        )
        self.assertTrue(result.physical_generator_is_product)

    def test_food_allowed_when_unique_mechanism_stated(self) -> None:
        payload = _food_tray_analogy_payload()
        payload["directProductRouteAssessment"] = direct_product_route_assessment(
            readable=True,
            advantage_direct=True,
            mechanism_available=True,
            mechanism_summary="Shoe lineup alone reads as catalog breadth without causal proof",
            unique_gain=True,
            unique_gain_text="Partition removal supplies a causal one-space transformation that a static multi-shoe lineup cannot perform.",
            translation_cost="LOW",
            recommended_route="ANALOGY_LED",
            route_reason="Causal partition removal proves unified variety more forcefully than grouped shoes.",
        )
        result = parse_brand_physical_output(
            payload,
            product_description=SHOE_BRIEF,
            product_name_resolved="צעד צעד",
        )
        self.assertFalse(result.physical_generator_is_product)


class TestMethodologyPrompts(unittest.TestCase):
    def test_brand_physical_contains_direct_product_priority(self) -> None:
        self.assertIn(BUILDER1_SIMPLE_PRODUCT_DIRECT_ADVANTAGE_PRIORITY.splitlines()[0], STAGE_BRAND_PHYSICAL_SYSTEM)
        self.assertIn("directProductRouteAssessment", STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_popular_analogy_scoped_to_analogy_branch(self) -> None:
        self.assertIn("ONLY after ANALOGY_LED is already justified", BUILDER1_POPULAR_ANALOGY_FIRST)
        self.assertIn("does NOT decide PRODUCT_LED vs ANALOGY_LED", BUILDER1_POPULAR_ANALOGY_FIRST)

    def test_public_simplicity_not_analogy_requirement(self) -> None:
        self.assertIn("does NOT mean", BUILDER1_PUBLIC_SIMPLICITY)
        self.assertIn("simple everyday analogy", BUILDER1_PUBLIC_SIMPLICITY.lower())

    def test_product_evidence_tests_direct_route_first(self) -> None:
        self.assertIn("TESTED BEFORE EXTERNAL ANALOGY", BUILDER1_PRODUCT_EVIDENCE_EXCEPTION)

    def test_conceptual_stage_allows_multiple_physical_realizations(self) -> None:
        self.assertIn("product-integrated mechanism", STAGE_CONCEPTUAL_SCAN_SYSTEM.lower())


class TestLegacyCompatibility(unittest.TestCase):
    def test_stored_plan_without_assessment_uses_legacy_route(self) -> None:
        data = copy.deepcopy(_base_campaign(2))
        data["planningInternals"] = {
            "physicalGeneratorIsProduct": False,
            "productEvidenceRequired": False,
        }
        plan = _parse(data, 2)
        route = visual_route_for_plan(plan)
        self.assertEqual(route, VisualExecutionRoute.ANALOGY_LED)

    def test_stored_plan_with_assessment_uses_assessment(self) -> None:
        data = copy.deepcopy(_base_campaign(2))
        data["planningInternals"] = {
            "physicalGeneratorIsProduct": False,
            "productEvidenceRequired": False,
            "directProductRouteAssessment": direct_product_route_assessment_product_led(),
        }
        plan = _parse(data, 2)
        internals = dict(plan.planning_internals or {})
        internals.pop("visualExecutionRoute", None)
        internals["directProductRouteAssessment"] = direct_product_route_assessment_product_led()
        plan.planning_internals = internals
        route = visual_route_for_plan(plan)
        self.assertEqual(route, VisualExecutionRoute.PRODUCT_LED)


class TestLegitimateCrossDomainCampaigns(unittest.TestCase):
    def test_default_rubber_ball_analogy_still_valid(self) -> None:
        result = parse_brand_physical_output(_brand_physical())
        self.assertFalse(result.physical_generator_is_product)

    def test_different_worlds_exploration_still_required(self) -> None:
        payload = _brand_physical()
        payload["physicalCandidates"] = _physical_candidates(worlds=["kitchen", "kitchen", "kitchen", "kitchen"])
        payload["physicalEvaluations"] = _physical_evaluations()
        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_output(payload)
        self.assertIn("physical_all_candidates_same_world", ctx.exception.reasons)


if __name__ == "__main__":
    unittest.main()
