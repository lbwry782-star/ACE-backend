"""
Builder1 literal_product_embodiment false-positive regression tests (production בוסה).

Run: python -m unittest tests.test_builder1_literal_product_embodiment_regression -v
"""
from __future__ import annotations

import inspect
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder1_creative_methodology import deterministic_methodology_checks
from engine.builder1_direct_product_route import (
    AdditionalTranslationCost,
    DirectProductRouteAssessment,
    RecommendedVisualRoute,
)
from engine.builder1_final_stages import BrandPhysicalOutput
from engine.builder1_integrity_recovery import revalidate_rejected_plan_dict
from engine.builder1_literal_embodiment import (
    _detect_literal_product_embodiment,
    scan_brand_physical_early_literal_product_embodiment,
    scan_literal_embodiment_bias,
)
from engine.builder1_planning_pipeline import run_builder1_campaign_pipeline
from tests.test_builder1_series import _base_campaign, _graphic


def _marketing_block() -> str:
    return "word " * 50


def _route_assessment(**overrides: object) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "productOrCategoryImmediatelyReadable": True,
        "relativeAdvantageDirectlyExpressibleWithProduct": False,
        "productLedAdvertisingMechanismAvailable": False,
        "productLedMechanismSummary": "",
        "externalAnalogyAddsUniquePersuasiveGain": True,
        "externalAnalogyUniqueGain": "Isolated power-source switch proves local origin clearly",
        "additionalTranslationCost": "LOW",
        "recommendedRoute": "ANALOGY_LED",
        "routeDecisionReason": (
            "למרות שקיימת החלפת מוצר ישירה, מנגנון החלפת מקור החשמל מבודד וממחיש את יתרון המקור המקומי"
        ),
    }
    base.update(overrides)
    return base


def _analogy_label_mention_plan(
    *,
    product_name: str,
    physical_generator: str,
    transferred_object: str,
    language: str = "he",
) -> Dict[str, Any]:
    plan = _base_campaign(2)
    brand_slogan = "Local power you can trust"
    slogan_action = "Prove local origin through visible mechanism"
    plan.update(
        {
            "productNameResolved": product_name,
            "detectedLanguage": language,
            "brandSlogan": brand_slogan,
            "sloganAction": slogan_action,
            "physicalGenerator": physical_generator,
            "transferredObject": transferred_object,
            "transferredObjectAction": "Power source switches from import to local supply",
            "conceptualGenerator": "Local origin becomes visible",
            "conceptualGeneratorAction": "Switch proves local source",
            "campaignRationale": "External lamp mechanism proves origin without product packshot",
            "graphicGenerator": _graphic(),
            "planningInternals": {
                "physicalGeneratorIsProduct": False,
                "physicalGeneratorIsPackaging": False,
                "directProductRouteAssessment": _route_assessment(),
                "whyClearerThanShowingProduct": "Lamp switch proves local origin more clearly than product shot",
                "conceptualGeneratorWhyItExpressesSlogan": (
                    "Power-source switch expresses local origin without literal product depiction"
                ),
                "conceptualLineage": {
                    "selectedConceptCandidateId": "C01",
                    "sourceSloganCandidateId": "S01",
                    "fixedBrandSlogan": brand_slogan,
                    "fixedImpliedAction": slogan_action,
                },
                "adInternals": {
                    1: {
                        "conceptualActionProof": "Plug moves from import-labeled source to local-labeled source",
                        "categoryRelevanceReason": "Lamp is external mechanism, not product packshot",
                        "relativeAdvantageConnection": "Visible local source proves origin advantage",
                        "immediateClarityReason": "Two marked sources make the switch readable instantly",
                        "singleChangedPropertyOrAction": "Active power source changes to local connection",
                    },
                    2: {
                        "conceptualActionProof": "Second ad repeats switch law with alternate lamp angle",
                        "categoryRelevanceReason": "External lamp proves mechanism without product-as-object",
                        "relativeAdvantageConnection": "Local connection remains the persuasive proof",
                        "immediateClarityReason": "Marked sources remain legible in second execution",
                        "singleChangedPropertyOrAction": "Plug rests at local source",
                    },
                },
            },
            "ads": [
                {
                    "index": 1,
                    "variationLabel": "v1",
                    "newContribution": "Switch proof one",
                    "physicalExecution": "Desk lamp with labeled power sources",
                    "visualExecution": "Lamp plug moves to local outlet",
                    "sceneDescription": "Clean studio desk with lamp",
                    "conceptualExecution": "Import source replaced by local source",
                    "conceptualActionProof": "Visible plug switch proves mechanism",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": _marketing_block(),
                },
                {
                    "index": 2,
                    "variationLabel": "v2",
                    "newContribution": "Switch proof two",
                    "physicalExecution": "Second lamp variant with two marked sources",
                    "visualExecution": "Local connection emphasized",
                    "sceneDescription": "Minimal studio background",
                    "conceptualExecution": "Same switch law",
                    "conceptualActionProof": "Second switch proof",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": _marketing_block(),
                },
            ],
        }
    )
    return plan


def _bosa_production_rejected_plan() -> Dict[str, Any]:
    physical = (
        "מנורת שולחן קבועה שמקור החשמל שלה מוחלף מכבל המגיע מארגז המסומן ״מחו״ל״ "
        "לשקע המסומן ״ייצור בישראל״; השם בוסה מופיע ליד החיבור המקומי בטיפוגרפיה עברית פשוטה."
    )
    transferred = "מנורת שולחן עם תקע ושני מקורות חשמל מסומנים"
    return _analogy_label_mention_plan(
        product_name="בוסה",
        physical_generator=physical,
        transferred_object=transferred,
        language="he",
    )


def _brand_physical_from_plan(plan: Dict[str, Any]) -> BrandPhysicalOutput:
    internals = plan.get("planningInternals") or {}
    assessment_raw = internals.get("directProductRouteAssessment") or plan.get("directProductRouteAssessment")
    assessment = None
    if isinstance(assessment_raw, dict):
        assessment = DirectProductRouteAssessment(
            product_or_category_immediately_readable=bool(
                assessment_raw.get("productOrCategoryImmediatelyReadable")
            ),
            relative_advantage_directly_expressible_with_product=bool(
                assessment_raw.get("relativeAdvantageDirectlyExpressibleWithProduct")
            ),
            product_led_advertising_mechanism_available=bool(
                assessment_raw.get("productLedAdvertisingMechanismAvailable")
            ),
            product_led_mechanism_summary=str(assessment_raw.get("productLedMechanismSummary") or ""),
            external_analogy_adds_unique_persuasive_gain=bool(
                assessment_raw.get("externalAnalogyAddsUniquePersuasiveGain")
            ),
            external_analogy_unique_gain=str(assessment_raw.get("externalAnalogyUniqueGain") or ""),
            additional_translation_cost=AdditionalTranslationCost(
                str(assessment_raw.get("additionalTranslationCost") or AdditionalTranslationCost.LOW.value)
            ),
            recommended_route=RecommendedVisualRoute(
                str(assessment_raw.get("recommendedRoute") or RecommendedVisualRoute.ANALOGY_LED.value)
            ),
            route_decision_reason=str(assessment_raw.get("routeDecisionReason") or ""),
        )
    return BrandPhysicalOutput(
        product_name_resolved=str(plan.get("productNameResolved") or ""),
        physical_generator=str(plan.get("physicalGenerator") or ""),
        physical_generator_natural_purpose="",
        physical_generator_campaign_role="",
        physical_generator_is_product=bool(internals.get("physicalGeneratorIsProduct")),
        physical_generator_is_packaging=bool(internals.get("physicalGeneratorIsPackaging")),
        works_without_product_visible=True,
        transferred_object=str(plan.get("transferredObject") or ""),
        transferred_object_action=str(plan.get("transferredObjectAction") or ""),
        why_clearer_than_showing_product=str(internals.get("whyClearerThanShowingProduct") or ""),
        medium_participates=False,
        medium_role="",
        campaign_rationale=str(plan.get("campaignRationale") or ""),
        product_evidence_required=False,
        product_evidence_reason="",
        clearer_than_conventional_product_shot=True,
        survives_product_removal=True,
        direct_product_route_assessment=assessment,
    )


class TestLiteralProductEmbodimentRegression(unittest.TestCase):
    def test_analogy_led_label_mention_passes(self) -> None:
        plan = _analogy_label_mention_plan(
            product_name="BrightLamp",
            physical_generator=(
                "Desk lamp whose plug moves from import-labeled source to local-labeled source; "
                "the name BrightLamp appears as plain typography near the local connection."
            ),
            transferred_object="Desk lamp with plug and two labeled power sources",
            language="en",
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_product_embodiment", reasons)

    def test_bosa_production_regression_passes(self) -> None:
        reasons = scan_literal_embodiment_bias(_bosa_production_rejected_plan())
        self.assertNotIn("literal_product_embodiment", reasons)

    def test_english_long_product_name_label_mention_passes(self) -> None:
        plan = _analogy_label_mention_plan(
            product_name="NorthRiver",
            physical_generator=(
                "Table fan switching from import cord to local outlet; "
                "NorthRiver appears as small plain text near the local plug."
            ),
            transferred_object="Table fan with plug and two marked power sources",
            language="en",
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_product_embodiment", reasons)

    def test_hebrew_product_name_label_mention_passes(self) -> None:
        plan = _analogy_label_mention_plan(
            product_name="גלית",
            physical_generator="מאוורר שולחן עם תקע; השם גלית מופיע בטיפוגרפיה ליד החיבור המקומי",
            transferred_object="מאוורר שולחן עם תקע ושני מקורות חשמל",
            language="he",
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_product_embodiment", reasons)

    def test_headline_copy_label_description_alone_does_not_fail(self) -> None:
        plan = _bosa_production_rejected_plan()
        plan["ads"][0]["headline"] = "בוסה — ייצור בישראל"
        plan["ads"][0]["marketingText"] = "בוסה " + _marketing_block()
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_product_embodiment", reasons)

    def test_product_led_with_mechanism_not_rejected_for_name_mention(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "productNameResolved": "FreshBite",
                "physicalGenerator": "FreshBite bars arranged in ascending freshness-color gradient",
                "transferredObject": "FreshBite snack bars in gradient arrangement",
                "physicalGeneratorIsProduct": True,
                "directProductRouteAssessment": _route_assessment(
                    recommendedRoute="PRODUCT_LED",
                    productLedAdvertisingMechanismAvailable=True,
                    productLedMechanismSummary="Color gradient encodes freshness progression",
                    relativeAdvantageDirectlyExpressibleWithProduct=True,
                ),
                "ads": plan["ads"],
            }
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertNotIn("literal_product_embodiment", reasons)

    def test_lazy_packshot_still_blocked_at_brand_physical_parse(self) -> None:
        from engine.builder1_final_stages import parse_brand_physical_output
        from engine.builder1_staged_parsers import StageParseError

        payload = {
            "productNameResolved": "FreshBite",
            "physicalGenerator": "FreshBite catalog packshot on white background",
            "physicalGeneratorNaturalPurpose": "Snack packaging display",
            "physicalGeneratorCampaignRole": "Show product",
            "physicalGeneratorIsProduct": True,
            "physicalGeneratorIsPackaging": True,
            "worksWithoutProductVisible": False,
            "transferredObject": "FreshBite packshot",
            "transferredObjectAction": "Static product photo",
            "whyClearerThanShowingProduct": "Generic packshot",
            "mediumParticipates": False,
            "mediumRole": "",
            "campaignRationale": "Packshot only",
            "directProductRouteAssessment": _route_assessment(
                recommendedRoute="PRODUCT_LED",
                productLedAdvertisingMechanismAvailable=False,
            ),
        }
        with self.assertRaises(StageParseError) as ctx:
            parse_brand_physical_output(payload, product_description="Snack bars", product_name_resolved="FreshBite")
        self.assertIn("physical_generator_is_packaging", ctx.exception.reasons)

    def test_exact_product_as_visual_object_still_rejected(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "productNameResolved": "RoutePro",
                "physicalGenerator": "RoutePro",
                "transferredObject": "RoutePro",
                "planningInternals": {
                    "physicalGeneratorIsProduct": False,
                    "directProductRouteAssessment": _route_assessment(),
                },
            }
        )
        reasons = scan_literal_embodiment_bias(plan)
        self.assertIn("literal_product_embodiment", reasons)

    def test_genuine_violation_populates_diagnostic_evidence(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "productNameResolved": "RoutePro",
                "physicalGenerator": "RoutePro",
                "transferredObject": "RoutePro",
                "planningInternals": {
                    "directProductRouteAssessment": _route_assessment(),
                },
            }
        )
        evidence: list[dict] = []
        reasons = scan_literal_embodiment_bias(plan, evidence)
        self.assertIn("literal_product_embodiment", reasons)
        literal = [item for item in evidence if item.get("code") == "literal_product_embodiment"]
        self.assertTrue(literal)
        entry = literal[0]
        self.assertEqual(entry.get("field"), "transferredObject")
        self.assertEqual(entry.get("matchedTerms"), ["RoutePro"])
        self.assertIn("productIdentity", entry)
        self.assertIn("normalizedVisualObject", entry)
        self.assertIn("recommendedRoute", entry)
        self.assertIn("embodimentBasis", entry)

    def test_genuine_violation_blocks_campaign_integrity(self) -> None:
        plan = _base_campaign(2)
        plan.update(
            {
                "productNameResolved": "RoutePro",
                "physicalGenerator": "RoutePro",
                "transferredObject": "RoutePro",
                "planningInternals": {
                    "directProductRouteAssessment": _route_assessment(),
                },
            }
        )
        reasons = deterministic_methodology_checks(plan)
        self.assertIn("literal_product_embodiment", reasons)

    def test_early_brand_physical_gate_catches_genuine_violation_only(self) -> None:
        bad_plan = _base_campaign(2)
        bad_plan.update(
            {
                "productNameResolved": "RoutePro",
                "physicalGenerator": "RoutePro",
                "transferredObject": "RoutePro",
                "planningInternals": {"directProductRouteAssessment": _route_assessment()},
            }
        )
        good_plan = _bosa_production_rejected_plan()
        bad_reasons = scan_brand_physical_early_literal_product_embodiment(
            product_name_resolved="RoutePro",
            brand_physical=_brand_physical_from_plan(bad_plan),
        )
        good_reasons = scan_brand_physical_early_literal_product_embodiment(
            product_name_resolved="בוסה",
            brand_physical=_brand_physical_from_plan(good_plan),
        )
        self.assertIn("literal_product_embodiment", bad_reasons)
        self.assertEqual(good_reasons, [])

    def test_pipeline_wires_early_brand_physical_literal_gate(self) -> None:
        source = inspect.getsource(run_builder1_campaign_pipeline)
        self.assertIn("scan_brand_physical_early_literal_product_embodiment", source)
        self.assertIn('StageParseError("brand_physical"', source)

    def test_bosa_production_plan_revalidates_without_openai(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as plan_mock:
            result = revalidate_rejected_plan_dict(_bosa_production_rejected_plan())
            plan_mock.assert_not_called()
        self.assertTrue(result["ok"], msg=str(result["reasons"]))

    def test_recovery_assessment_performs_no_image_generation(self) -> None:
        import engine.builder1_integrity_recovery as recovery_mod

        source = inspect.getsource(recovery_mod.revalidate_rejected_plan_dict)
        self.assertNotIn("generate_builder1_ad_image", source)
        self.assertNotIn("image_provider", source.lower())

    def test_detect_helper_matches_scan_for_bosa(self) -> None:
        plan = _bosa_production_rejected_plan()
        self.assertEqual(_detect_literal_product_embodiment(plan), [])
        self.assertNotIn("literal_product_embodiment", scan_literal_embodiment_bias(plan))


if __name__ == "__main__":
    unittest.main()
