"""
Builder1 essential fact fusion / selection preservation tests.

Run: python -m unittest tests.test_builder1_essential_fact_fusion -v
"""
from __future__ import annotations

import unittest

from engine.builder1_creative_methodology import (
    earliest_methodology_repair_stage,
    deterministic_methodology_checks,
)
from engine.builder1_essential_fact_fusion import (
    BUILDER1_ESSENTIAL_FACT_CULTURAL_CONTEXT,
    BUILDER1_ESSENTIAL_FACT_FUSION,
    BUILDER1_ESSENTIAL_FACT_FUSION_TEST,
    classify_essential_fact,
    format_essential_fact_fusion_prompt_block,
    fusion_required_for_brief,
    partition_essential_facts,
    scan_essential_fact_fusion,
)
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
    build_brand_physical_user_prompt,
    build_conceptual_stage_user_prompt,
)
from engine.builder1_selected_creative_brief import SelectedCreativeBrief
from tests.test_builder1_series import _base_campaign, _graphic


def _marketing_block() -> str:
    return "word " * 50


def _bosa_brief() -> SelectedCreativeBrief:
    return SelectedCreativeBrief(
        essential_facts=[
            "בוסה הוא בושם לגברים",
            "בוסה מיוצר בישראל",
        ],
        supporting_evidence=["Israeli men's fragrance category alternative"],
        mandatory_constraints=[],
    )


def _bosa_bad_lamp_plan() -> dict:
    plan = _base_campaign(2)
    plan.update(
        {
            "productName": "בוסה",
            "productNameResolved": "בוסה",
            "productDescription": "בושם לגברים מתוצרת ישראל",
            "detectedLanguage": "he",
            "relativeAdvantage": "A locally produced Israeli alternative in the men's fragrance category",
            "brandSlogan": "בוסה. ריח מקומי.",
            "sloganAction": "להציג חלופה ישראלית בקטגוריית בושם לגברים",
            "conceptualGenerator": "Foreign origin replaced by local origin",
            "conceptualGeneratorAction": "Imported source becomes Israeli source",
            "physicalGenerator": "Desk lamp with foreign plug replaced by Israeli plug",
            "transferredObject": "Desk lamp with foreign plug replaced by Israeli plug",
            "transferredObjectAction": "The foreign power source is replaced by an Israeli one",
            "campaignRationale": "Shows local origin through a generic electrical replacement metaphor",
            "graphicGenerator": _graphic(),
            "planningInternals": {
                "selectedCreativeBrief": _bosa_brief().to_dict(),
                "physicalGeneratorIsProduct": False,
                "directProductRouteAssessment": {
                    "productOrCategoryImmediatelyReadable": True,
                    "relativeAdvantageDirectlyExpressibleWithProduct": True,
                    "productLedAdvertisingMechanismAvailable": True,
                    "productLedMechanismSummary": "Fragrance form could carry local identity",
                    "externalAnalogyAddsUniquePersuasiveGain": True,
                    "externalAnalogyUniqueGain": "Electrical replacement shows import-to-local shift",
                    "additionalTranslationCost": "LOW",
                    "recommendedRoute": "ANALOGY_LED",
                    "routeDecisionReason": "Electrical metaphor chosen",
                },
            },
            "ads": [
                {
                    "index": 1,
                    "variationLabel": "v1",
                    "newContribution": "Lamp import swap",
                    "physicalExecution": "Desk lamp with Israeli plug replacing foreign plug",
                    "visualExecution": "Office lamp on desk",
                    "sceneDescription": "Clean studio desk with lamp",
                    "conceptualExecution": "Foreign becomes local",
                    "conceptualActionProof": "Plug replacement proves local origin",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": _marketing_block(),
                },
                {
                    "index": 2,
                    "variationLabel": "v2",
                    "newContribution": "Second lamp variant",
                    "physicalExecution": "Second lamp with Israeli socket",
                    "visualExecution": "Different lamp angle",
                    "sceneDescription": "Neutral background",
                    "conceptualExecution": "Local source proof",
                    "conceptualActionProof": "Second electrical proof",
                    "headline": None,
                    "headlineNeededReason": "Self-explanatory",
                    "marketingText": _marketing_block(),
                },
            ],
        }
    )
    return plan


def _bosa_fused_perfume_plan() -> dict:
    plan = _bosa_bad_lamp_plan()
    plan.update(
        {
            "conceptualGenerator": "Israel expressed through men's fragrance form",
            "conceptualGeneratorAction": "Perfume bottle carries a second Israeli identity",
            "physicalGenerator": "Men's perfume bottle whose silhouette evokes a familiar Israeli canteen form",
            "transferredObject": "Men's perfume bottle whose silhouette evokes a familiar Israeli canteen form",
            "transferredObjectAction": "The bottle remains recognizably perfume while its shape expresses Israeli identity",
            "campaignRationale": "Fuses men's fragrance category with Israeli local identity in one object",
            "planningInternals": {
                **plan["planningInternals"],
                "directProductRouteAssessment": {
                    **plan["planningInternals"]["directProductRouteAssessment"],
                    "recommendedRoute": "PRODUCT_INTEGRATED_ANALOGY",
                    "routeDecisionReason": "Fragrance form carries local identity directly",
                },
            },
            "ads": [
                {
                    **plan["ads"][0],
                    "physicalExecution": "Men's perfume bottle with canteen-like silhouette",
                    "visualExecution": "Fragrance bottle as hero object",
                    "sceneDescription": "Studio product scene with perfume bottle",
                    "conceptualExecution": "Perfume and Israel fused",
                    "conceptualActionProof": "Category remains fragrance while form expresses Israel",
                },
                {
                    **plan["ads"][1],
                    "physicalExecution": "Second men's fragrance bottle variant with Israeli visual shorthand",
                    "visualExecution": "Different perfume bottle angle",
                    "sceneDescription": "Clean background with fragrance bottle",
                    "conceptualExecution": "Second fused fragrance proof",
                    "conceptualActionProof": "Still clearly men's perfume with Israeli identity",
                },
            ],
        }
    )
    return plan


class TestEssentialFactClassification(unittest.TestCase):
    def test_bosa_facts_partition(self) -> None:
        category, advantage, _general = partition_essential_facts(_bosa_brief().essential_facts)
        self.assertEqual(len(category), 1)
        self.assertEqual(len(advantage), 1)
        self.assertEqual(classify_essential_fact("בוסה הוא בושם לגברים"), "category_identity")
        self.assertEqual(classify_essential_fact("בוסה מיוצר בישראל"), "advantage")

    def test_fusion_required_for_bosa_brief(self) -> None:
        self.assertTrue(
            fusion_required_for_brief(
                _bosa_brief(),
                relative_advantage="Israeli men's fragrance alternative",
            )
        )


class TestEssentialFactFusionDetection(unittest.TestCase):
    def test_bosa_lamp_plan_rejected(self) -> None:
        evidence: list[dict] = []
        reasons = scan_essential_fact_fusion(_bosa_bad_lamp_plan(), integrity_evidence=evidence)
        self.assertIn("relative_advantage_without_product_application", reasons)
        self.assertTrue(evidence)
        self.assertEqual(evidence[0]["detector"], "essential_fact_fusion")

    def test_bosa_fused_perfume_plan_passes(self) -> None:
        reasons = scan_essential_fact_fusion(_bosa_fused_perfume_plan())
        self.assertNotIn("relative_advantage_without_product_application", reasons)

    def test_single_category_fact_only_does_not_require_fusion(self) -> None:
        plan = _bosa_bad_lamp_plan()
        plan["planningInternals"]["selectedCreativeBrief"]["essentialFacts"] = ["בוסה הוא בושם לגברים"]
        reasons = scan_essential_fact_fusion(plan)
        self.assertNotIn("relative_advantage_without_product_application", reasons)

    def test_integrated_into_methodology_checks(self) -> None:
        reasons = deterministic_methodology_checks(_bosa_bad_lamp_plan())
        self.assertIn("relative_advantage_without_product_application", reasons)

    def test_repair_stage_routes_to_conceptual_scan(self) -> None:
        stage = earliest_methodology_repair_stage(["relative_advantage_without_product_application"])
        self.assertEqual(stage, "conceptual_scan")


class TestEssentialFactFusionPrompts(unittest.TestCase):
    def test_methodology_blocks_present_in_stage_systems(self) -> None:
        self.assertIn(BUILDER1_ESSENTIAL_FACT_FUSION.splitlines()[0], STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn(BUILDER1_ESSENTIAL_FACT_FUSION_TEST.splitlines()[0], STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertIn("ESSENTIAL FACT FUSION", STAGE_BRAND_PHYSICAL_SYSTEM)
        self.assertIn(BUILDER1_ESSENTIAL_FACT_CULTURAL_CONTEXT.splitlines()[0], STAGE_BRAND_PHYSICAL_SYSTEM)
        self.assertIn("ESSENTIAL FACT FUSION", STAGE_SERIES_ADS_SYSTEM)
        self.assertIn("fuse them", STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM.lower())

    def test_conceptual_user_prompt_includes_fusion_guidance(self) -> None:
        prompt = build_conceptual_stage_user_prompt(
            product_description="בושם לגברים מתוצרת ישראל",
            product_name_resolved="בוסה",
            strategic_problem="Imported men's fragrances dominate",
            relative_advantage="Israeli men's fragrance alternative",
            brand_slogan="בוסה. ריח מקומי.",
            slogan_derivation="Local fragrance alternative",
            implied_action="Present local men's fragrance choice",
            exploration_seed="seed-1",
            selected_creative_brief=_bosa_brief(),
        )
        self.assertIn("Essential Fact Fusion Test", prompt)
        self.assertIn("בושם לגברים", prompt)
        self.assertIn("fusion inside the product/category", prompt)

    def test_brand_physical_user_prompt_includes_fusion_guidance(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="בוסה",
            product_description="בושם לגברים מתוצרת ישראל",
            detected_language="he",
            format_value="portrait",
            strategic_problem="Imported fragrances dominate",
            relative_advantage="Israeli men's fragrance alternative",
            brand_slogan="בוסה. ריח מקומי.",
            slogan_derivation="Local alternative",
            implied_action="Present local choice",
            conceptual={"generator": "Fused identity", "action": "Express Israel through fragrance"},
            selected_creative_brief=_bosa_brief(),
        )
        self.assertIn("Product/category identity facts", prompt)
        self.assertIn("Relative-advantage facts", prompt)

    def test_format_block_empty_without_brief(self) -> None:
        self.assertEqual(format_essential_fact_fusion_prompt_block(None), "")


if __name__ == "__main__":
    unittest.main()
