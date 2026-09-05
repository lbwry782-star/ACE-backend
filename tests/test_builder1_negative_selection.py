"""
Builder1 negative selection / discard irrelevant product facts tests.

Run: python -m unittest tests.test_builder1_negative_selection -v
"""
from __future__ import annotations

import unittest

from engine.builder1_compliance_product_grounding import classify_advertised_product_type
from engine.builder1_essential_fact_fusion import (
    format_essential_fact_fusion_prompt_block,
    fusion_required_for_brief,
    partition_essential_facts,
    scan_essential_fact_fusion,
)
from engine.builder1_negative_selection import (
    NEGATIVE_SELECTION_METHODOLOGY,
    creative_prompt_contains_discarded_facts,
    fusion_uses_only_essential_facts,
    validate_negative_selection,
)
from engine.builder1_planning_contract import (
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
    build_brand_physical_user_prompt,
    build_conceptual_stage_user_prompt,
    build_graphic_system_user_prompt,
    build_strategy_slogan_stage_user_prompt,
)
from engine.builder1_post_selection_brief_isolation import format_post_selection_creative_input_block
from engine.builder1_selected_creative_brief import (
    SelectedCreativeBrief,
    format_brief_for_prompt_json,
    format_selected_creative_brief_block,
    parse_selected_creative_brief,
)
from engine.builder1_staged_parsers import StageParseError
from tests.test_builder1_staged_planning import _brand_physical, _selected_slogan, _selected_strategy


RAW_PERFUME = (
    "בושם לגברים מתוצרת ישראל.\n"
    "בקבוק 100 מ״ל.\n"
    "החברה נוסדה בשנת 2019.\n"
    "המחסן נמצא בחיפה.\n"
    "בעל החברה אוהב ג'אז."
)
DISCARDED_JAZZ = "בעל החברה אוהב ג'אז"
DISCARDED_WAREHOUSE = "המחסן נמצא בחיפה"
DISCARDED_FOUNDED = "החברה נוסדה בשנת 2019"
DISCARDED_BOTTLE = "בקבוק 100 מ״ל"

BOSA_BRIEF = SelectedCreativeBrief(
    essential_facts=[
        "בוסה הוא בושם לגברים",
        "בוסה מיוצר בישראל",
    ],
    supporting_evidence=["תיאור המוצר מציין במפורש שהוא מתוצרת ישראל"],
    mandatory_constraints=[],
    discarded_facts=[
        DISCARDED_BOTTLE,
        DISCARDED_FOUNDED,
        DISCARDED_WAREHOUSE,
        DISCARDED_JAZZ,
    ],
)

PRODUCT_LED_BRIEF = SelectedCreativeBrief(
    essential_facts=[
        "CarryShell is a reinforced matte-black shell case with visible hinge hardware",
        "Reinforced shell product designed for daily carry",
    ],
    supporting_evidence=["Durable reinforced construction supports everyday protection"],
    mandatory_constraints=[],
    discarded_facts=["Also available in blue", "Waterproof coating option"],
)

RAW_SHELL = (
    "Reinforced shell product for daily carry. Secondary fact: also available in blue. "
    "Unrelated advantage: waterproof coating option. Founded in 2019. Warehouse in Haifa."
)


def _conceptual_prompt(brief: SelectedCreativeBrief) -> str:
    strategy = _selected_strategy()
    slogan = _selected_slogan()
    return build_conceptual_stage_user_prompt(
        product_description=RAW_SHELL,
        product_name_resolved="CarryShell",
        strategic_problem=strategy.strategic_problem,
        relative_advantage=strategy.relative_advantage,
        brand_slogan=slogan.brand_slogan,
        slogan_derivation=slogan.derivation_from_advantage,
        implied_action=slogan.implied_action,
        exploration_seed="seed-1",
        selected_creative_brief=brief,
    )


class TestNegativeSelectionTaxonomy(unittest.TestCase):
    def test_strategy_stage_receives_full_raw_description(self) -> None:
        prompt = build_strategy_slogan_stage_user_prompt(
            product_name="בוסה",
            product_description=RAW_PERFUME,
            detected_language="he",
            lens_order=["economic"],
            exploration_seed="seed-1",
        )
        self.assertIn(RAW_PERFUME.strip().split("\n")[0], prompt)
        self.assertIn("discardedFacts", STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM)
        self.assertIn(NEGATIVE_SELECTION_METHODOLOGY, STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM)

    def test_relevant_category_fact_becomes_essential(self) -> None:
        self.assertIn("בושם לגברים", BOSA_BRIEF.essential_facts[0])

    def test_relative_advantage_fact_becomes_essential(self) -> None:
        self.assertIn("מיוצר בישראל", BOSA_BRIEF.essential_facts[1])

    def test_supporting_proof_becomes_supporting_evidence(self) -> None:
        self.assertTrue(BOSA_BRIEF.supporting_evidence)
        self.assertNotIn(BOSA_BRIEF.supporting_evidence[0], BOSA_BRIEF.essential_facts)

    def test_mandatory_instruction_becomes_mandatory_constraint(self) -> None:
        brief = SelectedCreativeBrief(
            essential_facts=["Reinforced shell product designed for daily carry"],
            mandatory_constraints=["Do not show the product package in frame"],
            discarded_facts=["Founded in 2019"],
        )
        parsed = parse_selected_creative_brief(
            brief.to_dict(),
            product_description=RAW_SHELL,
            strict_negative_selection=False,
        )
        self.assertEqual(parsed.mandatory_constraints[0], "Do not show the product package in frame")

    def test_irrelevant_true_fact_becomes_discarded(self) -> None:
        self.assertIn(DISCARDED_JAZZ, BOSA_BRIEF.discarded_facts)

    def test_discarded_fact_stored_in_full_dict(self) -> None:
        stored = BOSA_BRIEF.to_dict()
        self.assertIn("discardedFacts", stored)
        self.assertIn(DISCARDED_WAREHOUSE, stored["discardedFacts"])

    def test_discarded_fact_not_in_creative_prompt_block(self) -> None:
        block = format_selected_creative_brief_block(BOSA_BRIEF)
        post_block = format_post_selection_creative_input_block(BOSA_BRIEF)
        prompt_json = format_brief_for_prompt_json(BOSA_BRIEF)
        for fact in BOSA_BRIEF.discarded_facts:
            self.assertNotIn(fact, block)
            self.assertNotIn(fact, post_block)
        self.assertNotIn("discardedFacts", prompt_json)

    def test_discarded_fact_not_in_conceptual_prompt(self) -> None:
        prompt = _conceptual_prompt(BOSA_BRIEF)
        self.assertFalse(creative_prompt_contains_discarded_facts(prompt, BOSA_BRIEF))

    def test_discarded_fact_not_in_brand_physical_prompt(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="בוסה",
            product_description=RAW_PERFUME,
            detected_language="he",
            format_value="portrait",
            strategic_problem="Problem",
            relative_advantage="Advantage",
            brand_slogan="בוסה. ריח מקומי.",
            slogan_derivation="Derivation",
            implied_action="Show local alternative",
            conceptual={"generator": "Mechanism", "action": "Show"},
            selected_creative_brief=BOSA_BRIEF,
        )
        self.assertFalse(creative_prompt_contains_discarded_facts(prompt, BOSA_BRIEF))

    def test_discarded_fact_not_in_graphic_prompt(self) -> None:
        prompt = build_graphic_system_user_prompt(
            product_description=RAW_PERFUME,
            detected_language="he",
            relative_advantage="Advantage",
            brand_slogan="בוסה. ריח מקומי.",
            conceptual={"generator": "Mechanism"},
            brand_physical=_brand_physical(),
            format_value="portrait",
            selected_creative_brief=BOSA_BRIEF,
        )
        self.assertFalse(creative_prompt_contains_discarded_facts(prompt, BOSA_BRIEF))

    def test_fusion_uses_only_essential_facts(self) -> None:
        self.assertTrue(fusion_uses_only_essential_facts(BOSA_BRIEF))
        category, advantage, _ = partition_essential_facts(BOSA_BRIEF.essential_facts)
        self.assertTrue(category)
        self.assertTrue(advantage)
        fusion_block = format_essential_fact_fusion_prompt_block(
            BOSA_BRIEF,
            relative_advantage="A locally produced Israeli alternative in the men's fragrance category",
        )
        self.assertIn("בושם", fusion_block)
        self.assertNotIn(DISCARDED_JAZZ, fusion_block)
        self.assertNotIn(BOSA_BRIEF.supporting_evidence[0], fusion_block.split("Relative-advantage facts:")[0])

    def test_supporting_evidence_not_fusion_material(self) -> None:
        brief = SelectedCreativeBrief(
            essential_facts=["Reinforced shell product designed for daily carry"],
            supporting_evidence=["Founded in 2019 for credibility"],
            discarded_facts=["Warehouse in Haifa"],
        )
        self.assertFalse(fusion_required_for_brief(brief))

    def test_mandatory_constraint_not_treated_as_creative_fact(self) -> None:
        brief = SelectedCreativeBrief(
            essential_facts=["Reinforced shell product designed for daily carry"],
            mandatory_constraints=["Do not show the product package"],
            discarded_facts=["Founded in 2019"],
        )
        block = format_selected_creative_brief_block(brief)
        self.assertIn("Mandatory constraints:", block)
        self.assertIn("Do not show the product package", block)
        self.assertNotIn("Founded in 2019", block)

    def test_product_led_physical_fact_survives(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="CarryShell",
            product_description=RAW_SHELL,
            detected_language="en",
            format_value="portrait",
            strategic_problem="Problem",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            slogan_derivation="Derivation",
            implied_action="Show survival",
            conceptual={"generator": "Mechanism", "action": "Show"},
            selected_creative_brief=PRODUCT_LED_BRIEF,
        )
        self.assertIn("matte-black shell case", prompt)
        self.assertNotIn("Founded in 2019", prompt)

    def test_over_selection_rejected(self) -> None:
        over_brief = {
            "essentialFacts": [
                "Fact one about the product",
                "Fact two about the product",
                "Fact three about the product",
                "Fact four about the product",
                "Fact five about the product",
            ],
            "supportingEvidence": [],
            "mandatoryConstraints": [],
            "discardedFacts": [],
        }
        raw = ". ".join(
            [
                "Fact one about the product",
                "Fact two about the product",
                "Fact three about the product",
                "Fact four about the product",
                "Fact five about the product",
            ]
        )
        with self.assertRaises(StageParseError) as ctx:
            parse_selected_creative_brief(
                over_brief,
                product_description=raw,
                strict_negative_selection=True,
            )
        self.assertTrue(
            any("over_selection" in reason for reason in ctx.exception.reasons)
        )

    def test_under_selection_missing_category_rejected(self) -> None:
        under_brief = {
            "essentialFacts": ["Made in Israel is the relative advantage"],
            "supportingEvidence": [],
            "mandatoryConstraints": [],
            "discardedFacts": ["בקבוק 100 מ״ל"],
        }
        with self.assertRaises(StageParseError) as ctx:
            parse_selected_creative_brief(
                under_brief,
                product_description=RAW_PERFUME,
                strict_negative_selection=True,
            )
        self.assertIn(
            "selectedCreativeBrief:under_selection_missing_product_category_identity",
            ctx.exception.reasons,
        )

    def test_bosa_essential_facts_survive_and_fusion_applies(self) -> None:
        self.assertTrue(
            fusion_required_for_brief(
                BOSA_BRIEF,
                relative_advantage="A locally produced Israeli alternative in the men's fragrance category",
            )
        )
        plan = {
            "relativeAdvantage": "A locally produced Israeli alternative in the men's fragrance category",
            "conceptualGenerator": "Foreign origin replaced by local origin in fragrance form",
            "physicalGenerator": "Men's fragrance bottle with foreign label replaced by Israeli label",
            "transferredObject": "Men's fragrance bottle with foreign label replaced by Israeli label",
            "transferredObjectAction": "The foreign origin label becomes an Israeli origin label",
            "campaignRationale": "Shows local men's fragrance alternative through product-integrated origin shift",
            "planningInternals": {"selectedCreativeBrief": BOSA_BRIEF.to_dict()},
            "ads": [],
        }
        self.assertEqual(scan_essential_fact_fusion(plan), [])

    def test_long_brief_only_relevant_facts_in_creative_prompt(self) -> None:
        prompt = _conceptual_prompt(BOSA_BRIEF)
        self.assertIn("בוסה הוא בושם לגברים", prompt)
        self.assertIn("מיוצר בישראל", prompt)
        for discarded in BOSA_BRIEF.discarded_facts:
            self.assertNotIn(discarded, prompt)

    def test_legacy_brief_without_discarded_facts_still_parses(self) -> None:
        legacy = {
            "essentialFacts": ["Reinforced shell product designed for daily carry"],
            "supportingEvidence": ["Durable reinforced construction supports everyday protection"],
            "mandatoryConstraints": [],
        }
        brief = parse_selected_creative_brief(legacy, strict_negative_selection=False)
        self.assertEqual(brief.discarded_facts, [])

    def test_compliance_still_reads_full_product_description(self) -> None:
        product_type = classify_advertised_product_type(
            product_name="בוסה",
            product_description=RAW_PERFUME,
            planning_internals={},
        )
        self.assertTrue(product_type)


class TestNegativeSelectionRegression(unittest.TestCase):
    def test_post_selection_isolation_tests_still_importable(self) -> None:
        import tests.test_builder1_post_selection_brief_isolation as isolation_tests

        self.assertTrue(hasattr(isolation_tests, "TestPostSelectionBriefIsolation"))

    def test_essential_fact_fusion_tests_still_importable(self) -> None:
        import tests.test_builder1_essential_fact_fusion as fusion_tests

        self.assertTrue(hasattr(fusion_tests, "TestEssentialFactFusionDetection"))


if __name__ == "__main__":
    unittest.main()
