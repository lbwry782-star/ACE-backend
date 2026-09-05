"""
Builder1 post-selection brief isolation regression tests.

Run: python -m unittest tests.test_builder1_post_selection_brief_isolation -v
"""
from __future__ import annotations

import unittest

from engine.builder1_compliance_product_grounding import classify_advertised_product_type
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_CONCEPTUAL_STAGE_SYSTEM,
    STAGE_GRAPHIC_SYSTEM_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM,
    build_brand_physical_user_prompt,
    build_conceptual_stage_user_prompt,
    build_graphic_system_user_prompt,
    build_series_ads_user_prompt,
    build_strategy_slogan_stage_user_prompt,
)
from engine.builder1_post_selection_brief_isolation import (
    POST_SELECTION_BRIEF_ISOLATION,
    raw_product_description_visible_in_prompt,
)
from engine.builder1_selected_creative_brief import SelectedCreativeBrief
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _selected_slogan,
    _selected_strategy,
)


RAW_DESCRIPTION = (
    "Reinforced shell product for daily carry. Secondary fact: also available in blue. "
    "Unrelated advantage: waterproof coating option."
)
UNSELECTED_SECONDARY = "waterproof coating option"
SELECTED_ESSENTIAL = "Reinforced shell product designed for daily carry"
MANDATORY = "Do not show the product package in frame"

BRIEF = SelectedCreativeBrief(
    essential_facts=[SELECTED_ESSENTIAL],
    supporting_evidence=["Durable reinforced construction supports everyday protection"],
    mandatory_constraints=[MANDATORY],
)

PRODUCT_LED_BRIEF = SelectedCreativeBrief(
    essential_facts=[
        "CarryShell is a reinforced matte-black shell case with visible hinge hardware",
        SELECTED_ESSENTIAL,
    ],
    supporting_evidence=["Product form is immediately readable on shelf"],
    mandatory_constraints=[],
)


class TestPostSelectionBriefIsolation(unittest.TestCase):
    def test_strategy_stage_still_receives_full_raw_description(self) -> None:
        prompt = build_strategy_slogan_stage_user_prompt(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            lens_order=["economic"],
            exploration_seed="seed-1",
        )
        self.assertIn(RAW_DESCRIPTION, prompt)
        self.assertIn("Product description (raw information", prompt)
        self.assertIn("RAW INFORMATION", STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM)

    def test_conceptual_stage_excludes_raw_after_selection(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        prompt = build_conceptual_stage_user_prompt(
            product_description=RAW_DESCRIPTION,
            product_name_resolved="CarryShell",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            slogan_derivation=slogan.derivation_from_advantage,
            implied_action=slogan.implied_action,
            exploration_seed="seed-1",
            selected_creative_brief=BRIEF,
        )
        self.assertFalse(raw_product_description_visible_in_prompt(prompt, RAW_DESCRIPTION))
        self.assertNotIn(UNSELECTED_SECONDARY, prompt)

    def test_conceptual_stage_receives_selected_creative_brief(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        prompt = build_conceptual_stage_user_prompt(
            product_description=RAW_DESCRIPTION,
            product_name_resolved="CarryShell",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            slogan_derivation=slogan.derivation_from_advantage,
            implied_action=slogan.implied_action,
            exploration_seed="seed-1",
            selected_creative_brief=BRIEF,
        )
        self.assertIn("Selected creative brief", prompt)
        self.assertIn(SELECTED_ESSENTIAL, prompt)
        self.assertIn(POST_SELECTION_BRIEF_ISOLATION, STAGE_CONCEPTUAL_STAGE_SYSTEM)

    def test_brand_physical_excludes_unrestricted_raw(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="CarryShell",
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            format_value="portrait",
            strategic_problem="Problem",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            slogan_derivation="Derivation",
            implied_action="Show impact survival",
            conceptual={"generator": "Mechanism", "action": "Show survival"},
            selected_creative_brief=BRIEF,
        )
        self.assertFalse(raw_product_description_visible_in_prompt(prompt, RAW_DESCRIPTION))
        self.assertNotIn(UNSELECTED_SECONDARY, prompt)

    def test_brand_physical_receives_selected_product_facts(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="CarryShell",
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            format_value="portrait",
            strategic_problem="Problem",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            slogan_derivation="Derivation",
            implied_action="Show impact survival",
            conceptual={"generator": "Mechanism", "action": "Show survival"},
            selected_creative_brief=BRIEF,
        )
        self.assertIn(SELECTED_ESSENTIAL, prompt)
        self.assertIn("Fixed productNameResolved", prompt)
        self.assertIn(POST_SELECTION_BRIEF_ISOLATION, STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_graphic_system_excludes_raw_product_description(self) -> None:
        prompt = build_graphic_system_user_prompt(
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            conceptual={"generator": "Mechanism"},
            brand_physical=_brand_physical(),
            format_value="portrait",
            selected_creative_brief=BRIEF,
        )
        self.assertFalse(raw_product_description_visible_in_prompt(prompt, RAW_DESCRIPTION))
        self.assertNotIn(UNSELECTED_SECONDARY, prompt)

    def test_graphic_system_receives_brief_and_upstream_state(self) -> None:
        conceptual = {"generator": "Stress-test mechanism", "action": "Show survival"}
        brand_physical = _brand_physical()
        prompt = build_graphic_system_user_prompt(
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            conceptual=conceptual,
            brand_physical=brand_physical,
            format_value="portrait",
            selected_creative_brief=BRIEF,
        )
        self.assertIn(SELECTED_ESSENTIAL, prompt)
        self.assertIn("Conceptual generator", prompt)
        self.assertIn("Physical system", prompt)
        self.assertIn(POST_SELECTION_BRIEF_ISOLATION, STAGE_GRAPHIC_SYSTEM_SYSTEM)

    def test_series_ads_remains_isolated(self) -> None:
        prompt = build_series_ads_user_prompt(
            ad_count=2,
            format_value="portrait",
            detected_language="en",
            strategic_problem="Problem",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            implied_action="Show impact survival",
            conceptual={"generator": "Mechanism"},
            brand_physical=_brand_physical(),
            graphic_generator={"layoutTemplate": "visual_right_copy_left"},
            visibility_policy="CREATIVE_DECISION",
        )
        self.assertNotIn(RAW_DESCRIPTION, prompt)
        self.assertNotIn("Product description", prompt)
        self.assertNotIn(POST_SELECTION_BRIEF_ISOLATION, STAGE_SERIES_ADS_SYSTEM)

    def test_mandatory_constraints_survive_downstream(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        conceptual_prompt = build_conceptual_stage_user_prompt(
            product_description=RAW_DESCRIPTION,
            product_name_resolved="CarryShell",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            slogan_derivation=slogan.derivation_from_advantage,
            implied_action=slogan.implied_action,
            exploration_seed="seed-1",
            selected_creative_brief=BRIEF,
        )
        graphic_prompt = build_graphic_system_user_prompt(
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            conceptual={"generator": "Mechanism"},
            brand_physical=_brand_physical(),
            format_value="portrait",
            selected_creative_brief=BRIEF,
        )
        self.assertIn(MANDATORY, conceptual_prompt)
        self.assertIn(MANDATORY, graphic_prompt)

    def test_unselected_secondary_fact_not_visible_in_creative_prompts(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        prompts = [
            build_conceptual_stage_user_prompt(
                product_description=RAW_DESCRIPTION,
                product_name_resolved="CarryShell",
                strategic_problem=strategy.strategic_problem,
                relative_advantage=strategy.relative_advantage,
                brand_slogan=slogan.brand_slogan,
                slogan_derivation=slogan.derivation_from_advantage,
                implied_action=slogan.implied_action,
                exploration_seed="seed-1",
                selected_creative_brief=BRIEF,
            ),
            build_brand_physical_user_prompt(
                product_name_resolved="CarryShell",
                product_description=RAW_DESCRIPTION,
                detected_language="en",
                format_value="portrait",
                strategic_problem=strategy.strategic_problem,
                relative_advantage=strategy.relative_advantage,
                brand_slogan=slogan.brand_slogan,
                slogan_derivation=slogan.derivation_from_advantage,
                implied_action=slogan.implied_action,
                conceptual={"generator": "Mechanism", "action": "Show survival"},
                selected_creative_brief=BRIEF,
            ),
            build_graphic_system_user_prompt(
                product_description=RAW_DESCRIPTION,
                detected_language="en",
                relative_advantage=strategy.relative_advantage,
                brand_slogan=slogan.brand_slogan,
                conceptual={"generator": "Mechanism"},
                brand_physical=_brand_physical(),
                format_value="portrait",
                selected_creative_brief=BRIEF,
            ),
        ]
        for prompt in prompts:
            self.assertNotIn(UNSELECTED_SECONDARY, prompt)

    def test_selected_essential_fact_visible_downstream(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        prompts = [
            build_conceptual_stage_user_prompt(
                product_description=RAW_DESCRIPTION,
                product_name_resolved="CarryShell",
                strategic_problem=strategy.strategic_problem,
                relative_advantage=strategy.relative_advantage,
                brand_slogan=slogan.brand_slogan,
                slogan_derivation=slogan.derivation_from_advantage,
                implied_action=slogan.implied_action,
                exploration_seed="seed-1",
                selected_creative_brief=BRIEF,
            ),
            build_brand_physical_user_prompt(
                product_name_resolved="CarryShell",
                product_description=RAW_DESCRIPTION,
                detected_language="en",
                format_value="portrait",
                strategic_problem=strategy.strategic_problem,
                relative_advantage=strategy.relative_advantage,
                brand_slogan=slogan.brand_slogan,
                slogan_derivation=slogan.derivation_from_advantage,
                implied_action=slogan.implied_action,
                conceptual={"generator": "Mechanism", "action": "Show survival"},
                selected_creative_brief=BRIEF,
            ),
            build_graphic_system_user_prompt(
                product_description=RAW_DESCRIPTION,
                detected_language="en",
                relative_advantage=strategy.relative_advantage,
                brand_slogan=slogan.brand_slogan,
                conceptual={"generator": "Mechanism"},
                brand_physical=_brand_physical(),
                format_value="portrait",
                selected_creative_brief=BRIEF,
            ),
        ]
        for prompt in prompts:
            self.assertIn(SELECTED_ESSENTIAL, prompt)

    def test_product_led_retains_selected_physical_facts(self) -> None:
        prompt = build_brand_physical_user_prompt(
            product_name_resolved="CarryShell",
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            format_value="portrait",
            strategic_problem="Problem",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            slogan_derivation="Derivation",
            implied_action="Show impact survival",
            conceptual={"generator": "Mechanism", "action": "Show survival"},
            selected_creative_brief=PRODUCT_LED_BRIEF,
        )
        self.assertIn("reinforced matte-black shell case", prompt)
        self.assertNotIn(UNSELECTED_SECONDARY, prompt)
        self.assertFalse(raw_product_description_visible_in_prompt(prompt, RAW_DESCRIPTION))

    def test_compliance_still_reads_full_product_description(self) -> None:
        product_type = classify_advertised_product_type(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            planning_internals={},
        )
        self.assertTrue(product_type)

    def test_legacy_fallback_when_brief_absent(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        prompt = build_conceptual_stage_user_prompt(
            product_description=RAW_DESCRIPTION,
            product_name_resolved="CarryShell",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            slogan_derivation=slogan.derivation_from_advantage,
            implied_action=slogan.implied_action,
            exploration_seed="seed-1",
            selected_creative_brief=None,
        )
        self.assertIn(RAW_DESCRIPTION, prompt)


if __name__ == "__main__":
    unittest.main()
