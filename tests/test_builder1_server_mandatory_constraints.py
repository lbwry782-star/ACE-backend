"""
Builder1 server-owned mandatory user constraints tests.

Run: python -m unittest tests.test_builder1_server_mandatory_constraints -v
"""
from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from typing import Any, Dict, List
from unittest.mock import patch

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
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    derive_product_visibility_policy,
)
from engine.builder1_selected_creative_brief import (
    SelectedCreativeBrief,
    default_selected_creative_brief_for_tests,
)
from engine.builder1_server_mandatory_constraints import (
    effective_creative_brief_for_prompts,
    effective_mandatory_constraints_for_brief,
    extract_builder1_server_mandatory_constraints,
    merge_effective_mandatory_constraints,
)
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _full_final_responses,
    _selected_slogan,
    _selected_strategy,
)

RAW_DESCRIPTION = (
    "Reinforced shell product for daily carry. Secondary fact: also available in blue."
)


class TestServerMandatoryExtraction(unittest.TestCase):
    def test_instructions_field_preserved(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines={"instructions": "Do not show children."},
        )
        self.assertIn("Do not show children.", constraints)

    def test_user_instructions_field_preserved(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines={"userInstructions": "The price must appear."},
        )
        self.assertIn("The price must appear.", constraints)

    def test_hebrew_directive_preserved(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines={"instructions": "אסור להראות ילדים בפרסומת"},
        )
        self.assertTrue(any("ילדים" in item for item in constraints))

    def test_product_description_price_directive(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=f"{RAW_DESCRIPTION} חובה להציג את המחיר.",
            brand_guidelines=None,
        )
        self.assertTrue(any("מחיר" in item for item in constraints))

    def test_factual_negative_not_extracted_from_description(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description="We don't manufacture plastic chairs.",
            brand_guidelines=None,
        )
        self.assertEqual(constraints, [])

    def test_hebrew_factual_negative_not_extracted(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description="המוצר אינו כולל ילדים",
            brand_guidelines=None,
        )
        self.assertEqual(constraints, [])

    def test_creative_brief_descriptive_text_not_whole_mandatory(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines={
                "creativeBrief": "Premium reinforced shell for commuters with long battery life.",
            },
        )
        self.assertEqual(constraints, [])

    def test_creative_brief_explicit_directive_extracted(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines={
                "creativeBrief": "Premium product. Do not use cars in the visual.",
            },
        )
        self.assertTrue(any("cars" in item.lower() for item in constraints))

    def test_visibility_directive_not_duplicated_in_server_constraints(self) -> None:
        constraints = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines={"instructions": "Do not show the product in the ad."},
        )
        self.assertEqual(constraints, [])
        decision = derive_product_visibility_policy(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            brand_guidelines={"instructions": "Do not show the product in the ad."},
        )
        self.assertEqual(decision.policy, ProductVisibilityPolicy.FORBIDDEN)


class TestEffectiveConstraintMerge(unittest.TestCase):
    def test_server_constraints_survive_empty_model_constraints(self) -> None:
        merged = merge_effective_mandatory_constraints(
            ["Do not show children."],
            [],
        )
        self.assertEqual(merged, ["Do not show children."])

    def test_model_may_add_additional_constraints(self) -> None:
        merged = merge_effective_mandatory_constraints(
            ["Do not show children."],
            ["Keep typography minimal."],
        )
        self.assertEqual(len(merged), 2)

    def test_duplicate_constraint_deduped(self) -> None:
        merged = merge_effective_mandatory_constraints(
            ["Do not show children."],
            ["Do not show children."],
        )
        self.assertEqual(len(merged), 1)

    def test_effective_brief_includes_server_constraints(self) -> None:
        base = default_selected_creative_brief_for_tests()
        effective = effective_creative_brief_for_prompts(
            base,
            ["Do not show children."],
        )
        assert effective is not None
        self.assertIn("Do not show children.", effective.mandatory_constraints)


class TestPromptInjection(unittest.TestCase):
    def test_strategy_prompt_receives_server_constraints(self) -> None:
        prompt = build_strategy_slogan_stage_user_prompt(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            lens_order=["economic"],
            exploration_seed="seed",
            server_mandatory_constraints=["Do not show children."],
            visibility_policy="CREATIVE_DECISION",
        )
        self.assertIn("Do not show children.", prompt)
        self.assertIn("Server mandatory user constraints", prompt)

    def test_conceptual_prompt_uses_effective_constraints_not_raw_description(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        brief = effective_creative_brief_for_prompts(
            default_selected_creative_brief_for_tests(),
            ["Do not show children."],
        )
        prompt = build_conceptual_stage_user_prompt(
            product_description=f"{RAW_DESCRIPTION} Unrelated waterproof coating.",
            product_name_resolved="CarryShell",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            slogan_derivation=slogan.derivation_from_advantage,
            implied_action=slogan.implied_action,
            exploration_seed="seed",
            selected_creative_brief=brief,
        )
        self.assertIn("Do not show children.", prompt)
        self.assertNotIn("waterproof coating", prompt)
        self.assertNotIn(f"Brief: {RAW_DESCRIPTION}", prompt)

    def test_series_ads_receives_mandatory_block_not_raw_description(self) -> None:
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
            effective_mandatory_constraints=["Do not show children."],
        )
        self.assertIn("Do not show children.", prompt)
        self.assertNotIn(RAW_DESCRIPTION, prompt)

    def test_graphic_prompt_still_narrow(self) -> None:
        brief = effective_creative_brief_for_prompts(
            default_selected_creative_brief_for_tests(),
            ["Do not show children."],
        )
        prompt = build_graphic_system_user_prompt(
            product_description=f"{RAW_DESCRIPTION} extra noise",
            detected_language="en",
            relative_advantage="Advantage",
            brand_slogan="Built To Last",
            conceptual={"generator": "Mechanism"},
            brand_physical=_brand_physical(),
            format_value="portrait",
            selected_creative_brief=brief,
        )
        self.assertIn("Do not show children.", prompt)
        self.assertNotIn("extra noise", prompt)
        self.assertNotIn(f"Brief: {RAW_DESCRIPTION}", prompt)


class TestEndToEndPlanning(unittest.TestCase):
    def _model_caller(self, stages: List[str]):
        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            responses = copy.deepcopy(_full_final_responses(2))
            if stage == "strategy_slogan_stage":
                payload = copy.deepcopy(responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM])
                payload["strategy"]["selectedCreativeBrief"]["mandatoryConstraints"] = []
                return payload
            return responses.get(system, {})

        return model_caller

    def test_plan_persists_server_constraints_when_instructions_present(self) -> None:
        stages: List[str] = []
        plan = plan_builder1(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=self._model_caller(stages),
            ad_count=2,
            brand_guidelines={"instructions": "Do not show children."},
        )
        internals = plan.planning_internals or {}
        server = internals.get("serverMandatoryConstraints") or []
        self.assertIn("Do not show children.", server)
        brief = internals.get("selectedCreativeBrief") or {}
        self.assertEqual(brief.get("mandatoryConstraints"), [])

    def test_supplied_name_call_count_remains_five(self) -> None:
        stages: List[str] = []
        plan_builder1(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=self._model_caller(stages),
            ad_count=2,
            brand_guidelines={"instructions": "Do not show children."},
        )
        paid = {
            "strategy_slogan_stage",
            "conceptual_stage",
            "brand_physical",
            "graphic_system",
            "series_ads",
        }
        self.assertEqual(len([s for s in stages if s in paid]), NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_call_count_remains_six(self) -> None:
        stages: List[str] = []
        plan_builder1(
            product_name="",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=self._model_caller(stages),
            ad_count=2,
            brand_guidelines={"instructions": "Do not show children."},
        )
        paid = {
            "product_name_resolution",
            "strategy_slogan_stage",
            "conceptual_stage",
            "brand_physical",
            "graphic_system",
            "series_ads",
        }
        self.assertEqual(
            len([s for s in stages if s in paid]),
            NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
        )


if __name__ == "__main__":
    unittest.main()
