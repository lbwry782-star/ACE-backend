"""
Builder1 Target Audience ownership + selectedCreativeBrief regression tests.

Run: python -m unittest tests.test_builder1_target_audience_selected_brief -v
"""
from __future__ import annotations

import copy
import inspect
import unittest
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
    get_planning_metrics,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_product_shot_methodology import BUILDER1_PUBLIC_SIMPLICITY
from engine.builder1_compliance_product_grounding import classify_advertised_product_type
from engine.builder1_selected_creative_brief import (
    default_selected_creative_brief_for_tests,
    parse_selected_creative_brief,
    selected_creative_brief_from_plan,
)
from engine.builder1_target_audience_methodology import (
    TARGET_AUDIENCE_DECODING_MODE,
    TARGET_AUDIENCE_METHODOLOGY,
)
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _full_final_responses,
    _selected_slogan,
    _selected_strategy,
    _strategy_slogan_stage_payload,
)

RAW_DESCRIPTION = (
    "Reinforced shell product for daily carry. Secondary fact: also available in blue. "
    "Unrelated advantage: waterproof coating option."
)
BRIEF = default_selected_creative_brief_for_tests()


class TestTargetAudienceOwnership(unittest.TestCase):
    def test_methodology_wraps_public_simplicity(self) -> None:
        self.assertIn("TARGET AUDIENCE", TARGET_AUDIENCE_METHODOLOGY)
        self.assertIn(BUILDER1_PUBLIC_SIMPLICITY, TARGET_AUDIENCE_METHODOLOGY)
        self.assertIn("THE AUDIENCE MAY BE SOPHISTICATED", TARGET_AUDIENCE_METHODOLOGY)
        self.assertEqual(TARGET_AUDIENCE_DECODING_MODE, "UNIVERSAL_SIMPLE_DECODING")

    def test_brand_physical_and_series_ads_prompts_use_target_audience_block(self) -> None:
        for system_prompt in (STAGE_BRAND_PHYSICAL_SYSTEM, STAGE_SERIES_ADS_SYSTEM):
            self.assertIn("TARGET AUDIENCE", system_prompt)
            self.assertIn("PUBLIC SIMPLICITY", system_prompt)

    def test_no_demographic_target_audience_field_on_plan(self) -> None:
        captured: dict[str, Any] = {}

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        import engine.builder1_planning_metrics as metrics_module

        real_reset = metrics_module.reset_planning_metrics

        def capture_reset(token) -> None:
            metrics = get_planning_metrics()
            if metrics is not None:
                captured["metrics"] = metrics
            real_reset(token)

        with patch("engine.builder1_planner.reset_planning_metrics", side_effect=capture_reset):
            plan = plan_builder1(
                product_name="CarryShell",
                product_description=RAW_DESCRIPTION,
                format_value="portrait",
                model_caller=model_caller,
                ad_count=2,
            )
        self.assertNotIn("targetAudience", plan.__dict__)
        internals = plan.planning_internals or {}
        self.assertEqual(internals.get("targetAudienceDecodingMode"), TARGET_AUDIENCE_DECODING_MODE)
        self.assertNotIn("targetAudience", internals)


class TestSelectedCreativeBriefPromptIsolation(unittest.TestCase):
    def test_strategy_stage_receives_full_raw_description(self) -> None:
        prompt = build_strategy_slogan_stage_user_prompt(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            lens_order=["economic"],
            exploration_seed="seed-1",
        )
        self.assertIn(RAW_DESCRIPTION, prompt)
        self.assertIn("RAW INFORMATION", STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM)

    def test_conceptual_stage_uses_brief_not_full_raw(self) -> None:
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
        self.assertNotIn("waterproof coating option", prompt)
        self.assertNotIn(f"Brief: {RAW_DESCRIPTION}", prompt)

    def test_conceptual_legacy_fallback_uses_full_description(self) -> None:
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
        self.assertIn(f"Brief: {RAW_DESCRIPTION}", prompt)

    def test_brand_physical_uses_identity_and_brief(self) -> None:
        conceptual = {
            "generator": "Stress-test mechanism",
            "action": "Show everyday impact survival visually",
        }
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
            conceptual=conceptual,
            selected_creative_brief=BRIEF,
        )
        self.assertIn("Selected creative brief", prompt)
        self.assertNotIn("waterproof coating option", prompt)
        self.assertNotIn(f"Description: {RAW_DESCRIPTION}", prompt)

    def test_graphic_system_excludes_full_raw_when_brief_present(self) -> None:
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
        self.assertNotIn(f"Brief: {RAW_DESCRIPTION}", prompt)
        self.assertNotIn("waterproof coating option", prompt)

    def test_series_ads_prompt_never_includes_full_product_description(self) -> None:
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


class TestSelectedCreativeBriefPersistence(unittest.TestCase):
    def test_strategy_payload_requires_and_parses_brief(self) -> None:
        payload = _strategy_slogan_stage_payload()
        brief = parse_selected_creative_brief(payload["strategy"]["selectedCreativeBrief"])
        self.assertTrue(brief.essential_facts)

    def test_plan_persists_selected_creative_brief(self) -> None:
        plan = plan_builder1(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=lambda system, user, stage=None: copy.deepcopy(
                _full_final_responses(2).get(system, {})
            ),
            ad_count=2,
        )
        brief = selected_creative_brief_from_plan(plan)
        self.assertIsNotNone(brief)
        assert brief is not None
        self.assertTrue(brief.essential_facts)
        self.assertEqual(plan.product_description, RAW_DESCRIPTION)

    def test_legacy_plan_without_brief_still_builds_prompts(self) -> None:
        strategy = _selected_strategy()
        slogan = _selected_slogan()
        prompt = build_graphic_system_user_prompt(
            product_description="Legacy full description",
            detected_language="en",
            relative_advantage=strategy.relative_advantage,
            brand_slogan=slogan.brand_slogan,
            conceptual={"generator": "Mechanism"},
            brand_physical=_brand_physical(),
            format_value="portrait",
            selected_creative_brief=None,
        )
        self.assertIn("Brief: Legacy full description", prompt)
        self.assertIsNone(selected_creative_brief_from_plan({"planningInternals": {}}))


class TestPlanningCallCounts(unittest.TestCase):
    def test_supplied_name_remains_five_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
        )
        paid = {
            "strategy_slogan_stage",
            "conceptual_stage",
            "brand_physical",
            "graphic_system",
            "series_ads",
        }
        self.assertEqual(len([s for s in stages if s in paid]), NORMAL_PLANNING_CALLS_WITH_NAME)

    def test_generated_name_remains_six_calls(self) -> None:
        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            return copy.deepcopy(_full_final_responses(2).get(system, {}))

        plan_builder1(
            product_name="",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
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


class TestFactualGroundingPreserved(unittest.TestCase):
    def test_compliance_still_reads_full_product_description(self) -> None:
        product_type = classify_advertised_product_type(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            planning_internals={},
        )
        self.assertTrue(product_type)

    def test_no_web_search_in_strategy_stage_source(self) -> None:
        from engine.builder1_consolidated_stages import run_strategy_slogan_stage

        source = inspect.getsource(run_strategy_slogan_stage)
        self.assertNotIn("web_search", source)
        self.assertNotIn("tools=", source)


class TestPublicSimplicityUnchanged(unittest.TestCase):
    def test_two_sentence_test_still_active(self) -> None:
        from engine.builder1_advertising_comprehension import _passes_two_sentence_test

        ok = _passes_two_sentence_test(
            immediate="A magnet pulls metal toward it",
            bridge="That pull shows the product attracts customers reliably",
            relative_advantage="Attracts customers reliably",
            execution_blob="magnet and metal",
        )
        self.assertTrue(ok)

    def test_conceptual_system_not_given_target_audience_block(self) -> None:
        self.assertNotIn("TARGET AUDIENCE — CURRENT RESPONSIBILITY", STAGE_CONCEPTUAL_STAGE_SYSTEM)
        self.assertNotIn("TARGET AUDIENCE — CURRENT RESPONSIBILITY", STAGE_GRAPHIC_SYSTEM_SYSTEM)


if __name__ == "__main__":
    unittest.main()
