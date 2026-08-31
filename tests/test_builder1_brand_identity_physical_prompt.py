"""
Builder1 brand_physical post-strategy brand guideline filter tests.

Run: python -m unittest tests.test_builder1_brand_identity_physical_prompt -v
"""
from __future__ import annotations

import copy
import inspect
import unittest
from typing import Any, Dict, List

from engine.builder1_brand_identity_guidelines import (
    PHYSICAL_BRAND_IDENTITY_ALLOWLIST,
    PHYSICAL_BRAND_RAW_BRIEF_DENYLIST,
    brand_identity_guidelines_for_physical_prompt,
)
from engine.builder1_no_logo import brand_guidelines_for_prompt, sanitize_brand_guidelines_for_builder1
from engine.builder1_object_design_integrity import BUILDER1_OBJECT_DESIGN_PALETTE_BOUNDARY
from engine.builder1_planning_contract import (
    STAGE_BRAND_PHYSICAL_SYSTEM,
    STAGE_SERIES_ADS_SYSTEM,
    build_brand_physical_user_prompt,
    build_product_name_resolution_user_prompt,
)
from engine.builder1_planning_metrics import (
    NORMAL_PLANNING_CALLS_WITH_GENERATED_NAME,
    NORMAL_PLANNING_CALLS_WITH_NAME,
)
from engine.builder1_planner import plan_builder1
from engine.builder1_product_shot_methodology import BUILDER1_PUBLIC_SIMPLICITY
from engine.builder1_selected_creative_brief import (
    SelectedCreativeBrief,
    default_selected_creative_brief_for_tests,
)
from engine.builder1_server_mandatory_constraints import (
    effective_creative_brief_for_prompts,
    extract_builder1_server_mandatory_constraints,
)
from engine.builder1_target_audience_methodology import TARGET_AUDIENCE_METHODOLOGY
from tests.test_builder1_staged_planning import (
    _full_final_responses,
)

RAW_DESCRIPTION = (
    "Reinforced shell product for daily carry. Secondary fact: also available in blue."
)
BYPASS_CREATIVE_BRIEF = "Also highlight our 10-year warranty bypass marker"
BYPASS_BRIEF = "Alternate campaign direction bypass marker"
BYPASS_NOTES = "Secondary advantage notes bypass marker"
BYPASS_UNKNOWN = "Unknown-key alternate advantage bypass marker"
INSTRUCTIONS_BLOB = "Do not show children. Extra instruction blob line."
USER_INSTRUCTIONS_BLOB = "The price must appear. Extra user instruction blob."


def _conceptual() -> Dict[str, str]:
    return {"generator": "Stress-test mechanism", "action": "Show impact survival visually"}


def _selected_brief() -> SelectedCreativeBrief:
    return SelectedCreativeBrief(
        essential_facts=["Reinforced shell for daily carry"],
        supporting_evidence=["Built for impact survival"],
        mandatory_constraints=[],
    )


def _physical_prompt(
    *,
    brand_guidelines: Dict[str, Any] | None = None,
    selected_creative_brief: SelectedCreativeBrief | None = None,
    product_description: str = RAW_DESCRIPTION,
) -> str:
    brief = selected_creative_brief if selected_creative_brief is not None else _selected_brief()
    return build_brand_physical_user_prompt(
        product_name_resolved="CarryShell",
        product_description=product_description,
        detected_language="en",
        format_value="portrait",
        strategic_problem="Problem",
        relative_advantage="Advantage",
        brand_slogan="Built To Last",
        slogan_derivation="Derivation",
        implied_action="Show impact survival",
        conceptual=_conceptual(),
        brand_guidelines=brand_guidelines,
        visibility_policy="CREATIVE_DECISION",
        selected_creative_brief=brief,
    )


class TestBrandIdentityGuidelineFilter(unittest.TestCase):
    def test_creative_brief_excluded_from_physical_prompt(self) -> None:
        prompt = _physical_prompt(
            brand_guidelines={"creativeBrief": BYPASS_CREATIVE_BRIEF, "primaryColor": "#111111"},
        )
        self.assertNotIn(BYPASS_CREATIVE_BRIEF, prompt)
        self.assertNotIn('"creativeBrief"', prompt)

    def test_brief_excluded_from_physical_prompt(self) -> None:
        prompt = _physical_prompt(brand_guidelines={"brief": BYPASS_BRIEF})
        self.assertNotIn(BYPASS_BRIEF, prompt)
        self.assertNotIn('"brief"', prompt)

    def test_notes_excluded_from_physical_prompt(self) -> None:
        prompt = _physical_prompt(brand_guidelines={"notes": BYPASS_NOTES})
        self.assertNotIn(BYPASS_NOTES, prompt)
        self.assertNotIn('"notes"', prompt)

    def test_instructions_blob_excluded_from_physical_prompt(self) -> None:
        prompt = _physical_prompt(brand_guidelines={"instructions": INSTRUCTIONS_BLOB})
        self.assertNotIn(INSTRUCTIONS_BLOB, prompt)
        self.assertNotIn('"instructions"', prompt)

    def test_user_instructions_blob_excluded_from_physical_prompt(self) -> None:
        prompt = _physical_prompt(brand_guidelines={"userInstructions": USER_INSTRUCTIONS_BLOB})
        self.assertNotIn(USER_INSTRUCTIONS_BLOB, prompt)
        self.assertNotIn('"userInstructions"', prompt)

    def test_instructions_reach_through_effective_mandatory_constraints(self) -> None:
        guidelines = {"instructions": "Do not show children."}
        server = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines=guidelines,
        )
        brief = effective_creative_brief_for_prompts(_selected_brief(), server)
        prompt = _physical_prompt(brand_guidelines=guidelines, selected_creative_brief=brief)
        self.assertIn("Do not show children.", prompt)
        self.assertNotIn('"instructions"', prompt)

    def test_user_instructions_reach_through_effective_mandatory_constraints(self) -> None:
        guidelines = {"userInstructions": "The price must appear."}
        server = extract_builder1_server_mandatory_constraints(
            product_description=RAW_DESCRIPTION,
            brand_guidelines=guidelines,
        )
        brief = effective_creative_brief_for_prompts(_selected_brief(), server)
        prompt = _physical_prompt(brand_guidelines=guidelines, selected_creative_brief=brief)
        self.assertIn("The price must appear.", prompt)
        self.assertNotIn('"userInstructions"', prompt)

    def test_primary_color_survives_in_visual_identity_block(self) -> None:
        prompt = _physical_prompt(brand_guidelines={"primaryColor": "#FF0000", "creativeBrief": BYPASS_CREATIVE_BRIEF})
        self.assertIn("BRAND VISUAL IDENTITY:", prompt)
        self.assertIn("primaryColor", prompt)
        self.assertIn("#FF0000", prompt)

    def test_accent_color_survives(self) -> None:
        prompt = _physical_prompt(brand_guidelines={"accentColor": "#00AAFF"})
        self.assertIn("accentColor", prompt)
        self.assertIn("#00AAFF", prompt)

    def test_palette_survives(self) -> None:
        palette = {"dominant": "#111111", "accent": "#FF5500"}
        prompt = _physical_prompt(brand_guidelines={"palette": palette})
        self.assertIn("palette", prompt)
        self.assertIn("#FF5500", prompt)

    def test_typography_and_tone_survive_when_supported(self) -> None:
        prompt = _physical_prompt(
            brand_guidelines={
                "typography": "Bold geometric sans",
                "tone": "confident",
                "visualTone": "editorial",
            },
        )
        self.assertIn("typography", prompt)
        self.assertIn("Bold geometric sans", prompt)
        self.assertIn("tone", prompt)
        self.assertIn("visualTone", prompt)

    def test_unknown_free_form_key_denied(self) -> None:
        prompt = _physical_prompt(
            brand_guidelines={
                "campaignHint": BYPASS_UNKNOWN,
                "primaryColor": "#222222",
            },
        )
        self.assertNotIn(BYPASS_UNKNOWN, prompt)
        self.assertNotIn("campaignHint", prompt)
        self.assertIn("primaryColor", prompt)

    def test_selected_creative_brief_remains_present(self) -> None:
        prompt = _physical_prompt()
        self.assertIn("Selected creative brief", prompt)
        self.assertIn("Reinforced shell for daily carry", prompt)

    def test_product_name_resolved_remains_present(self) -> None:
        prompt = _physical_prompt()
        self.assertIn("Fixed productNameResolved (echo exactly): CarryShell", prompt)

    def test_product_visibility_policy_remains_present(self) -> None:
        prompt = _physical_prompt()
        self.assertIn("Server product visibility policy: CREATIVE_DECISION", prompt)

    def test_frozen_strategy_relative_advantage_slogan_remain(self) -> None:
        prompt = _physical_prompt()
        self.assertIn("Fixed strategic problem: Problem", prompt)
        self.assertIn("Fixed relative advantage: Advantage", prompt)
        self.assertIn("Fixed brand slogan (do not change): Built To Last", prompt)

    def test_raw_product_description_absent_when_brief_present(self) -> None:
        prompt = _physical_prompt()
        self.assertNotIn(f"Description: {RAW_DESCRIPTION}", prompt)
        self.assertNotIn("also available in blue", prompt)

    def test_object_design_palette_boundary_unchanged_in_series_stage(self) -> None:
        self.assertIn("Do not recolor real-world objects merely to match the campaign palette", STAGE_SERIES_ADS_SYSTEM)
        self.assertIn("Campaign palette governs graphic design by default", BUILDER1_OBJECT_DESIGN_PALETTE_BOUNDARY)

    def test_physical_repair_uses_same_prompt_builder(self) -> None:
        from engine import builder1_physical_repair

        source = inspect.getsource(builder1_physical_repair)
        self.assertIn("build_brand_physical_user_prompt", source)
        repair_prompt = _physical_prompt(
            brand_guidelines={"creativeBrief": BYPASS_CREATIVE_BRIEF, "accentColor": "#ABCDEF"},
        )
        self.assertNotIn(BYPASS_CREATIVE_BRIEF, repair_prompt)
        self.assertIn("accentColor", repair_prompt)

    def test_product_name_resolution_still_receives_full_sanitized_guidelines(self) -> None:
        guidelines = {
            "creativeBrief": BYPASS_CREATIVE_BRIEF,
            "tone": "energetic",
            "logoUrl": "https://cdn.example.com/logo.png",
        }
        prompt = build_product_name_resolution_user_prompt(
            product_description=RAW_DESCRIPTION,
            detected_language="en",
            brand_guidelines=guidelines,
        )
        self.assertIn(BYPASS_CREATIVE_BRIEF, prompt)
        self.assertIn("tone", prompt)
        self.assertNotIn("logo", prompt.lower())

    def test_full_storage_sanitization_unchanged(self) -> None:
        guidelines = {
            "creativeBrief": BYPASS_CREATIVE_BRIEF,
            "primaryColor": "#FF0000",
            "logoUrl": "https://cdn.example.com/logo.png",
        }
        stored = sanitize_brand_guidelines_for_builder1(guidelines)
        assert stored is not None
        self.assertIn("creativeBrief", stored)
        self.assertIn("primaryColor", stored)
        self.assertNotIn("logoUrl", stored)

    def test_filter_helper_does_not_mutate_source(self) -> None:
        guidelines = {"primaryColor": "#FF0000", "creativeBrief": BYPASS_CREATIVE_BRIEF}
        original = copy.deepcopy(guidelines)
        filtered = brand_identity_guidelines_for_physical_prompt(guidelines)
        self.assertEqual(guidelines, original)
        assert filtered is not None
        self.assertNotIn("creativeBrief", filtered)

    def test_public_simplicity_unchanged(self) -> None:
        self.assertIn(BUILDER1_PUBLIC_SIMPLICITY, TARGET_AUDIENCE_METHODOLOGY)
        self.assertIn(TARGET_AUDIENCE_METHODOLOGY, STAGE_BRAND_PHYSICAL_SYSTEM)

    def test_brand_guidelines_for_prompt_unchanged_for_other_callers(self) -> None:
        guidelines = {"creativeBrief": BYPASS_CREATIVE_BRIEF, "tone": "bold"}
        safe = brand_guidelines_for_prompt(guidelines)
        assert safe is not None
        self.assertIn("creativeBrief", safe)
        self.assertIn("tone", safe)


class TestBrandIdentityPlanningCallCounts(unittest.TestCase):
    def test_supplied_name_call_count_remains_five(self) -> None:
        from engine.builder1_planning_contract import STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM

        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            responses = copy.deepcopy(_full_final_responses(2))
            if stage == "strategy_slogan_stage":
                payload = copy.deepcopy(responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM])
                payload["strategy"]["selectedCreativeBrief"]["mandatoryConstraints"] = []
                return payload
            return responses.get(system, {})

        plan_builder1(
            product_name="CarryShell",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
            brand_guidelines={
                "creativeBrief": BYPASS_CREATIVE_BRIEF,
                "primaryColor": "#111111",
            },
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
        from engine.builder1_planning_contract import STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM

        stages: List[str] = []

        def model_caller(system: str, user: str, stage: str | None = None) -> object:
            if stage:
                stages.append(stage)
            responses = copy.deepcopy(_full_final_responses(2))
            if stage == "strategy_slogan_stage":
                payload = copy.deepcopy(responses[STAGE_STRATEGY_SLOGAN_STAGE_SYSTEM])
                payload["strategy"]["selectedCreativeBrief"]["mandatoryConstraints"] = []
                return payload
            return responses.get(system, {})

        plan_builder1(
            product_name="",
            product_description=RAW_DESCRIPTION,
            format_value="portrait",
            model_caller=model_caller,
            ad_count=2,
            brand_guidelines={"creativeBrief": BYPASS_CREATIVE_BRIEF},
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


class TestBrandIdentityAllowDenyContracts(unittest.TestCase):
    def test_denylist_contains_required_raw_brief_keys(self) -> None:
        for key in ("creativeBrief", "brief", "notes", "instructions", "userInstructions"):
            normalized = "".join(ch for ch in key.casefold() if ch.isalnum())
            self.assertIn(normalized, PHYSICAL_BRAND_RAW_BRIEF_DENYLIST)

    def test_allowlist_contains_supported_visual_keys(self) -> None:
        for key in ("primaryColor", "accentColor", "palette", "tone", "typography"):
            normalized = "".join(ch for ch in key.casefold() if ch.isalnum())
            self.assertIn(normalized, PHYSICAL_BRAND_IDENTITY_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()
