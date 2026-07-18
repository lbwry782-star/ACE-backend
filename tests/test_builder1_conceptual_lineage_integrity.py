"""
Builder1 conceptual lineage and final-integrity tests.

Run: python -m unittest tests.test_builder1_conceptual_lineage_integrity -v
"""
from __future__ import annotations

import copy
import inspect
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder1_campaign_integrity import (
    make_upstream_snapshot,
    validate_builder1_campaign_integrity,
)
from engine.builder1_creative_methodology import (
    deterministic_builder1_integrity_checks,
    deterministic_methodology_checks,
)
from engine.builder1_final_stages import (
    assemble_builder1_campaign,
    parse_brand_physical_output,
    parse_graphic_system_output,
    parse_series_ads_output,
)
from engine.builder1_plan_spec import series_plan_to_store_dict
from engine.builder1_slogan_stage import parse_slogan_scan, parse_slogan_selection
from engine.builder1_staged_parsers import detect_brief_language, parse_strategy_scan, parse_strategy_selection
from tests.test_builder1_staged_planning import (
    _brand_physical,
    _graphic,
    _internal_ad_fields,
    _selected_conceptual,
    _selected_strategy,
    _series_ads,
    _slogan_scan_payload,
    _strategy_scan_payload,
    _strategy_selection_payload,
)

FIXED_SLOGAN = "Built To Last"
FIXED_ACTION = "Show everyday impact survival visually"
BRIEF = "Reinforced shell product for daily carry"


def _selected_slogan():
    candidates = parse_slogan_scan(_slogan_scan_payload())
    _, selected = parse_slogan_selection(
        {
            "selectedCandidateId": "L01",
            "selectionReason": "Best",
            "scores": {
                "directAdvantageExpression": 9,
                "naturalness": 8,
                "memorability": 8,
                "credibility": 9,
                "brandOwnership": 8,
                "competitorTransferResistance": 8,
                "actionClarity": 9,
                "campaignGenerativePower": 9,
            },
        },
        candidates,
    )
    return selected


def _assemble(*, ad_count: int = 2, series_payload: Dict[str, Any] | None = None):
    strategy = _selected_strategy()
    selected_slogan = _selected_slogan()
    conceptual = _selected_conceptual()
    series = parse_series_ads_output(
        series_payload or _series_ads(ad_count),
        expected_ad_count=ad_count,
    )
    return assemble_builder1_campaign(
        product_name="",
        product_description=BRIEF,
        format_value="portrait",
        ad_count=ad_count,
        detected_language=detect_brief_language(BRIEF),
        exploration_seed="seed-test",
        product_name_resolved="TestBrand",
        strategy=strategy,
        strategy_selection=parse_strategy_selection(
            _strategy_selection_payload(selected_id=strategy.id),
            parse_strategy_scan(_strategy_scan_payload(), product_description=BRIEF),
        )[0],
        selected_slogan=selected_slogan,
        conceptual=conceptual,
        brand_physical=parse_brand_physical_output(_brand_physical()),
        graphic=parse_graphic_system_output(_graphic()),
        series_ads=series,
    )


def _upstream_for_plan(plan):
    return make_upstream_snapshot(
        product_name_resolved=plan.product_name_resolved,
        selected_strategy=_selected_strategy(),
        selected_slogan=_selected_slogan(),
        selected_conceptual=_selected_conceptual(),
        brand_physical=parse_brand_physical_output(_brand_physical()),
        graphic=parse_graphic_system_output(_graphic()),
    )


class TestRootCause(unittest.TestCase):
    def test_legacy_check_used_lexical_overlap_without_why_field(self) -> None:
        source = inspect.getsource(deterministic_methodology_checks)
        self.assertIn("conceptual_generator_not_derived_from_slogan", source)
        self.assertIn("action_tokens", source)
        self.assertIn("concept_tokens", source)

    def test_integrity_path_excludes_semantic_concept_derivation(self) -> None:
        plan_dict = {
            "brandSlogan": FIXED_SLOGAN,
            "sloganAction": "Animate gentle motion across surfaces",
            "conceptualGeneratorAction": "Reveal durability through escalating scenes",
            "detectedLanguage": "en",
            "ads": [],
        }
        legacy = deterministic_methodology_checks(plan_dict)
        integrity = deterministic_builder1_integrity_checks(plan_dict)
        self.assertIn("conceptual_generator_not_derived_from_slogan", legacy)
        self.assertNotIn("conceptual_generator_not_derived_from_slogan", integrity)

    def test_paraphrased_concept_passes_final_integrity(self) -> None:
        plan = _assemble(ad_count=2)
        plan.planning_internals["conceptualGeneratorWhyItExpressesSlogan"] = (
            "The concept proves durability without repeating the slogan literally"
        )
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertTrue(result.ok, msg=str(result.reasons))
        self.assertNotIn("conceptual_generator_not_derived_from_slogan", result.reasons)


class TestStructuralLineage(unittest.TestCase):
    def test_assembly_records_conceptual_lineage(self) -> None:
        plan = _assemble(ad_count=2)
        lineage = plan.planning_internals.get("conceptualLineage")
        assert isinstance(lineage, dict)
        self.assertEqual(lineage["sourceSloganCandidateId"], "L01")
        self.assertEqual(lineage["selectedConceptCandidateId"], "C01")
        self.assertEqual(lineage["fixedBrandSlogan"], FIXED_SLOGAN)
        self.assertEqual(lineage["fixedImpliedAction"], FIXED_ACTION)

    def test_missing_lineage_fails_structurally(self) -> None:
        plan = _assemble(ad_count=2)
        plan.planning_internals.pop("conceptualLineage", None)
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertIn("conceptual_lineage_missing", result.reasons)

    def test_mismatched_source_slogan_id_fails(self) -> None:
        plan = _assemble(ad_count=2)
        plan.planning_internals["conceptualLineage"]["sourceSloganCandidateId"] = "L99"
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertIn("conceptual_source_slogan_mismatch", result.reasons)

    def test_mismatched_implied_action_fails(self) -> None:
        plan = _assemble(ad_count=2)
        plan.planning_internals["conceptualLineage"]["fixedImpliedAction"] = "Different action"
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertIn("conceptual_source_action_mismatch", result.reasons)

    def test_changed_conceptual_generator_fails_structurally(self) -> None:
        plan = _assemble(ad_count=2)
        mutated = copy.deepcopy(plan)
        object.__setattr__(mutated, "conceptual_generator", "Mutated concept family")
        result = validate_builder1_campaign_integrity(
            mutated,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertIn("conceptual_generator_mutated", result.reasons)

    def test_unchanged_conceptual_generator_passes(self) -> None:
        plan = _assemble(ad_count=2)
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertTrue(result.ok, msg=str(result.reasons))


class TestNoCreativeServerJudgment(unittest.TestCase):
    def test_integrity_source_has_no_substring_derivation_helper(self) -> None:
        source = inspect.getsource(deterministic_builder1_integrity_checks)
        self.assertNotIn("action_tokens", source)
        self.assertNotIn("conceptual_generator_not_derived_from_slogan", source)

    def test_validate_integrity_has_lineage_not_semantic_derivation(self) -> None:
        source = inspect.getsource(validate_builder1_campaign_integrity)
        self.assertIn("_validate_conceptual_lineage", source)
        self.assertNotIn("conceptual_generator_not_derived_from_slogan", source)


class TestProductionShapedRegression(unittest.TestCase):
    def test_six_stage_shape_passes_with_paraphrased_concept_explanation(self) -> None:
        plan = _assemble(ad_count=2)
        plan.planning_internals["conceptualGeneratorWhyItExpressesSlogan"] = (
            "Escalating impact scenes express the fixed durability action without quoting the slogan"
        )
        for index in (1, 2):
            plan.planning_internals["adInternals"][index]["sloganConnection"] = (
                "Expresses durability through a distinct visual proof"
            )
        result = validate_builder1_campaign_integrity(
            plan,
            upstream=_upstream_for_plan(plan),
            detected_language=plan.detected_language,
        )
        self.assertTrue(result.ok, msg=str(result.reasons))
        stored = series_plan_to_store_dict(plan)
        self.assertIn("conceptualLineage", stored.get("planningInternals", {}))
        self.assertNotIn(
            "conceptual_generator_not_derived_from_slogan",
            deterministic_builder1_integrity_checks(
                {**stored, "detectedLanguage": plan.detected_language}
            ),
        )


class TestConceptualStagePreserved(unittest.TestCase):
    def test_conceptual_stage_prompt_still_receives_fixed_slogan_and_action(self) -> None:
        from engine.builder1_planning_contract import build_conceptual_stage_user_prompt

        prompt = build_conceptual_stage_user_prompt(
            product_description=BRIEF,
            product_name_resolved="TestBrand",
            strategic_problem="Daily carry damage",
            relative_advantage="Survives daily drops",
            brand_slogan=FIXED_SLOGAN,
            slogan_derivation="Distills advantage",
            implied_action=FIXED_ACTION,
            exploration_seed="seed",
        )
        self.assertIn(FIXED_SLOGAN, prompt)
        self.assertIn(FIXED_ACTION, prompt)

    def test_no_final_judge_in_active_pipeline(self) -> None:
        from engine import builder1_planning_pipeline as module

        source = inspect.getsource(module)
        self.assertNotIn("judge_builder1_strategy", source)


if __name__ == "__main__":
    unittest.main()
