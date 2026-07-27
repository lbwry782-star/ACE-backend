"""
Builder2 complete-ad tournament tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_contract import (
    apply_semantic_eligibility_rules,
    build_default_creator_advertising_closure,
    build_default_creator_semantic_bridge,
    validate_creator_complete_ad_fields,
    validate_winner_slogan_preservation,
)
from engine.builder2_judge import validate_judge_response
from engine.builder2_new_format_config import (
    DEFAULT_BUILDER2_END_CARD_DURATION_SECONDS,
    DEFAULT_BUILDER2_FINAL_VIDEO_DURATION_SECONDS,
    DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS,
    NORMAL_REASONING_CALL_BUDGET,
    validate_new_format_runway_configuration,
)
from engine.builder2_new_format_preflight import inspect_builder2_new_format_preflight
from engine.builder2_normal_production_guard import NORMAL_PRODUCTION_BLOCKED_ROLES, NormalProductionGuard
from engine.builder2_runway_config import resolve_builder2_runway_video_model, resolve_builder2_video_duration_seconds
from engine.builder2_tournament_contracts import Builder2TournamentError
from tests.builder2_methodology_fixtures import complete_ad_judgment_extras
from tests.test_builder2_tournament import _candidate, _judgment, _strategy


class TestCompleteAdCreator(unittest.TestCase):
    def test_each_creator_includes_one_slogan(self) -> None:
        candidate = _candidate("closest")
        self.assertTrue(str(candidate["advertisingClosure"]["sloganText"]))
        validate_creator_complete_ad_fields(candidate, strategy_foundation=_strategy(), assigned_prototype_id="closest")

    def test_generic_slogan_structurally_accepted(self) -> None:
        candidate = _candidate("closest")
        candidate["advertisingClosure"]["sloganText"] = "חלק מהדרך"
        validate_creator_complete_ad_fields(
            candidate,
            strategy_foundation=_strategy(),
            assigned_prototype_id="closest",
            product_name="ACE Product",
        )

    def test_generic_slogan_quality_rejected_at_judge_layer(self) -> None:
        from engine.builder2_advertising_closure_contract import validate_slogan_text_quality

        with self.assertRaises(Builder2TournamentError):
            validate_slogan_text_quality(slogan="חלק מהדרך", product_name="ACE Product")


class TestSemanticAlignment(unittest.TestCase):
    def test_semantic_alignment_mandatory_for_eligibility(self) -> None:
        judgment = _judgment("c1", eligible=True)
        judgment["semanticAlignmentAssessment"]["semanticAlignment"] = False
        adjusted = apply_semantic_eligibility_rules(judgment)
        self.assertFalse(adjusted["eligible"])

    def test_high_score_cannot_override_semantic_mismatch(self) -> None:
        judgment = _judgment("c1", eligible=True, total_hint=95)
        judgment["semanticAlignmentAssessment"]["semanticAlignment"] = False
        judgment["scores"]["problemAdvantageIntegrity"] = 20
        adjusted = apply_semantic_eligibility_rules(judgment)
        self.assertFalse(adjusted["eligible"])

    def test_judge_records_visual_and_slogan_meanings_separately(self) -> None:
        parsed, _, _ = validate_judge_response(_judgment("c1"), candidate_id="c1", candidate=_candidate("closest"))
        assessment = parsed["semanticAlignmentAssessment"]
        self.assertTrue(assessment["visualMeaning"])
        self.assertTrue(assessment["sloganMeaning"])

    def test_weak_prototype_fit_does_not_force_ineligibility(self) -> None:
        judgment = _judgment("c1", eligible=True)
        judgment["prototypeApplicationAssessment"]["prototypeFitScore"] = 2
        judgment["scores"]["prototypeMethodApplication"] = 2
        adjusted = apply_semantic_eligibility_rules(judgment)
        self.assertTrue(adjusted["eligible"])


class TestWinnerSloganPreservation(unittest.TestCase):
    def test_winner_cannot_replace_creator_slogan(self) -> None:
        candidate = _candidate("closest")
        plan = {"advertisingClosure": {"sloganText": "Different slogan", "productNameText": "ACE Product", "required": True}}
        with self.assertRaises(Builder2TournamentError):
            validate_winner_slogan_preservation(plan, winning_candidate=candidate)


class TestNormalProductionBudget(unittest.TestCase):
    def test_blocked_roles_include_post_winner_copy(self) -> None:
        self.assertIn("advertising_closure", NORMAL_PRODUCTION_BLOCKED_ROLES)
        self.assertIn("marketing_copy", NORMAL_PRODUCTION_BLOCKED_ROLES)

    def test_normal_budget_is_fourteen(self) -> None:
        self.assertEqual(NORMAL_REASONING_CALL_BUDGET, 14)

    def test_blocked_role_raises_in_normal_production(self) -> None:
        NormalProductionGuard.begin()
        try:
            with self.assertRaises(Builder2TournamentError):
                NormalProductionGuard.assert_reasoning_call_allowed("advertising_closure")
        finally:
            NormalProductionGuard.end()


class TestRunwayAndDurationConfig(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_gen4_5_ten_twelve_two(self) -> None:
        self.assertEqual(resolve_builder2_runway_video_model(), "gen4.5")
        self.assertEqual(resolve_builder2_video_duration_seconds(), DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS)
        ok, failures = validate_new_format_runway_configuration()
        self.assertTrue(ok, failures)
        self.assertEqual(DEFAULT_BUILDER2_RUNWAY_DURATION_SECONDS, 10)
        self.assertEqual(DEFAULT_BUILDER2_END_CARD_DURATION_SECONDS, 2.0)
        self.assertEqual(DEFAULT_BUILDER2_FINAL_VIDEO_DURATION_SECONDS, 12.0)


class TestDualMeaningRule(unittest.TestCase):
    def test_missing_strategic_activation_fails_eligibility(self) -> None:
        judgment = _judgment("c1", eligible=True)
        assessment = judgment["semanticAlignmentAssessment"]
        assessment["dualMeaningUsed"] = True
        assessment["physicalMeaningActivatedByVisual"] = True
        assessment["strategicMeaningActivatedBySlogan"] = False
        assessment["meaningsConverge"] = False
        adjusted = apply_semantic_eligibility_rules(judgment)
        self.assertFalse(adjusted["eligible"])

    def test_dual_meaning_convergence_passes(self) -> None:
        judgment = _judgment("c1", eligible=True)
        assessment = judgment["semanticAlignmentAssessment"]
        assessment.update(
            {
                "dualMeaningUsed": True,
                "physicalMeaningActivatedByVisual": True,
                "strategicMeaningActivatedBySlogan": True,
                "meaningsConverge": True,
            }
        )
        adjusted = apply_semantic_eligibility_rules(judgment)
        self.assertTrue(adjusted["eligible"])


class TestHeadlineDecisionAndClosure(unittest.TestCase):
    def test_omit_headline_still_requires_end_card(self) -> None:
        candidate = _candidate("closest")
        self.assertTrue(candidate["advertisingClosure"]["required"])
        self.assertEqual(candidate["advertisingClosure"]["presentationMode"], "end_card")


class TestPrototypeScoring(unittest.TestCase):
    def test_prototype_fit_changes_score(self) -> None:
        strong = _judgment("c1", eligible=True)
        weak = _judgment("c2", eligible=True)
        weak["prototypeApplicationAssessment"]["prototypeFitScore"] = 3
        weak["scores"]["prototypeMethodApplication"] = 3
        self.assertGreater(
            strong["scores"]["prototypeMethodApplication"],
            weak["scores"]["prototypeMethodApplication"],
        )


class TestPreflight(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_preflight_zero_paid_calls(self) -> None:
        report = inspect_builder2_new_format_preflight("")
        self.assertTrue(report["ok"])
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)


class TestBuilder1Unchanged(unittest.TestCase):
    def test_builder1_module_still_present(self) -> None:
        import app

        self.assertIn("builder1_generate", open(app.__file__, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
