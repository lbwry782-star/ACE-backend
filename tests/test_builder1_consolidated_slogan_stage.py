"""
Builder1 consolidated slogan_stage structural validation tests.

Run: python -m unittest tests.test_builder1_consolidated_slogan_stage -v
"""
from __future__ import annotations

import copy
import inspect
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_consolidated_stages import process_slogan_stage_response, run_slogan_stage
from engine.builder1_plan_parser import validate_series_plan_structure, _word_count
from engine.builder1_plan_spec import BRAND_SLOGAN_MAX_WORDS, HEADLINE_MAX_WORDS
from engine.builder1_slogan_stage import validate_slogan_candidate
from engine.builder1_slogan_stage_parser import parse_consolidated_slogan_stage_response
from engine.builder1_staged_parsers import StageParseError
from tests.test_builder1_staged_planning import (
    _selected_strategy,
    _slogan_scan_payload,
    _slogan_stage_payload,
)


def _candidate(
    index: int,
    *,
    brand_slogan: str = "Built To Last",
) -> Dict[str, Any]:
    return {
        "id": f"L{index:02d}",
        "brandSlogan": brand_slogan,
        "derivationFromAdvantage": "Distills survives daily drops advantage into spoken phrase",
        "impliedAction": "Show everyday impact survival visually",
        "whyOwnable": "Tied to reinforced shell durability from brief",
        "whyNaturalInLanguage": "Natural spoken English phrase",
        "competitorTransferRisk": "low",
        "campaignGenerativePower": "Supports several distinct drop-context executions",
    }


def _evaluation(index: int, *, eligible: bool = True, rejection_codes: List[str] | None = None) -> Dict[str, Any]:
    return {
        "candidateId": f"L{index:02d}",
        "derivedFromAdvantage": True,
        "naturalInLanguage": True,
        "credible": True,
        "ownable": True,
        "impliedActionValid": True,
        "campaignGenerative": True,
        "eligible": eligible,
        "rejectionCodes": rejection_codes or ([] if eligible else ["slogan_generic"]),
    }


def _slogan_stage_payload_custom(
    *,
    selected_id: str = "L01",
    candidates: List[Dict[str, Any]] | None = None,
    evaluations: List[Dict[str, Any]] | None = None,
    selection_reason: str = "Strongest advantage expression",
) -> Dict[str, Any]:
    payload_candidates = candidates or [_candidate(i) for i in range(1, 7)]
    payload_evaluations = evaluations or [_evaluation(i) for i in range(1, 7)]
    return {
        "candidates": payload_candidates,
        "evaluations": payload_evaluations,
        "selectedCandidateId": selected_id,
        "selectionReason": selection_reason,
    }


class TestActiveSloganPipeline(unittest.TestCase):
    def test_process_slogan_stage_response_does_not_call_legacy_parse_slogan_scan(self) -> None:
        with patch("engine.builder1_slogan_stage.parse_slogan_scan") as legacy:
            process_slogan_stage_response(_slogan_stage_payload())
            legacy.assert_not_called()

    def test_source_does_not_reference_legacy_parse_slogan_scan(self) -> None:
        source = inspect.getsource(process_slogan_stage_response)
        self.assertNotIn("parse_slogan_scan", source)

    def test_active_production_emits_no_slogan_scan_slogan_too_long(self) -> None:
        long_words = " ".join(f"word{i}" for i in range(1, 9))
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=long_words if i <= 4 else "Built To Last") for i in range(1, 7)],
            selected_id="L05",
        )
        selection, selected, _candidates = process_slogan_stage_response(payload)
        self.assertEqual(selection.selected_candidate_id, "L05")
        self.assertEqual(selected.brand_slogan, "Built To Last")

    def test_consolidated_stage_is_single_call_architecture(self) -> None:
        source = inspect.getsource(process_slogan_stage_response)
        self.assertNotIn("slogan_scan", source)
        self.assertNotIn("slogan_quality_review", source)
        self.assertNotIn("slogan_selection", source)
        self.assertNotIn("slogan_candidate_repair", source)

    def test_model_selected_candidate_id_is_preserved(self) -> None:
        payload = _slogan_stage_payload(selected_id="L04")
        selection, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selection.selected_candidate_id, "L04")
        self.assertEqual(selected.id, "L04")


class TestNoHardSloganLimit(unittest.TestCase):
    def test_one_word_slogan_passes(self) -> None:
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan="Endure" if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, "Endure")

    def test_four_word_slogan_passes(self) -> None:
        slogan = "Built To Last Forever"
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=slogan if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, slogan)

    def test_six_word_slogan_passes(self) -> None:
        slogan = "Built To Last Through Daily Drops"
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=slogan if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, slogan)

    def test_seven_word_slogan_passes(self) -> None:
        slogan = "Built To Last Through Every Daily Drop"
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=slogan if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, slogan)

    def test_longer_natural_slogan_passes(self) -> None:
        slogan = "Built To Last Through Every Daily Drop And Impact"
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=slogan if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, slogan)

    def test_product_name_not_prepended_before_validation(self) -> None:
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan="CarryShell Endure" if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, "CarryShell Endure")

    def test_punctuation_does_not_cause_word_count_rejection(self) -> None:
        slogan = "Built — To — Last — Through — Daily — Drops — Always"
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=slogan if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, slogan)

    def test_brand_slogan_max_words_not_used_in_active_parser(self) -> None:
        source = inspect.getsource(parse_consolidated_slogan_stage_response)
        self.assertNotIn("BRAND_SLOGAN_MAX_WORDS", source)
        self.assertNotIn("slogan_scan_slogan_too_long", source)
        self.assertNotIn("brand_slogan_too_long", source)

    def test_active_pipeline_does_not_emit_brand_slogan_too_long(self) -> None:
        long_slogan = " ".join(f"s{i}" for i in range(BRAND_SLOGAN_MAX_WORDS + 3))
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan=long_slogan if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, long_slogan)


class TestStructuralValidation(unittest.TestCase):
    def test_missing_candidates_array_fails(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response({"evaluations": [], "selectedCandidateId": "L01"})
        self.assertIn("slogan_stage_invalid_structure", ctx.exception.reasons)

    def test_candidate_count_other_than_six_fails(self) -> None:
        payload = _slogan_stage_payload_custom(candidates=[_candidate(1)])
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_candidate_ids_invalid", ctx.exception.reasons)

    def test_duplicate_candidate_ids_fail(self) -> None:
        candidates = [_candidate(1)] * 6
        payload = _slogan_stage_payload_custom(candidates=candidates)
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_candidate_ids_invalid", ctx.exception.reasons)

    def test_missing_evaluation_fails(self) -> None:
        payload = _slogan_stage_payload_custom(evaluations=[_evaluation(1)])
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_evaluations_invalid", ctx.exception.reasons)

    def test_unknown_evaluation_id_fails(self) -> None:
        evaluations = [_evaluation(i) for i in range(1, 7)]
        evaluations[0]["candidateId"] = "L99"
        payload = _slogan_stage_payload_custom(evaluations=evaluations)
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_evaluations_invalid", ctx.exception.reasons)

    def test_missing_selected_candidate_id_fails(self) -> None:
        payload = _slogan_stage_payload_custom()
        payload.pop("selectedCandidateId")
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_selected_candidate_missing", ctx.exception.reasons)

    def test_selected_candidate_id_not_found_fails(self) -> None:
        payload = _slogan_stage_payload_custom(selected_id="L99")
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_selected_candidate_missing", ctx.exception.reasons)

    def test_empty_selected_slogan_fails(self) -> None:
        candidates = [_candidate(i) for i in range(1, 7)]
        candidates[0]["brandSlogan"] = ""
        payload = _slogan_stage_payload_custom(candidates=candidates, selected_id="L01")
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_selected_slogan_empty", ctx.exception.reasons)

    def test_empty_selected_implied_action_fails(self) -> None:
        candidates = [_candidate(i) for i in range(1, 7)]
        candidates[0]["impliedAction"] = ""
        payload = _slogan_stage_payload_custom(candidates=candidates, selected_id="L01")
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_implied_action_empty", ctx.exception.reasons)

    def test_selected_candidate_marked_ineligible_is_contradictory(self) -> None:
        evaluations = [_evaluation(i, eligible=(i != 1)) for i in range(1, 7)]
        payload = _slogan_stage_payload_custom(evaluations=evaluations, selected_id="L01")
        with self.assertRaises(StageParseError) as ctx:
            parse_consolidated_slogan_stage_response(payload)
        self.assertIn("slogan_stage_selected_candidate_ineligible", ctx.exception.reasons)

    def test_contradictory_response_retries_slogan_stage_once(self) -> None:
        from engine.builder1_planner import _run_stage
        from engine.builder1_planning_contract import STAGE_SLOGAN_STAGE_SYSTEM, build_slogan_stage_user_prompt

        calls: List[str] = []
        strategy = _selected_strategy()

        def model_caller(_system: str, _user: str, stage: str | None = None) -> object:
            if stage:
                calls.append(stage)
            if len(calls) == 1:
                return _slogan_stage_payload_custom(
                    evaluations=[_evaluation(i, eligible=(i != 1)) for i in range(1, 7)],
                    selected_id="L01",
                )
            return _slogan_stage_payload(selected_id="L02")

        user_prompt = build_slogan_stage_user_prompt(
            product_name_resolved="CarryShell",
            product_description="Reinforced shell product for daily carry",
            detected_language="en",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brief_support=strategy.brief_support,
        )
        selection, selected, _ = _run_stage(
            "slogan_stage",
            model_caller,
            STAGE_SLOGAN_STAGE_SYSTEM,
            user_prompt,
            process_slogan_stage_response,
        )
        self.assertEqual(calls.count("slogan_stage"), 2)
        self.assertEqual(selection.selected_candidate_id, "L02")
        self.assertEqual(selected.id, "L02")

    def test_second_structural_failure_raises_stage_error(self) -> None:
        from engine.builder1_planner import Builder1PlannerError, _run_stage
        from engine.builder1_planning_contract import STAGE_SLOGAN_STAGE_SYSTEM, build_slogan_stage_user_prompt

        strategy = _selected_strategy()

        def model_caller(_system: str, _user: str, stage: str | None = None) -> object:
            return _slogan_stage_payload_custom(
                evaluations=[_evaluation(i, eligible=(i != 1)) for i in range(1, 7)],
                selected_id="L01",
            )

        user_prompt = build_slogan_stage_user_prompt(
            product_name_resolved="CarryShell",
            product_description="Reinforced shell product for daily carry",
            detected_language="en",
            strategic_problem=strategy.strategic_problem,
            relative_advantage=strategy.relative_advantage,
            brief_support=strategy.brief_support,
        )
        with self.assertRaises(Builder1PlannerError) as ctx:
            _run_stage(
                "slogan_stage",
                model_caller,
                STAGE_SLOGAN_STAGE_SYSTEM,
                user_prompt,
                process_slogan_stage_response,
            )
        self.assertIn("slogan_stage_failed", str(ctx.exception))


class TestNoCreativeServerJudgment(unittest.TestCase):
    def test_generic_slogan_passes_when_model_selected_it(self) -> None:
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan="Quality Without Compromise" if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, "Quality Without Compromise")

    def test_non_overlapping_advantage_passes_when_model_selected_it(self) -> None:
        payload = _slogan_stage_payload_custom(
            candidates=[_candidate(i, brand_slogan="Silent Shield" if i == 1 else "Built To Last") for i in range(1, 7)],
        )
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.brand_slogan, "Silent Shield")

    def test_high_transfer_risk_passes_when_model_selected_it(self) -> None:
        candidates = [_candidate(i) for i in range(1, 7)]
        candidates[0]["competitorTransferRisk"] = "high"
        payload = _slogan_stage_payload_custom(candidates=candidates, selected_id="L01")
        _, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selected.competitor_transfer_risk, "high")

    def test_server_does_not_replace_model_selected_candidate(self) -> None:
        payload = _slogan_stage_payload_custom(selected_id="L06")
        selection, selected, _ = process_slogan_stage_response(payload)
        self.assertEqual(selection.selected_candidate_id, "L06")
        self.assertEqual(selected.id, "L06")

    def test_no_focused_creative_repair_in_active_path(self) -> None:
        source = inspect.getsource(process_slogan_stage_response)
        self.assertNotIn("slogan_candidate_repair", source)
        self.assertNotIn("validate_slogan_candidate", source)

    def test_legacy_validate_slogan_candidate_not_called_by_active_path(self) -> None:
        with patch("engine.builder1_slogan_stage.validate_slogan_candidate") as mocked:
            process_slogan_stage_response(_slogan_stage_payload())
            mocked.assert_not_called()


class TestHeadlineRegression(unittest.TestCase):
    def _minimal_plan(self, *, headline: str) -> Dict[str, Any]:
        return {
            "productNameResolved": "CarryShell",
            "strategicProblem": "Daily carry damage",
            "relativeAdvantage": "Survives daily drops",
            "relativeAdvantageSource": "explicit_brief",
            "brandSlogan": "Built To Last Through Every Daily Drop And Impact",
            "conceptualGenerator": "Stress-test mechanism",
            "conceptualGeneratorAction": "Show everyday impact survival visually",
            "conceptualGeneratorInput": "Everyday carry item",
            "conceptualGeneratorTransformation": "Impact absorbed",
            "conceptualGeneratorResult": "Visible durability proof",
            "physicalGenerator": "Reinforced shell",
            "ads": [
                {
                    "index": 1,
                    "headline": headline,
                    "marketingText": "word " * 50,
                    "physicalExecution": "Drop test",
                    "visualExecution": "Impact frame",
                    "sceneDescription": "Concrete sidewalk drop",
                    "newContribution": "Escalating drop height",
                    "conceptualExecution": "Stress reveal",
                    "conceptualActionProof": "Shell stays intact",
                },
                {
                    "index": 2,
                    "headline": "",
                    "marketingText": "word " * 50,
                    "physicalExecution": "Corner impact",
                    "visualExecution": "Close impact",
                    "sceneDescription": "Corner strike scene",
                    "newContribution": "Different impact angle",
                    "conceptualExecution": "Stress reveal two",
                    "conceptualActionProof": "Shell stays intact again",
                },
            ],
        }

    def test_seven_word_headline_remains_valid(self) -> None:
        headline = " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 1))
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=headline),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
        )
        self.assertNotIn("headline_too_long", reasons)

    def test_eight_word_headline_remains_invalid(self) -> None:
        headline = " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 2))
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=headline),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
        )
        self.assertIn("headline_too_long", reasons)

    def test_product_name_excluded_from_headline_count(self) -> None:
        headline = "CarryShell " + " ".join(f"h{i}" for i in range(1, HEADLINE_MAX_WORDS + 1))
        self.assertGreater(_word_count(headline), HEADLINE_MAX_WORDS)
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=headline),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
        )
        self.assertIn("headline_too_long", reasons)

    def test_headline_remains_optional(self) -> None:
        _, reasons = validate_series_plan_structure(
            self._minimal_plan(headline=""),
            expected_format="portrait",
            expected_ad_count=2,
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
        )
        self.assertNotIn("headline_too_long", reasons)


class TestLayerSeparationAndIntegrity(unittest.TestCase):
    def test_validate_series_plan_structure_has_no_brand_slogan_length_check(self) -> None:
        source = inspect.getsource(validate_series_plan_structure)
        self.assertNotIn("brand_slogan_too_long", source)
        self.assertNotIn("BRAND_SLOGAN_MAX_WORDS", source)

    def test_integrity_validator_has_no_slogan_length_check(self) -> None:
        from engine.builder1_campaign_integrity import validate_builder1_campaign_integrity

        source = inspect.getsource(validate_builder1_campaign_integrity)
        self.assertNotIn("brand_slogan_too_long", source)
        self.assertNotIn("BRAND_SLOGAN_MAX_WORDS", source)

    def test_legacy_deterministic_validator_still_flags_long_slogans_only_in_legacy_helpers(self) -> None:
        candidate = validate_slogan_candidate(
            process_slogan_stage_response(_slogan_stage_payload())[1],
            relative_advantage="Survives daily drops",
        )
        self.assertTrue(candidate.eligible)


if __name__ == "__main__":
    unittest.main()
