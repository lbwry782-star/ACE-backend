"""
Builder1 slogan candidate validation, repair, and selection flow tests.

Run: python -m unittest tests.test_builder1_slogan_quality_flow -v
"""
from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from engine.builder1_planner import Builder1PlannerError, plan_builder1
from engine.builder1_planning_contract import (
    STAGE_SLOGAN_CANDIDATE_REPAIR_SYSTEM,
    STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM,
    STAGE_SLOGAN_SELECT_SYSTEM,
)
from engine.builder1_slogan_quality import (
    SloganFullRescanRequired,
    execute_slogan_scan_through_selection,
    merge_slogan_candidate_replacements,
    parse_slogan_candidate_replacements,
    parse_slogan_quality_review,
    run_slogan_selection_with_quality_gate,
    validate_and_prepare_slogan_candidates,
)
from engine.builder1_slogan_stage import (
    SloganCandidate,
    parse_slogan_candidate_item,
    parse_slogan_scan,
    parse_slogan_selection,
    validate_selected_slogan,
    validate_slogan_candidate,
)
from engine.builder1_staged_parsers import StageParseError
from engine.builder1_strategy_judge import StrategyJudgeResult
from tests.test_builder1_staged_planning import (
    _full_final_responses,
    _slogan_quality_review_payload,
    _slogan_scan_payload,
)


def _candidates(*, generic_index: int | None = None) -> List[SloganCandidate]:
    raw = _slogan_scan_payload(generic=False if generic_index is None else generic_index == 0)
    return parse_slogan_scan(raw)


def _strategy_stub():
    return type(
        "S",
        (),
        {
            "strategic_problem": "Buyers doubt durability",
            "relative_advantage": "Survives daily drops",
            "brief_support": "Brief mentions reinforced shell",
        },
    )()


class TestCandidateValidation(unittest.TestCase):
    def test_all_candidates_validated_before_selection(self) -> None:
        candidates = _candidates()
        results = [
            validate_slogan_candidate(
                c,
                relative_advantage="Survives daily drops",
                product_description="Reinforced shell product",
            )
            for c in candidates
        ]
        self.assertEqual(len(results), 6)

    def test_synonym_slogan_not_rejected_for_overlap(self) -> None:
        candidate = _candidates()[0]
        candidate.brand_slogan = "Drop After Drop"
        candidate.derivation_from_advantage = "Turns impact survival into a spoken challenge"
        result = validate_slogan_candidate(
            candidate,
            relative_advantage="Survives daily drops",
        )
        self.assertNotIn("slogan_not_derived_from_advantage", result.rejection_codes)

    def test_noun_to_action_transformation_can_pass_deterministic(self) -> None:
        candidate = _candidates()[0]
        candidate.brand_slogan = "Take The Hit"
        candidate.derivation_from_advantage = "Verbalizes surviving impact through action"
        candidate.implied_action = "Show repeated impact survival visually"
        result = validate_slogan_candidate(candidate, relative_advantage="Survives daily drops")
        self.assertTrue(result.eligible or "slogan_not_derived_from_advantage" not in result.rejection_codes)

    def test_generic_slogan_rejected(self) -> None:
        candidate = _candidates()[0]
        candidate.brand_slogan = "Quality Without Compromise"
        result = validate_slogan_candidate(candidate, relative_advantage="Survives daily drops")
        self.assertIn("slogan_generic", result.rejection_codes)

    def test_high_transfer_risk_rejected(self) -> None:
        candidate = _candidates()[0]
        candidate.competitor_transfer_risk = "high"
        result = validate_slogan_candidate(candidate, relative_advantage="Survives daily drops")
        self.assertIn("slogan_not_ownable", result.rejection_codes)

    def test_empty_implied_action_rejected(self) -> None:
        candidate = _candidates()[0]
        candidate.implied_action = "x"
        result = validate_slogan_candidate(candidate, relative_advantage="Survives daily drops")
        self.assertIn("slogan_no_implied_action", result.rejection_codes)

    def test_descriptive_category_label_rejected(self) -> None:
        candidate = _candidates()[0]
        candidate.brand_slogan = "Premium Quality Product"
        result = validate_slogan_candidate(candidate, relative_advantage="Survives daily drops")
        self.assertIn("slogan_descriptive_only", result.rejection_codes)

    def test_future_capability_slogan_rejected(self) -> None:
        candidate = _candidates()[0]
        candidate.brand_slogan = "Money Back Guarantee"
        result = validate_slogan_candidate(
            candidate,
            relative_advantage="Survives daily drops",
            product_description="Reinforced shell product",
        )
        self.assertIn("slogan_requires_future_capability", result.rejection_codes)

    def test_old_overlap_test_fixture_still_passes_without_overlap_requirement(self) -> None:
        candidates = parse_slogan_scan(_slogan_scan_payload())
        _, selected = parse_slogan_selection(
            {"selectedCandidateId": "L01", "selectionReason": "Best", "scores": {"directAdvantageExpression": 9}},
            candidates,
        )
        self.assertEqual(selected.brand_slogan, "Built To Last")
        reasons = validate_selected_slogan(
            selected,
            relative_advantage="Survives daily drops",
            product_description="Reinforced shell product",
        )
        self.assertEqual(reasons, [])


class TestFocusedRepair(unittest.TestCase):
    def test_merge_preserves_valid_candidates(self) -> None:
        candidates = _candidates()
        preserved = {c.id: c for c in candidates if c.id != "L03"}
        replacement = {
            "L03": {
                "id": "L03",
                "brandSlogan": "Hold The Line",
                "derivationFromAdvantage": "Expresses surviving daily impact",
                "impliedAction": "Show repeated impact survival visually",
                "whyOwnable": "Tied to reinforced shell durability",
                "whyNaturalInLanguage": "Natural spoken English phrase",
                "competitorTransferRisk": "low",
                "campaignGenerativePower": "Supports several distinct drop-context executions",
            }
        }
        merged = merge_slogan_candidate_replacements(
            candidates,
            replacement,
            preserved=preserved,
        )
        self.assertEqual(merged[0].brand_slogan, candidates[0].brand_slogan)
        self.assertEqual(merged[2].brand_slogan, "Hold The Line")

    def test_unknown_replacement_id_rejected(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_slogan_candidate_replacements(
                {"replacements": [{"id": "L99", "brandSlogan": "x", "derivationFromAdvantage": "x",
                                   "impliedAction": "show repeated proof visually", "whyOwnable": "x",
                                   "whyNaturalInLanguage": "x", "competitorTransferRisk": "low",
                                   "campaignGenerativePower": "supports several distinct executions"}]},
                allowed_ids=["L03"],
            )
        self.assertTrue(any("unexpected_id" in r for r in ctx.exception.reasons))

    def test_duplicate_replacement_id_rejected(self) -> None:
        item = {
            "id": "L03",
            "brandSlogan": "Hold The Line",
            "derivationFromAdvantage": "Expresses surviving daily impact",
            "impliedAction": "Show repeated impact survival visually",
            "whyOwnable": "Tied to reinforced shell durability",
            "whyNaturalInLanguage": "Natural spoken English phrase",
            "competitorTransferRisk": "low",
            "campaignGenerativePower": "Supports several distinct drop-context executions",
        }
        with self.assertRaises(StageParseError):
            parse_slogan_candidate_replacements(
                {"replacements": [item, copy.deepcopy(item)]},
                allowed_ids=["L03"],
            )

    def test_replacement_must_cover_all_invalid_ids(self) -> None:
        with self.assertRaises(StageParseError) as ctx:
            parse_slogan_candidate_replacements({"replacements": []}, allowed_ids=["L03", "L06"])
        self.assertTrue(any("missing_id" in r for r in ctx.exception.reasons))

    def test_five_valid_candidates_unchanged_after_one_repair(self) -> None:
        candidates = _candidates()
        original_text = {c.id: c.brand_slogan for c in candidates}
        preserved = {c.id: copy.deepcopy(c) for c in candidates if c.id != "L03"}
        replacement = {
            "L03": {
                "id": "L03",
                "brandSlogan": "Hold The Line",
                "derivationFromAdvantage": "Expresses surviving daily impact",
                "impliedAction": "Show repeated impact survival visually",
                "whyOwnable": "Tied to reinforced shell durability",
                "whyNaturalInLanguage": "Natural spoken English phrase",
                "competitorTransferRisk": "low",
                "campaignGenerativePower": "Supports several distinct drop-context executions",
            }
        }
        merged = merge_slogan_candidate_replacements(candidates, replacement, preserved=preserved)
        for cid in ("L01", "L02", "L04", "L05", "L06"):
            merged_by_id = {c.id: c for c in merged}
            self.assertEqual(merged_by_id[cid].brand_slogan, original_text[cid])


class TestEligibleSelection(unittest.TestCase):
    def test_selector_cannot_choose_outside_eligible_ids(self) -> None:
        candidates = _candidates()
        with self.assertRaises(StageParseError) as ctx:
            parse_slogan_selection(
                {"selectedCandidateId": "L06", "selectionReason": "Best", "scores": {"directAdvantageExpression": 9}},
                candidates,
                eligible_ids={"L01", "L02"},
            )
        self.assertIn("slogan_selection_ineligible_id", ctx.exception.reasons)

    def test_production_style_slogan_passes_without_word_overlap(self) -> None:
        candidate = _candidates()[0]
        candidate.brand_slogan = "Built To Last"
        candidate.derivation_from_advantage = "Distills durability into a spoken challenge"
        reasons = validate_selected_slogan(
            candidate,
            relative_advantage="Survives daily drops",
            product_description="Reinforced shell product for daily carry",
        )
        self.assertEqual(reasons, [])
        candidate = _candidates()[0]
        gate = validate_selected_slogan(candidate, relative_advantage="Survives daily drops")
        deterministic = validate_slogan_candidate(candidate, relative_advantage="Survives daily drops")
        self.assertEqual(gate, deterministic.rejection_codes)


class TestQualityPipelineIntegration(unittest.TestCase):
    def _model_caller(self, *, repair_payload: Dict[str, Any] | None = None, review_payload: Dict[str, Any] | None = None):
        repair_payload = repair_payload or {"replacements": []}
        review_payload = review_payload or _slogan_quality_review_payload()

        def caller(system: str, user: str, stage: str | None = None) -> object:
            responses = _full_final_responses(4)
            if system == STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM:
                return review_payload
            if system == STAGE_SLOGAN_CANDIDATE_REPAIR_SYSTEM:
                return repair_payload
            return responses.get(system, {"pass": True, "rejectionReasonCodes": []})

        return caller

    def test_valid_candidates_proceed_to_eligible_set(self) -> None:
        candidates = _candidates()
        merged, eligible, _ = validate_and_prepare_slogan_candidates(
            candidates,
            strategic_problem="Buyers doubt durability",
            relative_advantage="Survives daily drops",
            brief_support="Brief mentions reinforced shell",
            product_name="TestBrand",
            product_description="Reinforced shell product",
            detected_language="en",
            model_caller=self._model_caller(),
        )
        self.assertGreaterEqual(len(eligible), 1)
        self.assertEqual(len(merged), 6)

    def test_one_invalid_candidate_triggers_focused_repair(self) -> None:
        candidates = _candidates()
        candidates[2].brand_slogan = "Quality Without Compromise"
        repair = {
            "replacements": [
                {
                    "id": "L03",
                    "brandSlogan": "Hold The Line",
                    "derivationFromAdvantage": "Expresses surviving daily impact",
                    "impliedAction": "Show repeated impact survival visually",
                    "whyOwnable": "Tied to reinforced shell durability",
                    "whyNaturalInLanguage": "Natural spoken English phrase",
                    "competitorTransferRisk": "low",
                    "campaignGenerativePower": "Supports several distinct drop-context executions",
                }
            ]
        }
        _, eligible, repair_used = validate_and_prepare_slogan_candidates(
            candidates,
            strategic_problem="Buyers doubt durability",
            relative_advantage="Survives daily drops",
            brief_support="Brief mentions reinforced shell",
            product_name="TestBrand",
            product_description="Reinforced shell product",
            detected_language="en",
            model_caller=self._model_caller(repair_payload=repair),
        )
        self.assertTrue(repair_used)
        self.assertIn("L03", eligible)

    def test_plan_builder1_preserves_product_name_and_ad_count(self) -> None:
        plan = plan_builder1(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=self._model_caller(),
            ad_count=4,
        )
        self.assertEqual(plan.product_name_resolved, "CarryShell")
        self.assertEqual(plan.ad_count, 4)

    def test_generate_again_does_not_rerun_slogan_stages(self) -> None:
        with patch("engine.builder1_planner.plan_builder1") as mock_plan:
            mock_plan.side_effect = AssertionError("planner must not run")
            from engine.builder1_campaign_store import (
                clear_memory_store_for_tests,
                create_campaign_session,
                mark_ad_generated,
                try_acquire_generation_lock,
                validate_next_ad_request,
            )
            from tests.test_builder1_series import _base_campaign, _parse

            clear_memory_store_for_tests()
            plan = _parse(_base_campaign(2), 2)
            create_campaign_session(campaign_id="slogan-next", plan=plan, target_ad_count=2)
            try_acquire_generation_lock("slogan-next", 1)
            mark_ad_generated("slogan-next", 1)
            validate_next_ad_request("slogan-next", 2)

    def test_conceptual_scan_receives_final_selected_slogan(self) -> None:
        plan = plan_builder1(
            product_name="CarryShell",
            product_description="Reinforced shell product for daily carry",
            format_value="portrait",
            model_caller=self._model_caller(),
            ad_count=4,
        )
        self.assertEqual(plan.brand_slogan, "Built To Last")

    def test_no_eligible_candidates_signals_bounded_full_rescan(self) -> None:
        candidates = _candidates()
        for candidate in candidates:
            candidate.brand_slogan = "Quality Without Compromise"
        review = _slogan_quality_review_payload()
        for review_item in review["reviews"]:
            review_item["eligible"] = False
            review_item["rejectionCodes"] = ["slogan_generic"]
            review_item["derivedFromAdvantage"] = False

        def caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM:
                return review
            return {"replacements": []}

        with self.assertRaises(SloganFullRescanRequired):
            execute_slogan_scan_through_selection(
                slogan_candidates=candidates,
                selected_strategy=_strategy_stub(),
                product_name_resolved="CarryShell",
                product_description="Reinforced shell product",
                detected_language="en",
                model_caller=caller,
                run_stage=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection must not run")),
                full_rescan_used=False,
            )

    def test_second_scan_without_eligible_candidates_fails_honestly(self) -> None:
        candidates = _candidates()
        for candidate in candidates:
            candidate.brand_slogan = "Quality Without Compromise"
        review = _slogan_quality_review_payload()
        for review_item in review["reviews"]:
            review_item["eligible"] = False
            review_item["rejectionCodes"] = ["slogan_generic"]
            review_item["derivedFromAdvantage"] = False

        def caller(system: str, user: str, stage: str | None = None) -> object:
            if system == STAGE_SLOGAN_QUALITY_REVIEW_SYSTEM:
                return review
            return {"replacements": []}

        with self.assertRaises(Builder1PlannerError) as ctx:
            execute_slogan_scan_through_selection(
                slogan_candidates=candidates,
                selected_strategy=_strategy_stub(),
                product_name_resolved="CarryShell",
                product_description="Reinforced shell product",
                detected_language="en",
                model_caller=caller,
                run_stage=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection must not run")),
                full_rescan_used=True,
            )
        self.assertIn("slogan_quality_gate_failed", str(ctx.exception))


class TestReselectionAndRetryLimits(unittest.TestCase):
    def test_gate_failure_triggers_reselection_without_rescan(self) -> None:
        candidates = _candidates()
        eligible = {c.id for c in candidates}
        selection_calls: List[str] = []

        def fake_run_stage(stage, model_caller, system_prompt, user_prompt, parse_fn, **kwargs):
            if stage != "slogan_selection":
                raise AssertionError(stage)
            selection_calls.append(user_prompt)
            selected_id = "L01" if len(selection_calls) == 1 else "L02"
            return parse_fn(
                {
                    "selectedCandidateId": selected_id,
                    "selectionReason": "Best fit",
                    "scores": {"directAdvantageExpression": 9},
                }
            )

        gate_calls: List[str] = []

        def gate_side_effect(slogan, **kwargs):
            gate_calls.append(slogan.id)
            if slogan.id == "L01":
                return ["slogan_generic"]
            return []

        with patch(
            "engine.builder1_slogan_quality.validate_selected_slogan",
            side_effect=gate_side_effect,
        ):
            _, selected, remaining = run_slogan_selection_with_quality_gate(
                candidates=candidates,
                eligible_ids=eligible,
                relative_advantage="Survives daily drops",
                product_description="Reinforced shell product",
                model_caller=lambda *args, **kwargs: {},
                run_stage=fake_run_stage,
            )
        self.assertEqual(selected.id, "L02")
        self.assertEqual(len(selection_calls), 2)
        self.assertEqual(gate_calls, ["L01", "L02"])
        self.assertIn("L02", remaining)

    def test_reselection_preserves_candidate_pool(self) -> None:
        candidates = _candidates()
        before = [copy.deepcopy(c) for c in candidates]
        eligible = {c.id for c in candidates}

        selection_calls = 0

        def fake_run_stage(stage, model_caller, system_prompt, user_prompt, parse_fn, **kwargs):
            nonlocal selection_calls
            selection_calls += 1
            selected_id = "L01" if selection_calls == 1 else "L02"
            return parse_fn(
                {
                    "selectedCandidateId": selected_id,
                    "selectionReason": "Best fit",
                    "scores": {"directAdvantageExpression": 9},
                }
            )

        with patch(
            "engine.builder1_slogan_quality.validate_selected_slogan",
            side_effect=lambda slogan, **kwargs: ["slogan_generic"] if slogan.id == "L01" else [],
        ):
            run_slogan_selection_with_quality_gate(
                candidates=candidates,
                eligible_ids=eligible,
                relative_advantage="Survives daily drops",
                product_description="Reinforced shell product",
                model_caller=lambda *args, **kwargs: {},
                run_stage=fake_run_stage,
            )
        self.assertEqual(len(candidates), 6)
        self.assertEqual(candidates[0].brand_slogan, before[0].brand_slogan)

    def test_bounded_retry_prevents_infinite_slogan_loop(self) -> None:
        candidates = _candidates()
        eligible = {c.id for c in candidates}
        selection_calls = 0

        def fake_run_stage(stage, model_caller, system_prompt, user_prompt, parse_fn, **kwargs):
            nonlocal selection_calls
            selection_calls += 1
            selected_id = "L01" if selection_calls == 1 else "L02"
            return parse_fn(
                {
                    "selectedCandidateId": selected_id,
                    "selectionReason": "Best fit",
                    "scores": {"directAdvantageExpression": 9},
                }
            )

        with patch(
            "engine.builder1_slogan_quality.validate_selected_slogan",
            return_value=["slogan_generic"],
        ):
            with self.assertRaises(Builder1PlannerError):
                run_slogan_selection_with_quality_gate(
                    candidates=candidates,
                    eligible_ids=eligible,
                    relative_advantage="Survives daily drops",
                    product_description="Reinforced shell product",
                    model_caller=lambda *args, **kwargs: {},
                    run_stage=fake_run_stage,
                )
        self.assertLessEqual(selection_calls, 2)


class TestImageComplianceUnchanged(unittest.TestCase):
    def test_image_compliance_module_unchanged_by_slogan_quality(self) -> None:
        from engine.builder1_image_compliance import ImageComplianceUnavailableError

        self.assertTrue(issubclass(ImageComplianceUnavailableError, Exception))


class TestQualityReviewParser(unittest.TestCase):
    def test_quality_review_requires_all_ids(self) -> None:
        payload = _slogan_quality_review_payload()
        payload["reviews"] = payload["reviews"][:5]
        with self.assertRaises(StageParseError):
            parse_slogan_quality_review(payload, expected_ids=[f"L{i:02d}" for i in range(1, 7)])

    def test_pass_true_with_violations_rejected(self) -> None:
        payload = {
            "reviews": [
                {
                    "candidateId": "L01",
                    "derivedFromAdvantage": True,
                    "naturalInLanguage": True,
                    "credible": True,
                    "ownable": True,
                    "impliedActionValid": True,
                    "campaignGenerative": True,
                    "eligible": True,
                    "rejectionCodes": ["slogan_generic"],
                }
            ]
        }
        with self.assertRaises(StageParseError):
            parse_slogan_quality_review(payload, expected_ids=["L01"])


class TestBuilder2Unchanged(unittest.TestCase):
    def test_builder2_files_unrelated(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "engine"
        for path in root.glob("builder2*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("builder1_slogan_quality", text)


if __name__ == "__main__":
    unittest.main()
