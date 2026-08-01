"""
Builder2 advertising-slogan quality contract tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_advertising_slogan_quality_contract import (
    BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
    apply_advertising_slogan_eligibility_rules,
    build_default_creator_slogan_formulation,
    validate_creator_advertising_slogan_formulation,
    validate_judge_advertising_slogan_assessment,
    validate_slogan_advertising_quality_deterministic,
    validate_winner_advertising_slogan_evidence,
)
from engine.builder2_complete_ad_contract import (
    apply_semantic_eligibility_rules,
    validate_creator_complete_ad_fields,
)
from engine.builder2_creator_slogan_repair_patch import additional_paid_slogan_repair_allowed
from engine.builder2_creator_core_contract import build_creator_required_keys_prompt_text
from engine.builder2_methodology_validation import validate_judge_methodology, validate_winner_methodology
from engine.builder2_new_format_config import NORMAL_REASONING_CALL_BUDGET
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_prompts import build_winner_development_prompt
from engine.builder2_tournament_store import new_tournament_state
from tests.builder2_methodology_fixtures import (
    advertising_slogan_quality_creator_extras,
    advertising_slogan_quality_judge_extras,
    advertising_slogan_quality_winner_extras,
    complete_ad_creator_extras,
    methodology_candidate_extras,
    methodology_judgment_extras,
    methodology_strategy_extras,
    methodology_winner_extras,
    single_slogan_contract_extras,
)
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


HEBREW_RELATIVE_ADVANTAGE = "שוקולד בסגנון דובאי שמיוצר בישראל"
WEAK_STRATEGIC_PROSE = "סגנון דובאי ממקור ישראלי גלוי"
VALID_ADVERTISING_SLOGAN = "שוקולד דובאי תוצרת ישראל"
PRODUCT_NAME = "דובי"


def _hebrew_strategy(*, relative_advantage: str = HEBREW_RELATIVE_ADVANTAGE) -> Dict[str, Any]:
    strategy = _strategy(language="he")
    strategy["productNameResolved"] = PRODUCT_NAME
    strategy["relativeAdvantage"] = {
        "statement": relative_advantage,
        "derivationFromProblem": "הקונה מחפש שוקולד דובאי אמיתי שמיוצר מקומית.",
        "truthBoundary": "לא טוען ייצור בדובאי עצמו.",
        "admitsRelevantGap": True,
    }
    return strategy


def _hebrew_candidate(*, slogan_text: str, relative_advantage: str = HEBREW_RELATIVE_ADVANTAGE) -> Dict[str, Any]:
    candidate = _candidate("closest")
    candidate.update(
        complete_ad_creator_extras(
            product_name=PRODUCT_NAME,
            slogan_text=slogan_text,
            language="he",
            relative_advantage_source=relative_advantage,
        )
    )
    candidate.update(
        advertising_slogan_quality_creator_extras(
            relative_advantage_source=relative_advantage,
            final_slogan_text=slogan_text,
            transformation_type="contrast" if slogan_text == VALID_ADVERTISING_SLOGAN else "direct_distillation",
            why_advertising="ניסוח פרסומי קצר ולא הסבר אסטרטגי.",
        )
    )
    return candidate


class TestAdvertisingSloganQualityDeterministic(unittest.TestCase):
    def test_relative_advantage_description_not_auto_accepted(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_slogan_advertising_quality_deterministic(
                slogan=HEBREW_RELATIVE_ADVANTAGE,
                product_name=PRODUCT_NAME,
                relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
            )
        self.assertIn("restates_relative_advantage", ctx.exception.args[0])

    def test_near_verbatim_explanatory_restatement_rejected(self) -> None:
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_slogan_advertising_quality_deterministic(
                slogan=WEAK_STRATEGIC_PROSE,
                product_name=PRODUCT_NAME,
                relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
            )
        self.assertIn("strategic_description_markers", ctx.exception.args[0])

    def test_concise_advertising_transformation_accepted(self) -> None:
        validate_slogan_advertising_quality_deterministic(
            slogan=VALID_ADVERTISING_SLOGAN,
            product_name=PRODUCT_NAME,
            relative_advantage=HEBREW_RELATIVE_ADVANTAGE,
        )

    def test_direct_distillation_allowed_when_slogan_like(self) -> None:
        validate_slogan_advertising_quality_deterministic(
            slogan="קרוב יותר ממה שחשבת",
            product_name="ACE Product",
            relative_advantage="Closeness becomes the advantage.",
        )

    def test_wordplay_not_required(self) -> None:
        formulation = build_default_creator_slogan_formulation(
            relative_advantage_source="Closeness becomes the advantage.",
            final_slogan_text="קרוב יותר ממה שחשבת",
            transformation_type="direct_distillation",
        )
        self.assertEqual(formulation["advertisingTransformationType"], "direct_distillation")

    def test_factual_grounding_required_in_creator_formulation(self) -> None:
        candidate = _hebrew_candidate(slogan_text=VALID_ADVERTISING_SLOGAN)
        candidate["advertisingSloganFormulation"]["factualGroundingPreserved"] = False
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_advertising_slogan_formulation(
                candidate,
                strategy_foundation=_hebrew_strategy(),
                product_name=PRODUCT_NAME,
            )
        self.assertIn("factualGroundingPreserved", ctx.exception.args[0])


class TestAdvertisingSloganJudgeAndWinner(unittest.TestCase):
    def test_judge_distinguishes_description_from_advertising_copy(self) -> None:
        judgment = _judgment("cand-1")
        judgment.update(methodology_judgment_extras(prototype_id="closest"))
        assessment = judgment["advertisingSloganAssessment"]
        assessment["merelyDescriptive"] = True
        assessment["soundsLikeAdvertising"] = False
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_judge_advertising_slogan_assessment(judgment)
        self.assertIn("merelyDescriptive", ctx.exception.args[0])

    def test_judge_merely_descriptive_makes_candidate_ineligible(self) -> None:
        judgment = _judgment("cand-1")
        judgment.update(methodology_judgment_extras(prototype_id="closest"))
        judgment["eligible"] = True
        judgment["advertisingSloganAssessment"]["merelyDescriptive"] = True
        adjusted = apply_semantic_eligibility_rules(apply_advertising_slogan_eligibility_rules(judgment))
        self.assertFalse(adjusted["eligible"])
        self.assertIn("slogan_merely_descriptive", adjusted["disqualifiers"])

    def test_winner_cannot_persist_merely_descriptive_true(self) -> None:
        strategy = _hebrew_strategy()
        candidate = _hebrew_candidate(slogan_text=VALID_ADVERTISING_SLOGAN)
        plan = _winner_plan_from_prompt("")
        plan.update(
            methodology_winner_extras(
                headline_decision="omit",
                winning_candidate=candidate,
                strategy=strategy,
            )
        )
        plan["productNameResolved"] = PRODUCT_NAME
        plan["language"] = "he"
        plan["relativeAdvantage"] = HEBREW_RELATIVE_ADVANTAGE
        plan["advertisingClosure"] = candidate["advertisingClosure"]
        plan.update(single_slogan_contract_extras(slogan_text=VALID_ADVERTISING_SLOGAN))
        plan["advertisingSloganEvidence"]["merelyDescriptive"] = True
        state = new_tournament_state(job_id="job", language="he", active_prototype_ids=["closest"], random_seed="seed")
        state["strategyFoundation"] = strategy
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_winner_methodology(
                plan,
                winning_candidate=candidate,
                tournament_state=state,
            )
        self.assertIn("merelyDescriptive", ctx.exception.args[0])

    def test_winner_preserves_relative_advantage_while_improving_wording(self) -> None:
        strategy = _hebrew_strategy()
        candidate = _hebrew_candidate(slogan_text=VALID_ADVERTISING_SLOGAN)
        plan = _winner_plan_from_prompt("")
        plan.update(
            methodology_winner_extras(
                headline_decision="omit",
                winning_candidate=candidate,
                strategy=strategy,
            )
        )
        plan["productNameResolved"] = PRODUCT_NAME
        plan["language"] = "he"
        plan["relativeAdvantage"] = HEBREW_RELATIVE_ADVANTAGE
        plan["advertisingClosure"] = candidate["advertisingClosure"]
        state = new_tournament_state(job_id="job", language="he", active_prototype_ids=["closest"], random_seed="seed")
        state["strategyFoundation"] = strategy
        validate_winner_advertising_slogan_evidence(
            plan,
            winning_candidate=candidate,
            strategy_foundation=strategy,
        )
        self.assertEqual(plan["advertisingSloganEvidence"]["relativeAdvantageSource"], HEBREW_RELATIVE_ADVANTAGE)
        self.assertEqual(plan["advertisingClosure"]["sloganText"], VALID_ADVERTISING_SLOGAN)


    def test_generic_slogan_rejected_by_quality_contract(self) -> None:
        candidate = _hebrew_candidate(slogan_text="חלק מהדרך")
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_advertising_slogan_formulation(
                candidate,
                strategy_foundation=_hebrew_strategy(),
                product_name=PRODUCT_NAME,
            )
        self.assertIn("sloganText.generic", ctx.exception.args[0])

    def test_historical_candidate_without_formulation_skips_slogan_quality_without_v1_state(self) -> None:
        from engine.builder2_methodology_validation import validate_creator_methodology

        candidate = _candidate("closest")
        candidate.pop("advertisingSloganFormulation", None)
        validate_creator_methodology(
            candidate,
            assigned_prototype_id="closest",
            strategy_foundation=_strategy(),
            tournament_state={},
        )

    def test_v1_state_requires_advertising_formulation(self) -> None:
        from engine.builder2_methodology_validation import validate_creator_methodology

        candidate = _candidate("closest")
        candidate.pop("advertisingSloganFormulation", None)
        state = {"advertisingSloganQualityContractVersion": BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION}
        with self.assertRaises(Builder2TournamentError) as ctx:
            validate_creator_methodology(
                candidate,
                assigned_prototype_id="closest",
                strategy_foundation=_strategy(),
                tournament_state=state,
            )
        self.assertIn("advertisingSloganFormulation", ctx.exception.args[0])


class TestAdvertisingSloganIntegrationGuards(unittest.TestCase):
    def test_creator_prompt_requires_internal_formulation_without_new_call(self) -> None:
        prompt = build_creator_required_keys_prompt_text(prototype_id="closest")
        self.assertIn("advertisingSloganFormulation", prompt)
        self.assertIn("Inside this same Creator response", prompt)

    def test_winner_prompt_requires_evidence_not_new_slogan(self) -> None:
        from engine.builder2_prototypes import require_prototype

        prompt = build_winner_development_prompt(
            product_name=PRODUCT_NAME,
            product_description="premium chocolate",
            language="he",
            strategy_foundation=_hebrew_strategy(),
            winning_candidate=_hebrew_candidate(slogan_text=VALID_ADVERTISING_SLOGAN),
            winning_judgment=_judgment("cand-1"),
            prototype=require_prototype("closest"),
            runway_mode="text_to_video",
            preservation_snapshot={"strategyFoundationId": "sf-1"},
        )
        self.assertIn("advertisingsloganevidence", prompt.lower())
        self.assertIn("finalslogantext must equal advertisingclosure.slogantext", prompt.lower())

    def test_no_additional_normal_path_reasoning_call_added(self) -> None:
        self.assertEqual(NORMAL_REASONING_CALL_BUDGET, 14)
        prompt = build_creator_required_keys_prompt_text(prototype_id="closest")
        self.assertIn("Inside this same Creator response", prompt)
        self.assertIn("advertisingSloganFormulation", prompt)

    def test_bounded_slogan_repair_limit_remains_one(self) -> None:
        from engine.builder2_creator_slogan_repair_patch import reconcile_slogan_repair_call_ledger

        state: Dict[str, Any] = {"metrics": {}}
        self.assertTrue(additional_paid_slogan_repair_allowed(state, "closest"))
        state["metrics"] = {"creatorRepairCalls": 1}
        bucket = reconcile_slogan_repair_call_ledger(state, prototype_id="closest")
        self.assertEqual(bucket["canonicalCreatorRepairCalls"], 1)
        self.assertFalse(additional_paid_slogan_repair_allowed(state, "closest"))

    def test_single_slogan_render_contract_unchanged(self) -> None:
        plan = {"headlineDecision": {"decision": "omit"}, **single_slogan_contract_extras(slogan_text=VALID_ADVERTISING_SLOGAN)}
        candidate = _hebrew_candidate(slogan_text=VALID_ADVERTISING_SLOGAN)
        from engine.builder2_single_slogan_contract import apply_single_slogan_winner_normalization

        apply_single_slogan_winner_normalization(plan, winning_candidate=candidate)
        self.assertEqual(plan["headlineForm"], "none")
        self.assertEqual(plan["advertisingClosure"]["sloganText"], VALID_ADVERTISING_SLOGAN)

    def test_no_separate_headline_rendered(self) -> None:
        from engine.builder2_single_slogan_contract import builder2_requires_headline_overlay

        plan = {
            "headlineDecision": {"decision": "omit"},
            **single_slogan_contract_extras(),
        }
        self.assertFalse(builder2_requires_headline_overlay(plan=plan))

    def test_new_tournament_state_stamps_quality_contract(self) -> None:
        state = new_tournament_state(job_id="job", language="he", active_prototype_ids=["closest"], random_seed="seed")
        self.assertEqual(
            state["advertisingSloganQualityContractVersion"],
            BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
        )

    def test_creator_complete_ad_validation_accepts_advertising_formulation(self) -> None:
        candidate = methodology_candidate_extras("closest")
        validate_creator_complete_ad_fields(
            candidate,
            strategy_foundation=methodology_strategy_extras(),
            product_name="ACE Product",
        )
        validate_creator_advertising_slogan_formulation(
            candidate,
            strategy_foundation=methodology_strategy_extras(),
            product_name="ACE Product",
        )

    def test_judge_methodology_requires_slogan_assessment(self) -> None:
        judgment = _judgment("cand-1")
        judgment.update(methodology_judgment_extras(prototype_id="closest"))
        validate_judge_methodology(judgment, candidate=methodology_candidate_extras("closest"))


class TestClosureOverridePreserved(unittest.TestCase):
    def test_closure_only_override_still_zero_reasoning(self) -> None:
        from engine.builder2_closure_copy import resolve_trusted_closure_copy

        job_id = "edb3136e-21d3-419e-86cd-c5d5bda18012"
        corrected = "שוקולד דובאי תוצרת ישראל"
        state: Dict[str, Any] = {
            "jobId": job_id,
            "winnerDevelopmentPlan": {
                "productNameResolved": PRODUCT_NAME,
                "advertisingClosure": {
                    "required": True,
                    "productNameText": PRODUCT_NAME,
                    "sloganText": WEAK_STRATEGIC_PROSE,
                    "language": "he",
                    "presentationMode": "end_card",
                    "durationSeconds": 3.5,
                    "noLogo": True,
                },
            },
            "mediaResume": {"closureSloganOverride": corrected},
        }
        with patch.dict(os.environ, {"BUILDER2_CLOSURE_ONLY_RERENDER_SLOGAN_TEXT": corrected}, clear=False):
            product_name, slogan, language = resolve_trusted_closure_copy(state)
        self.assertEqual(product_name, PRODUCT_NAME)
        self.assertEqual(slogan, corrected)
        self.assertEqual(language, "he")


if __name__ == "__main__":
    unittest.main()
