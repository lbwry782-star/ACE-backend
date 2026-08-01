"""
Builder2 complete-ad Creator/Judge flow and winner preservation tests — mocks only.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_advertising_closure_contract import validate_slogan_text_quality, validate_slogan_text_structure
from engine.builder2_advertising_slogan_quality_contract import (
    BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION,
)
from engine.builder2_complete_ad_contract import (
    apply_complete_ad_winner_plan_normalization,
    apply_semantic_eligibility_rules,
    validate_creator_complete_ad_fields,
    validate_winner_slogan_preservation,
)
from engine.builder2_complete_ad_creator_recovery import (
    offline_revalidate_and_accept_rejected_creator,
    persist_rejected_creator_parsed_response,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_tournament_completion_gate import (
    assert_tournament_ready_for_winner_selection,
    invalidate_provisional_winner_if_incomplete,
    tournament_resolution_summary,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    offline_revalidate_parsed_winner_response,
    process_winner_development_response,
)
from engine.builder2_winner_plan import validate_builder2_winner_plan
from tests.builder2_methodology_fixtures import methodology_winner_extras
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _six_prototype_state(*, judged: int = 6, creators: int = 6) -> Dict[str, Any]:
    prototypes = ["winning_card", "summer_fan", "forgot", "closest", "think_small", "greenpeace_essential_pairing"]
    state: Dict[str, Any] = {
        "jobId": "job-six-way",
        "initialActivePrototypeIds": prototypes,
        "activePrototypeIds": prototypes,
        "strategyFoundation": _strategy(),
        "acceptedCreatorCandidates": {},
        "acceptedJudgments": {},
        "candidates": {},
        "judgments": {},
    }
    for idx, prototype_id in enumerate(prototypes):
        if idx >= creators:
            continue
        candidate_id = f"cand-1-{prototype_id}-1-test"
        candidate = _candidate(prototype_id)
        state["acceptedCreatorCandidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "validationStatus": "accepted",
            "creatorOutput": candidate,
        }
        state["candidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "validationStatus": "accepted",
            "creatorOutput": candidate,
            "eligible": True,
            "totalScore": 70 + idx,
            "judgmentId": f"j-{candidate_id}" if idx < judged else None,
        }
        if idx < judged:
            state["acceptedJudgments"][candidate_id] = {
                "candidateId": candidate_id,
                "prototypeId": prototype_id,
                "validationStatus": "accepted",
            }
            state["judgments"][f"j-{candidate_id}"] = {
                "judgmentId": f"j-{candidate_id}",
                "candidateId": candidate_id,
                "judgment": _judgment(candidate_id, total_hint=70 + idx),
            }
    return state


class TestCreatorStructuralBoundary(unittest.TestCase):
    def test_generic_slogan_structurally_accepted(self) -> None:
        candidate = _candidate("greenpeace_essential_pairing")
        candidate["advertisingClosure"]["sloganText"] = "חלק מהדרך"
        candidate["advertisingClosure"]["productNameText"] = "ACE Product"
        validate_creator_complete_ad_fields(
            candidate,
            strategy_foundation=_strategy(language="he"),
            assigned_prototype_id="greenpeace_essential_pairing",
            product_name="ACE Product",
        )

    def test_empty_slogan_structurally_rejected(self) -> None:
        candidate = _candidate("closest")
        candidate["advertisingClosure"]["sloganText"] = ""
        with self.assertRaises(Builder2TournamentError):
            validate_creator_complete_ad_fields(
                candidate,
                strategy_foundation=_strategy(),
                assigned_prototype_id="closest",
                product_name="ACE Product",
            )

    def test_wrong_product_identity_structurally_rejected(self) -> None:
        candidate = _candidate("closest")
        candidate["advertisingClosure"]["productNameText"] = "Wrong Product"
        with self.assertRaises(Builder2TournamentError):
            validate_creator_complete_ad_fields(
                candidate,
                strategy_foundation=_strategy(),
                assigned_prototype_id="closest",
                product_name="ACE Product",
            )

    def test_genericness_evaluated_by_judge(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_slogan_text_quality(
                slogan="חלק מהדרך",
                product_name="ACE Product",
            )

    def test_over_word_limit_structurally_rejected(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_slogan_text_structure(
                slogan="one two three four five six seven eight",
                product_name="ACE Product",
            )


class TestSixWayTournamentGate(unittest.TestCase):
    def test_five_judgments_block_winner_selection(self) -> None:
        state = _six_prototype_state(judged=5, creators=6)
        with self.assertRaises(Builder2TournamentError):
            assert_tournament_ready_for_winner_selection(state)

    def test_provisional_winner_invalidated(self) -> None:
        state = _six_prototype_state(judged=5, creators=6)
        state["winnerCandidateId"] = "cand-1-forgot-1-test"
        self.assertTrue(invalidate_provisional_winner_if_incomplete(state))
        self.assertEqual(state.get("provisionalWinnerCandidateId"), "cand-1-forgot-1-test")
        self.assertIsNone(state.get("winnerCandidateId"))

    def test_six_judgments_allow_winner_selection(self) -> None:
        state = _six_prototype_state(judged=6, creators=6)
        summary = tournament_resolution_summary(state)
        self.assertTrue(summary["readyForAuthoritativeWinnerSelection"])
        assert_tournament_ready_for_winner_selection(state)


class TestWinnerPreservationNewFormat(unittest.TestCase):
    def test_omit_headline_without_separate_headline_field(self) -> None:
        candidate = _candidate("forgot")
        strategy = _strategy(language="he")
        raw = _winner_plan_from_prompt("")
        raw.update(methodology_winner_extras(headline_decision="omit", winning_candidate=candidate, strategy=strategy))
        raw["headline"] = ""
        raw["headlineText"] = ""
        raw["headlineCoreKeyword"] = ""
        from engine.builder2_winner_preservation_contract import (
            build_server_owned_winner_source_reference,
            build_winning_candidate_preservation_snapshot,
            process_winner_development_response,
        )

        source = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-1-forgot-1-test",
        )
        snapshot = build_winning_candidate_preservation_snapshot(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-1-forgot-1-test",
        )
        validated = process_winner_development_response(
            raw,
            source_reference=source,
            winning_candidate=candidate,
            preservation_snapshot=snapshot,
            winning_judgment=_judgment("cand-1-forgot-1-test"),
        )
        self.assertEqual(validated["headlineDecision"]["decision"], "omit")
        self.assertEqual(validated["advertisingClosure"]["sloganText"], candidate["advertisingClosure"]["sloganText"])

    def test_winner_model_cannot_replace_slogan(self) -> None:
        candidate = _candidate("forgot")
        plan = {"advertisingClosure": {"required": True, "productNameText": "ACE Product", "sloganText": "Different"}}
        with self.assertRaises(Builder2TournamentError):
            validate_winner_slogan_preservation(plan, winning_candidate=candidate)

    def test_parsed_winner_cannot_revalidate_for_other_candidate(self) -> None:
        candidate = _candidate("forgot")
        other = _candidate("closest")
        state = {"winnerDevelopmentParsedResponse": {"parsed": _winner_plan_from_prompt(""), "candidateId": other["prototypeId"]}}
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference

        source = build_server_owned_winner_source_reference(
            strategy_foundation=_strategy(),
            winning_candidate=candidate,
            candidate_id="cand-1-forgot-1-test",
        )
        state[PARSED_WINNER_RESPONSE_KEY] = {
            "parsed": _winner_plan_from_prompt(""),
            "candidateId": "cand-1-closest-1-test",
        }
        with self.assertRaises(Builder2TournamentError):
            offline_revalidate_parsed_winner_response(
                state,
                source_reference=source,
                winning_candidate=candidate,
            )


class TestOfflineCreatorRecovery(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_offline_revalidation_accepts_valid_current_contract_candidate(self) -> None:
        candidate = _candidate("greenpeace_essential_pairing")
        candidate["advertisingClosure"]["productNameText"] = "ACE Product"
        valid_slogan = candidate["advertisingClosure"]["sloganText"]
        candidate["advertisingSloganFormulation"]["finalSloganText"] = valid_slogan
        candidate_id = "cand-1-greenpeace_essential_pairing-1-33964989"
        state = _six_prototype_state(judged=5, creators=5)
        state["jobId"] = "job-greenpeace"
        state["advertisingSloganQualityContractVersion"] = BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION
        persist_rejected_creator_parsed_response(
            state,
            candidate_id=candidate_id,
            prototype_id="greenpeace_essential_pairing",
            round_index=1,
            attempt_number=1,
            parsed=candidate,
            failure_reason="builder2_advertising_closure_invalid:sloganText.generic",
        )
        accepted = offline_revalidate_and_accept_rejected_creator(
            state,
            candidate_id=candidate_id,
            product_name="ACE Product",
        )
        self.assertEqual(accepted["advertisingClosure"]["sloganText"], valid_slogan)
        self.assertEqual(
            accepted["advertisingSloganFormulation"]["finalSloganText"],
            valid_slogan,
        )
        self.assertIn(candidate_id, state["acceptedCreatorCandidates"])

    def test_offline_revalidation_rejects_generic_slogan_under_quality_contract(self) -> None:
        candidate = _candidate("greenpeace_essential_pairing")
        generic_slogan = "חלק מהדרך"
        candidate["advertisingClosure"]["sloganText"] = generic_slogan
        candidate["advertisingClosure"]["productNameText"] = "ACE Product"
        candidate["advertisingSloganFormulation"]["finalSloganText"] = generic_slogan
        candidate_id = "cand-1-greenpeace_essential_pairing-1-33964989"
        state = _six_prototype_state(judged=5, creators=5)
        state["jobId"] = "job-greenpeace-generic"
        state["advertisingSloganQualityContractVersion"] = BUILDER2_ADVERTISING_SLOGAN_QUALITY_CONTRACT_VERSION
        persist_rejected_creator_parsed_response(
            state,
            candidate_id=candidate_id,
            prototype_id="greenpeace_essential_pairing",
            round_index=1,
            attempt_number=1,
            parsed=candidate,
            failure_reason="builder2_advertising_closure_invalid:sloganText.generic",
        )
        with self.assertRaises(Builder2TournamentError) as ctx:
            offline_revalidate_and_accept_rejected_creator(
                state,
                candidate_id=candidate_id,
                product_name="ACE Product",
            )
        self.assertIn("sloganText.generic", ctx.exception.args[0])
        self.assertNotIn(candidate_id, state["acceptedCreatorCandidates"])


class TestJudgeGenericIneligibility(unittest.TestCase):
    def test_generic_slogan_can_make_candidate_ineligible(self) -> None:
        judgment = _judgment("c1", eligible=True)
        judgment["advertisingCompletionAssessment"]["sloganSpecificToIdea"] = False
        adjusted = apply_semantic_eligibility_rules(judgment)
        self.assertFalse(adjusted["eligible"])


class TestInspectCLI(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    def test_inspector_zero_paid_calls(self, _raw: Any, read_raw: Any, _redis: Any) -> None:
        state = _six_prototype_state(judged=5, creators=6)
        state["provisionalWinnerCandidateId"] = "cand-1-forgot-1-test"
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_complete_ad_resume("job-six-way")
        self.assertTrue(report["ok"])
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)
        self.assertEqual(report["redisMutations"], 0)


if __name__ == "__main__":
    unittest.main()
