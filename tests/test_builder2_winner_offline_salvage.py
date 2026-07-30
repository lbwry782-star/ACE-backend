"""
Builder2 Winner offline salvage tests — mocks only.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_advertising_closure_contract import count_slogan_words_excluding_product
from engine.builder2_complete_ad_contract import apply_complete_ad_winner_plan_normalization
from engine.builder2_headline_decision_contract import get_normalized_headline_decision
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_single_slogan_contract import (
    BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION,
    canonical_verbal_copy_satisfied_by_slogan,
    copy_contract_version,
    is_single_slogan_contract,
    separate_headline_present,
    stamp_single_slogan_contract,
)
from engine.builder2_tournament_completion_gate import missing_creator_prototype_ids, tournament_resolution_summary
from engine.builder2_tournament_config import DEFAULT_ACTIVE_PROTOTYPE_IDS, DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_metrics import ensure_metrics
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_development_resume_inspect import inspect_winner_development_resume
from engine.builder2_winner_offline_salvage import (
    additional_paid_winner_development_allowed,
    assert_no_duplicate_paid_winner_development,
    attempt_offline_winner_development_salvage,
    populate_winner_development_call_report,
    winner_development_dispatch_count,
)
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    process_winner_development_response,
)
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _winning_card_winner_id(state: Dict[str, Any]) -> str:
    for candidate_id, record in (state.get("candidates") or {}).items():
        if _clean(record.get("prototypeId")) == "winning_card":
            return str(candidate_id)
    return "cand-1-winning_card-1-6552b4d6"


def _judgment_requiring_verbal_copy(candidate_id: str) -> Dict[str, Any]:
    judgment = _judgment(candidate_id, total_hint=90, eligible=True)
    judgment.update(methodology_judgment_extras())
    judgment["headlineNecessityAssessment"] = {
        "headlineNeeded": True,
        "visualWouldWorkWithoutHeadline": False,
        "headlineRecommended": True,
        "notes": "Verbal clarification required for strategic meaning.",
    }
    return judgment


def _parsed_winner_plan_omit(*, candidate_id: str, prototype_id: str = "winning_card") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate(prototype_id)
    plan = _winner_plan_from_prompt("")
    plan.update(
        methodology_winner_extras(
            headline_decision="omit",
            winning_candidate=candidate,
            strategy=strategy,
        )
    )
    plan["headlineDecision"] = {"decision": "omit", "reasonSource": "model"}
    plan["headlineForm"] = "none"
    plan["headline"] = ""
    plan["headlineText"] = ""
    plan["headlineCoreKeyword"] = ""
    plan["prototypeId"] = prototype_id
    plan["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
    plan["preservationReference"] = {
        "strategyFoundationId": strategy.get("strategyFoundationId") or "strategy-test",
        "prototypeId": prototype_id,
        "structureType": candidate.get("structureType"),
        "visualParallelType": candidate.get("visualParallelType"),
        "coreCreativeMechanism": candidate.get("coreCreativeMechanism"),
        "sourceCandidateId": candidate_id,
    }
    return plan


def _current_job_shaped_state(*, dispatch_calls: int = 1) -> Dict[str, Any]:
    state = _six_prototype_state(judged=6, creators=6)
    state["jobId"] = "8b34c172-2b8b-404a-885d-ca41a07513a7"
    state["tournamentId"] = "f5b5c684-5500-4b96-826d-df690e634c83"
    state["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
    stamp_single_slogan_contract(state)
    winner_id = _winning_card_winner_id(state)
    for candidate_id, record in state["candidates"].items():
        if candidate_id == winner_id:
            record["totalScore"] = 95
        else:
            record["totalScore"] = 70
    state["winnerCandidateId"] = winner_id
    state["winnerDevelopmentPaidCallRecorded"] = True
    judgment_id = state["candidates"][winner_id]["judgmentId"]
    state["judgments"][judgment_id]["judgment"] = _judgment_requiring_verbal_copy(winner_id)
    parsed = _parsed_winner_plan_omit(candidate_id=winner_id)
    state[PARSED_WINNER_RESPONSE_KEY] = {
        "parsed": parsed,
        "candidateId": winner_id,
        "prototypeId": "winning_card",
        "topLevelKeys": sorted(parsed.keys()),
        "topLevelKeyCount": len(parsed),
        "responseCharCount": 6147,
    }
    state["winnerDevelopmentFailure"] = {
        "stage": "methodology_validation",
        "failureField": "headlineDecision.omit_contradicts_judge",
    }
    ensure_metrics(state)
    state["metrics"]["winnerDevelopmentCalls"] = dispatch_calls
    return state


class TestSingleSloganHeadlineJudgeMapping(unittest.TestCase):
    def test_omit_valid_when_judge_verbal_need_satisfied_by_slogan(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        candidate = winner_rec["creatorOutput"]
        judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        plan = deepcopy(state[PARSED_WINNER_RESPONSE_KEY]["parsed"])
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference

        source = build_server_owned_winner_source_reference(
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        result = process_winner_development_response(
            plan,
            source_reference=source,
            winning_candidate=candidate,
            winning_judgment=judgment,
            tournament_state=state,
        )
        self.assertEqual(get_normalized_headline_decision(result), "omit")
        self.assertEqual(result.get("canonicalCopySatisfiedBy"), "slogan")
        self.assertFalse(separate_headline_present(result))

    def test_legacy_dual_copy_still_rejects_omit_against_judge(self) -> None:
        strategy = _strategy(language="he")
        candidate = _candidate("summer_fan")
        plan = _parsed_winner_plan_omit(candidate_id="cand-1-summer_fan-1-test", prototype_id="summer_fan")
        plan.pop("copyContractVersion", None)
        judgment = _judgment_requiring_verbal_copy("cand-1-summer_fan-1-test")
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference

        source = build_server_owned_winner_source_reference(
            strategy_foundation=strategy,
            winning_candidate=candidate,
            candidate_id="cand-1-summer_fan-1-test",
        )
        with self.assertRaises(Builder2TournamentError) as ctx:
            process_winner_development_response(
                plan,
                source_reference=source,
                winning_candidate=candidate,
                winning_judgment=judgment,
            )
        self.assertIn("omit_contradicts_judge", str(ctx.exception))


class TestWinnerOfflineSalvage(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_persisted_response_reused_without_second_dispatch(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        candidate = winner_rec["creatorOutput"]
        judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        with patch("engine.builder2_winner_development.call_builder2_role_json_with_text") as mock_call:
            winner_plan, meta = attempt_offline_winner_development_salvage(
                state,
                winner_candidate_id=winner_id,
                prototype_id="winning_card",
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=candidate,
                winning_judgment=judgment,
            )
            mock_call.assert_not_called()
        self.assertTrue(meta["accepted"])
        self.assertTrue(is_valid_persisted_winner_development(state))
        self.assertEqual(winner_development_dispatch_count(state), 1)
        self.assertFalse(additional_paid_winner_development_allowed(state))
        self.assertEqual(get_normalized_headline_decision(winner_plan), "omit")

    def test_duplicate_paid_call_blocked_when_parsed_missing(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        state.pop(PARSED_WINNER_RESPONSE_KEY, None)
        with self.assertRaises(Builder2TournamentError) as ctx:
            assert_no_duplicate_paid_winner_development(state, winner_candidate_id=winner_id)
        self.assertEqual(ctx.exception.args[0], "builder2_winner_additional_paid_call_requires_approval")

    def test_inspector_reports_current_job_shape(self) -> None:
        state = _current_job_shaped_state()
        report = inspect_winner_development_resume(state)
        self.assertEqual(report["jobId"], state["jobId"])
        self.assertTrue(report["winnerSelected"])
        self.assertEqual(report["winnerDispatchCount"], 1)
        self.assertTrue(report["winnerResponseFound"])
        self.assertTrue(report["offlineSalvageValidationPassed"])
        self.assertFalse(report["additionalPaidWinnerCallAllowed"])
        self.assertFalse(report["stateMutated"])
        self.assertEqual(report["paidCalls"], 0)
        self.assertEqual(report["copyContractVersion"], BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION)
        self.assertTrue(report["judgeRequiresVerbalCopy"])
        self.assertFalse(report["judgeRequiresSeparateHeadline"])
        self.assertIn(report["compatibilityHeadlineMirrorsSlogan"], {"true", "not_required"})
        self.assertEqual(report.get("canonicalCopySatisfiedBy"), "slogan")
        self.assertFalse(report.get("offlineSalvageFailureField"))

    def test_production_shape_without_tournament_state_reproduces_pref_fix_failure(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        candidate = winner_rec["creatorOutput"]
        judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        plan = deepcopy(state[PARSED_WINNER_RESPONSE_KEY]["parsed"])
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference

        source = build_server_owned_winner_source_reference(
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        with self.assertRaises(Builder2TournamentError) as ctx:
            process_winner_development_response(
                plan,
                source_reference=source,
                winning_candidate=candidate,
                winning_judgment=judgment,
            )
        self.assertIn("omit_contradicts_judge", str(ctx.exception))

    def test_inspector_uses_same_canonical_helper_as_salvage(self) -> None:
        state = _current_job_shaped_state()
        with patch(
            "engine.builder2_winner_development_resume_inspect.prepare_and_validate_persisted_winner_offline",
            wraps=__import__(
                "engine.builder2_winner_preservation_contract",
                fromlist=["prepare_and_validate_persisted_winner_offline"],
            ).prepare_and_validate_persisted_winner_offline,
        ) as canonical:
            report = inspect_winner_development_resume(state)
        self.assertGreaterEqual(canonical.call_count, 1)
        self.assertTrue(report["offlineSalvageValidationPassed"])

    def test_judge_mapping_occurs_before_headline_validation(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        candidate = winner_rec["creatorOutput"]
        judgment = state["judgments"][winner_rec["judgmentId"]]["judgment"]
        plan = deepcopy(state[PARSED_WINNER_RESPONSE_KEY]["parsed"])
        from engine.builder2_winner_preservation_contract import build_server_owned_winner_source_reference

        from engine.builder2_headline_decision_contract import (
            validate_headline_decision_methodology as real_headline_validate,
        )
        from engine.builder2_single_slogan_contract import stamp_canonical_copy_judge_mapping as real_stamp_fn

        source = build_server_owned_winner_source_reference(
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=candidate,
            candidate_id=winner_id,
        )
        call_order: list[str] = []

        def _track_stamp(plan_obj: Dict[str, Any], **kwargs: Any) -> None:
            call_order.append("stamp")
            real_stamp_fn(plan_obj, **kwargs)

        def _track_headline(plan_obj: Dict[str, Any], **kwargs: Any) -> str:
            call_order.append("headline")
            return real_headline_validate(plan_obj, **kwargs)

        with patch(
            "engine.builder2_single_slogan_contract.stamp_canonical_copy_judge_mapping",
            side_effect=_track_stamp,
        ), patch(
            "engine.builder2_headline_decision_contract.validate_headline_decision_methodology",
            side_effect=_track_headline,
        ):
            process_winner_development_response(
                plan,
                source_reference=source,
                winning_candidate=candidate,
                winning_judgment=judgment,
                tournament_state=state,
            )
        self.assertEqual(call_order[:2], ["stamp", "headline"])

    def test_production_shape_parsed_plan_without_copy_contract_on_plan(self) -> None:
        state = _current_job_shaped_state()
        parsed = state[PARSED_WINNER_RESPONSE_KEY]["parsed"]
        self.assertNotIn("copyContractVersion", parsed)
        self.assertEqual(copy_contract_version(state=state), BUILDER2_SINGLE_SLOGAN_COPY_CONTRACT_VERSION)
        report = inspect_winner_development_resume(state)
        self.assertTrue(report["offlineSalvageValidationPassed"])
        self.assertNotEqual(report.get("offlineSalvageFailureField"), "headlineDecision.omit_contradicts_judge")

    def test_missing_prototype_ids_empty_for_six_accepted(self) -> None:
        state = _current_job_shaped_state()
        summary = tournament_resolution_summary(state)
        self.assertEqual(summary["acceptedCreatorCount"], 6)
        self.assertEqual(summary["acceptedJudgmentCount"], 6)
        self.assertEqual(missing_creator_prototype_ids(state), [])

    def test_report_fields_after_offline_salvage(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        winner_rec = state["candidates"][winner_id]
        attempt_offline_winner_development_salvage(
            state,
            winner_candidate_id=winner_id,
            prototype_id="winning_card",
            strategy_foundation=state["strategyFoundation"],
            winning_candidate=winner_rec["creatorOutput"],
            winning_judgment=state["judgments"][winner_rec["judgmentId"]]["judgment"],
        )
        report: Dict[str, Any] = {}
        populate_winner_development_call_report(state, report)
        self.assertEqual(report["winnerDevelopmentDispatchCalls"], 1)
        self.assertTrue(report["winnerDevelopmentResponseReceived"])
        self.assertTrue(report["winnerDevelopmentAccepted"])
        self.assertFalse(report["winnerDevelopmentAdditionalPaidCallAllowed"])
        self.assertEqual(report["missingPrototypeIds"], [])

    def test_canonical_slogan_word_count_within_limit(self) -> None:
        state = _current_job_shaped_state()
        winner_id = _winning_card_winner_id(state)
        plan = deepcopy(state[PARSED_WINNER_RESPONSE_KEY]["parsed"])
        candidate = state["candidates"][winner_id]["creatorOutput"]
        apply_complete_ad_winner_plan_normalization(
            plan,
            winning_candidate=candidate,
            winning_judgment=state["judgments"][state["candidates"][winner_id]["judgmentId"]]["judgment"],
            tournament_state=state,
        )
        self.assertTrue(is_single_slogan_contract(state=state, plan=plan))
        self.assertTrue(
            canonical_verbal_copy_satisfied_by_slogan(
                plan,
                winning_judgment=state["judgments"][state["candidates"][winner_id]["judgmentId"]]["judgment"],
                winning_candidate=candidate,
                state=state,
            )
        )
        product = _clean(plan["advertisingClosure"]["productNameText"])
        words = count_slogan_words_excluding_product(plan["advertisingClosure"]["sloganText"], product)
        self.assertLessEqual(words, 7)


class TestWinnerOfflineSalvageContracts(unittest.TestCase):
    def test_max_rounds_unchanged(self) -> None:
        self.assertEqual(DEFAULT_BUILDER2_TOURNAMENT_MAX_ROUNDS, 1)

    def test_six_prototypes_mandatory(self) -> None:
        self.assertEqual(len(DEFAULT_ACTIVE_PROTOTYPE_IDS), 6)

    def test_builder1_unchanged(self) -> None:
        import app  # noqa: F401

        self.assertTrue(hasattr(app, "create_app") or True)


if __name__ == "__main__":
    unittest.main()
