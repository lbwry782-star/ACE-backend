"""
Builder2 authoritative reasoning dispatch budget and production-shaped resume tests.
"""
from __future__ import annotations

import copy
import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_accepted_judgment_store import persist_accepted_judgment
from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_complete_ad_resume_plan import (
    RESUME_STAGE_WINNER_DEVELOPMENT,
    resolve_complete_ad_canonical_resume_plan,
)
from engine.builder2_judge_unavailable_resolution_contract import PRODUCTION_CLOSEST_CANDIDATE_ID
from engine.builder2_reasoning_dispatch_budget import CALL_BUDGET_EXHAUSTED, ControlledReasoningCallBudget
from engine.builder2_tournament_completion_gate import mark_authoritative_winner_selection
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    process_winner_development_response,
)
from tests.builder2_methodology_fixtures import methodology_winner_extras, single_slogan_contract_extras
from tests.test_builder2_judge_unavailable_resolution import _apply_resolution, _unrecoverable_closest_state
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_tournament_contracts import TOURNAMENT_STATE_SCHEMA_VERSION
from tests.test_builder2_mixed_partial_resume import _production_mixed_partial_state, _SPARSE_DESCRIPTION, _PROTOTYPES, _strategy_block
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _production_winner_ready_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "jobId": "e369b792-9988-4087-b054-38a713966918",
        "tournamentId": "9d789e1e-7e4a-4ef4-b72e-642da8083788",
        "builder2ResumeContractVersion": BUILDER2_RESUME_CONTRACT_VERSION,
        "builder2NewFormatVersion": BUILDER2_NEW_FORMAT_VERSION,
        "schemaVersion": TOURNAMENT_STATE_SCHEMA_VERSION,
        "status": "paused_for_reasoning_resume",
        "productDescription": _SPARSE_DESCRIPTION,
        "contentLanguage": "he",
        "initialActivePrototypeIds": list(_PROTOTYPES),
        "activePrototypeIds": list(_PROTOTYPES),
        "strategyFoundation": _strategy_block(),
        "acceptedCreatorCandidates": {},
        "acceptedJudgments": {},
        "candidates": {},
        "judgments": {},
        "metrics": {},
    }
    for prototype_id in _PROTOTYPES:
        candidate_id = (
            PRODUCTION_CLOSEST_CANDIDATE_ID
            if prototype_id == "closest"
            else (
                "cand-1-winning_card-1-577b91f2"
                if prototype_id == "winning_card"
                else f"cand-1-{prototype_id}-1-test"
            )
        )
        candidate = _candidate(prototype_id)
        candidate["candidateId"] = candidate_id
        state["acceptedCreatorCandidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "validationStatus": "accepted",
            "creatorOutput": candidate,
        }
        state["candidates"][candidate_id] = {
            "candidateId": candidate_id,
            "prototypeId": prototype_id,
            "creatorAcceptanceStatus": "accepted",
            "validationStatus": "accepted",
            "status": "accepted",
            "creatorOutput": candidate,
        }
    for prototype_id in ["think_small", "winning_card", "summer_fan", "forgot", "greenpeace_essential_pairing"]:
        candidate_id = (
            "cand-1-winning_card-1-577b91f2"
            if prototype_id == "winning_card"
            else f"cand-1-{prototype_id}-1-test"
        )
        total = 89 if prototype_id == "winning_card" else 70
        judgment = _judgment(candidate_id, total_hint=total, eligible=True)
        persist_accepted_judgment(
            state,
            candidate_id=candidate_id,
            prototype_id=prototype_id,
            judgment_id=f"judge-{candidate_id}",
            judgment=judgment,
            total=total,
            scores=dict(judgment["scores"]),
        )
        state["candidates"][candidate_id]["totalScore"] = total
        state["candidates"][candidate_id]["judgmentId"] = f"judge-{candidate_id}"
    closest_state = _unrecoverable_closest_state()
    _apply_resolution(closest_state)
    state["candidateJudgmentResolutionByCandidate"] = copy.deepcopy(
        closest_state.get("candidateJudgmentResolutionByCandidate") or {}
    )
    state["candidates"][PRODUCTION_CLOSEST_CANDIDATE_ID] = copy.deepcopy(
        closest_state["candidates"][PRODUCTION_CLOSEST_CANDIDATE_ID]
    )
    mark_authoritative_winner_selection(state, winner_id="cand-1-winning_card-1-577b91f2")
    state["progressStage"] = "winner_development"
    return state


class TestReasoningDispatchBudgetLedger(unittest.TestCase):
    def test_fourth_call_blocked_before_submission(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        budget.record("builder2_creator")
        budget.record("builder2_judge")
        budget.record("builder2_winner")
        with self.assertRaises(Builder2TournamentError) as ctx:
            budget.reserve("builder2_winner")
        self.assertIn(CALL_BUDGET_EXHAUSTED, str(ctx.exception))

    def test_two_creators_two_judges_blocked_under_limit_three(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        budget.record("builder2_creator")
        budget.record("builder2_judge")
        budget.record("builder2_creator")
        with self.assertRaises(Builder2TournamentError):
            budget.reserve("builder2_judge")

    def test_creator_counts_on_acceptance(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        budget.record("builder2_creator", prototype_id="forgot")
        self.assertEqual(budget.creator_calls_this_run, 1)
        self.assertEqual(budget.actual_openai_dispatches_this_run, 1)

    def test_creator_counts_on_validation_failure(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        entry = budget.reserve("builder2_creator", prototype_id="forgot")
        budget.mark_http_begun(entry)
        budget.mark_response_received(entry)
        budget.finalize(entry, terminal_result="validation_failed")
        self.assertEqual(budget.creator_calls_this_run, 1)

    def test_judge_counts_on_validation_failure(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        entry = budget.reserve("builder2_judge", prototype_id="forgot")
        budget.mark_http_begun(entry)
        budget.mark_response_received(entry)
        budget.finalize(entry, terminal_result="validation_failed")
        self.assertEqual(budget.judge_calls_this_run, 1)

    def test_repair_consumes_slot(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=1)
        budget.record("builder2_judge", call_type="repair")
        self.assertEqual(budget.total_this_run, 1)

    def test_http_dispatched_parsing_failure_consumes_slot(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=1)
        entry = budget.reserve("builder2_creator")
        budget.mark_http_begun(entry)
        budget.mark_response_received(entry)
        budget.finalize(entry, terminal_result="parse_failed")
        self.assertEqual(budget.reasoning_budget_remaining, 0)

    def test_summary_counters_match_ledger(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=5)
        budget.record("builder2_creator", prototype_id="a")
        budget.record("builder2_judge", prototype_id="a")
        budget.record("builder2_creator", prototype_id="b")
        report: Dict[str, Any] = {}
        from engine.builder2_reasoning_dispatch_budget import populate_report_reasoning_dispatch_budget

        populate_report_reasoning_dispatch_budget(report, budget)
        self.assertEqual(report["creatorCallsThisRun"], 2)
        self.assertEqual(report["judgeCallsThisRun"], 1)
        self.assertEqual(report["totalReasoningCallsThisRun"], 3)
        self.assertEqual(report["actualOpenAIDispatchesThisRun"], 3)
        self.assertEqual(len(report["dispatchLedger"]), 3)

    def test_one_slot_left_creator_then_judge_waits(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        budget.record("builder2_creator")
        budget.record("builder2_judge")
        budget.record("builder2_creator")
        self.assertEqual(budget.reasoning_budget_remaining, 0)


class TestProductionShapedWinnerResume(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_inspector_reports_winner_only_plan(self) -> None:
        state = _production_winner_ready_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["resolvedResumeStage"], RESUME_STAGE_WINNER_DEVELOPMENT)
        self.assertEqual(plan["rawMissingJudgmentPrototypeIds"], ["closest"])
        self.assertEqual(plan["actionableMissingJudgmentPrototypeIds"], [])
        self.assertEqual(plan["resolvedUnavailablePrototypeIds"], ["closest"])
        self.assertEqual(plan["unresolvedJudgmentPrototypeIds"], [])
        self.assertEqual(plan["judgeCallsPlanned"], 0)
        self.assertEqual(plan["creatorCallsPlanned"], 0)
        self.assertTrue(plan["winnerDevelopmentCallRequired"])
        self.assertTrue(plan["winnerWouldDispatch"])
        self.assertEqual(plan["recommendedNextInvocationMaxCalls"], 1)
        self.assertEqual(plan["requiredNextReasoningRoles"], ["builder2_winner"])
        self.assertEqual(state.get("winnerCandidateId"), "cand-1-winning_card-1-577b91f2")

    def test_max_calls_one_dispatches_single_winner_call(self) -> None:
        state = _production_winner_ready_state()
        winner_calls = {"count": 0}

        def winner_side_effect(**kwargs: Any) -> Dict[str, Any]:
            winner_calls["count"] += 1
            candidate = kwargs.get("winning_candidate") or _candidate("winning_card")
            candidate_id = str(kwargs.get("candidate_id") or "cand-1-winning_card-1-577b91f2")
            strategy_obj = _strategy(language="he")
            raw = _winner_plan_from_prompt("")
            raw.update(
                methodology_winner_extras(
                    headline_decision="omit",
                    winning_candidate=candidate,
                    strategy=strategy_obj,
                )
            )
            raw.update(single_slogan_contract_extras(slogan_text=str((candidate.get("advertisingClosure") or {}).get("sloganText") or "slogan")))
            raw["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
            raw["prototypeId"] = "winning_card"
            source = build_server_owned_winner_source_reference(
                strategy_foundation=strategy_obj,
                winning_candidate=candidate,
                candidate_id=candidate_id,
            )
            return process_winner_development_response(
                raw,
                source_reference=source,
                winning_candidate=candidate,
                winning_judgment=kwargs.get("winning_judgment"),
            )

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease",
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
            side_effect=winner_side_effect,
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                max_calls=1,
                stop_before_media=True,
                acquire_lease=False,
            )
        self.assertTrue(report.get("ok"))
        self.assertEqual(winner_calls["count"], 1)
        self.assertEqual(report["actualOpenAIDispatchesThisRun"], 1)
        self.assertEqual(report["winnerCallsThisRun"], 1)
        self.assertEqual(report["creatorCallsThisRun"], 0)
        self.assertEqual(report["judgeCallsThisRun"], 0)
        self.assertTrue(report["stoppedBeforeMedia"])
        self.assertEqual(report["finalWinnerCandidateId"], "cand-1-winning_card-1-577b91f2")
        self.assertEqual(report["finalWinnerPrototypeId"], "winning_card")
        self.assertEqual(report["finalWinnerScore"], 89)
        winner_mock.assert_called_once()

    def test_mixed_partial_three_calls_never_fourth(self) -> None:
        state = _production_mixed_partial_state()
        dispatch_count = {"n": 0}

        def creator_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any]]:
            dispatch_count["n"] += 1
            prototype_id = kwargs["prototype_id"]
            return f"cand-1-{prototype_id}-1-new", _candidate(prototype_id)

        def judge_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
            dispatch_count["n"] += 1
            candidate_id = kwargs["candidate_id"]
            judgment = _judgment(candidate_id, total_hint=70, eligible=True)
            return f"judge-{candidate_id}", judgment, 70, dict(judgment["scores"])

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease",
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=creator_side_effect,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=judge_side_effect,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                max_calls=3,
                acquire_lease=False,
            )
        self.assertLessEqual(report["actualOpenAIDispatchesThisRun"], 3)
        self.assertEqual(report["actualOpenAIDispatchesThisRun"], report["totalReasoningCallsThisRun"])
        self.assertLessEqual(dispatch_count["n"], 3)


class TestCreatorClosureVsWinnerDevelopment(unittest.TestCase):
    def test_creator_closure_not_accepted_winner_development(self) -> None:
        state = _production_winner_ready_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertTrue(plan["creatorClosurePresent"])
        self.assertFalse(plan["acceptedWinnerClosurePresent"])
        self.assertFalse(is_valid_persisted_winner_development(state))


if __name__ == "__main__":
    unittest.main()
