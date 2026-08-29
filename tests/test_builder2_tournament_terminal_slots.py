"""
Builder2 tournament terminal-slot completion gate tests.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict

from engine.builder2_accepted_judgment_store import persist_accepted_judgment
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_tournament_completion_gate import (
    PROTOTYPE_TERMINAL_OUTCOME_CREATOR_REJECTED,
    PROTOTYPE_TERMINAL_OUTCOME_UNRESOLVED,
    TOURNAMENT_INCOMPLETE_BEFORE_WINNER,
    assert_tournament_ready_for_winner_selection,
    collect_terminal_prototype_slots,
    is_prototype_slot_terminal,
    is_tournament_ready_for_winner_selection,
    missing_creator_prototype_ids,
    resolve_prototype_terminal_outcome,
    tournament_resolution_summary,
)
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import select_global_winner
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _candidate, _judgment


_URI_LEV_JOB_ID = "62cd523c-2ca3-47ae-8f23-181d7617baf4"
_PROTOTYPES = [
    "winning_card",
    "summer_fan",
    "forgot",
    "closest",
    "think_small",
    "greenpeace_essential_pairing",
]


def _reject_creator(state: Dict[str, Any], prototype_id: str) -> None:
    candidate_id = f"cand-1-{prototype_id}-1-uri"
    state["candidates"][candidate_id] = {
        "candidateId": candidate_id,
        "prototypeId": prototype_id,
        "validationStatus": "creator_rejected",
        "status": "creator_rejected",
        "failureReason": "builder2_creator_validation_failed:newProductClaimsIntroduced",
    }


def _accept_creator_with_judgment(
    state: Dict[str, Any],
    *,
    prototype_id: str,
    eligible: bool,
    total_score: int,
) -> str:
    candidate_id = f"cand-1-{prototype_id}-1-uri"
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
        "validationStatus": "accepted",
        "creatorAcceptanceStatus": "accepted",
        "status": "accepted",
        "creatorOutput": candidate,
        "creatorSnapshot": candidate,
    }
    judgment = _judgment(candidate_id, eligible=eligible, total_hint=total_score)
    persist_accepted_judgment(
        state,
        candidate_id=candidate_id,
        prototype_id=prototype_id,
        judgment_id=f"judge-{candidate_id}",
        judgment=judgment,
        total=total_score,
        scores=dict(judgment["scores"]),
    )
    state["candidates"][candidate_id]["eligible"] = eligible
    state["candidates"][candidate_id]["totalScore"] = total_score
    return candidate_id


def _uri_lev_production_state() -> Dict[str, Any]:
    state = _six_prototype_state(judged=0, creators=0)
    state["jobId"] = _URI_LEV_JOB_ID
    state["tournamentId"] = "9d789e1e-7e4a-4ef4-b72e-642da8083788"
    state["candidates"] = {}
    state["acceptedCreatorCandidates"] = {}
    state["acceptedJudgments"] = {}
    state["judgments"] = {}
    for prototype_id in ("winning_card", "closest", "think_small"):
        _reject_creator(state, prototype_id)
    _accept_creator_with_judgment(state, prototype_id="forgot", eligible=True, total_score=87)
    _accept_creator_with_judgment(state, prototype_id="summer_fan", eligible=True, total_score=86)
    _accept_creator_with_judgment(state, prototype_id="greenpeace_essential_pairing", eligible=False, total_score=70)
    return state


class TestNormalSixAcceptedBehavior(unittest.TestCase):
    def test_six_accepted_and_judged_still_selects_winner(self) -> None:
        state = _six_prototype_state(judged=6, creators=6)
        assert_tournament_ready_for_winner_selection(state)
        winner_id = select_global_winner(state)
        self.assertTrue(winner_id)


class TestDegradedTerminalSlots(unittest.TestCase):
    def test_three_accepted_three_rejected_all_terminal(self) -> None:
        state = _uri_lev_production_state()
        summary = tournament_resolution_summary(state)
        self.assertEqual(summary["assignedPrototypeCount"], 6)
        self.assertEqual(summary["terminalPrototypeCount"], 6)
        self.assertEqual(summary["acceptedCreatorCount"], 3)
        self.assertEqual(summary["rejectedCreatorCount"], 3)
        self.assertEqual(summary["acceptedJudgmentCount"], 3)
        self.assertEqual(summary["eligibleCandidateCount"], 2)
        self.assertTrue(summary["winnerSelectionReady"])
        self.assertTrue(summary["degradedTournament"])
        self.assertTrue(summary["readyForAuthoritativeWinnerSelection"])

    def test_uri_lev_regression_selects_forgot_over_summer_fan(self) -> None:
        state = _uri_lev_production_state()
        assert_tournament_ready_for_winner_selection(state)
        winner_id = select_global_winner(state)
        self.assertEqual(winner_id, "cand-1-forgot-1-uri")

    def test_rejected_prototypes_not_counted_as_missing_creators(self) -> None:
        state = _uri_lev_production_state()
        self.assertEqual(missing_creator_prototype_ids(state), [])

    def test_structurally_rejected_slot_is_terminal(self) -> None:
        state = _uri_lev_production_state()
        self.assertEqual(
            resolve_prototype_terminal_outcome(state, "winning_card"),
            PROTOTYPE_TERMINAL_OUTCOME_CREATOR_REJECTED,
        )
        self.assertTrue(is_prototype_slot_terminal(state, "winning_card"))

    def test_judge_ineligible_slot_terminal_but_not_in_winner_pool(self) -> None:
        state = _uri_lev_production_state()
        self.assertTrue(is_prototype_slot_terminal(state, "greenpeace_essential_pairing"))
        eligible_ids = tournament_resolution_summary(state)["eligibleCandidateIds"]
        self.assertNotIn("cand-1-greenpeace_essential_pairing-1-uri", eligible_ids)


class TestMinimumWinnerRequirement(unittest.TestCase):
    def test_single_eligible_candidate_wins(self) -> None:
        state = _uri_lev_production_state()
        for candidate_id in list(state["candidates"].keys()):
            if candidate_id.endswith("summer_fan-1-uri"):
                state["candidates"][candidate_id]["eligible"] = False
        summary = tournament_resolution_summary(state)
        self.assertEqual(summary["eligibleCandidateCount"], 1)
        assert_tournament_ready_for_winner_selection(state)
        self.assertEqual(select_global_winner(state), "cand-1-forgot-1-uri")

    def test_zero_eligible_candidates_fails_explicitly(self) -> None:
        state = _uri_lev_production_state()
        for candidate_id, record in state["candidates"].items():
            if record.get("validationStatus") == "accepted":
                record["eligible"] = False
        assert_tournament_ready_for_winner_selection(state)
        with self.assertRaises(Builder2TournamentError) as ctx:
            select_global_winner(state)
        self.assertEqual(ctx.exception.args[0], "builder2_no_factually_eligible_candidate")


class TestIncompleteVsRejected(unittest.TestCase):
    def test_five_terminal_one_missing_blocks_winner(self) -> None:
        state = _uri_lev_production_state()
        state["candidates"].pop("cand-1-winning_card-1-uri", None)
        summary = tournament_resolution_summary(state)
        self.assertEqual(summary["terminalPrototypeCount"], 5)
        self.assertFalse(summary["readyForAuthoritativeWinnerSelection"])
        with self.assertRaises(Builder2TournamentError) as ctx:
            assert_tournament_ready_for_winner_selection(state)
        self.assertIn(TOURNAMENT_INCOMPLETE_BEFORE_WINNER, ctx.exception.args[0])

    def test_unattempted_prototype_is_unresolved_not_terminal(self) -> None:
        state = _uri_lev_production_state()
        state["candidates"].pop("cand-1-winning_card-1-uri", None)
        self.assertEqual(
            resolve_prototype_terminal_outcome(state, "winning_card"),
            PROTOTYPE_TERMINAL_OUTCOME_UNRESOLVED,
        )

    def test_accepted_creator_without_judgment_not_terminal(self) -> None:
        state = _uri_lev_production_state()
        candidate_id = "cand-1-forgot-1-uri"
        state["acceptedJudgments"].pop(candidate_id, None)
        state["candidates"][candidate_id]["judgmentId"] = None
        state["judgments"].pop(f"judge-{candidate_id}", None)
        self.assertFalse(is_prototype_slot_terminal(state, "forgot"))
        self.assertFalse(is_tournament_ready_for_winner_selection(state))


class TestJudgeUnavailableTerminal(unittest.TestCase):
    def test_judge_unavailable_counts_as_terminal(self) -> None:
        state = _uri_lev_production_state()
        candidate_id = "cand-1-forgot-1-uri"
        state["candidates"][candidate_id]["validationStatus"] = "judge_unavailable"
        state["candidates"][candidate_id]["status"] = "judge_unavailable"
        state["acceptedJudgments"].pop(candidate_id, None)
        state["judgments"].pop(f"judge-{candidate_id}", None)
        self.assertTrue(is_prototype_slot_terminal(state, "forgot"))
        terminal = collect_terminal_prototype_slots(state)
        self.assertIn("forgot", terminal)


class TestResumeCompatibility(unittest.TestCase):
    def test_persisted_three_accepted_three_rejected_advances_to_winner_without_creator_judge_dispatch(
        self,
    ) -> None:
        state = _uri_lev_production_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertTrue(is_tournament_ready_for_winner_selection(state))
        self.assertEqual(plan["resolvedResumeStage"], "winner_selection")
        self.assertEqual(plan["creatorCallsPlanned"], 0)
        self.assertEqual(plan["judgeCallsPlanned"], 0)
        self.assertTrue(plan["resumeEligible"])
        self.assertIn("builder2_winner_if_winner_changes", plan["expectedNextReasoningRoles"])


class TestRejectedNeverEntersWinnerPool(unittest.TestCase):
    def test_rejected_creator_cannot_be_selected(self) -> None:
        state = _uri_lev_production_state()
        rejected_id = "cand-1-winning_card-1-uri"
        state["candidates"][rejected_id]["eligible"] = True
        state["candidates"][rejected_id]["totalScore"] = 999
        state["candidates"][rejected_id]["judgmentId"] = "fake-judge"
        winner_id = select_global_winner(state)
        self.assertNotEqual(winner_id, rejected_id)


if __name__ == "__main__":
    unittest.main()
