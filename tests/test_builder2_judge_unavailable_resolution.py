"""
Builder2 Judge unavailable resolution — operator Policy A tests.
"""
from __future__ import annotations

import copy
import json
import unittest
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_judge_pending_repair import REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE
from engine.builder2_judge_repair_response_inspect import inspect_judge_repair_response
from engine.builder2_judge_response_ledger import append_judge_response_attempt
from engine.builder2_judge_unavailable_resolution import (
    apply_judge_unavailable_resolution,
    assess_judge_unavailable_resolution,
    run_judge_unavailable_resolution,
)
from engine.builder2_judge_unavailable_resolution_contract import (
    PRODUCTION_CLOSEST_CANDIDATE_ID,
    PRODUCTION_JOB_ID,
    PRODUCTION_SOURCE_JUDGMENT_ID,
    PRODUCTION_SOURCE_PARSED_RESPONSE_FINGERPRINT,
    PRODUCTION_SOURCE_RESPONSE_FINGERPRINT,
    PRODUCTION_TOURNAMENT_ID,
    has_operator_judgment_unavailable_resolution,
)
from engine.builder2_tournament_completion_gate import is_tournament_ready_for_winner_selection
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, load_tournament_state, save_tournament_state
from tests.test_builder2_judge_pending_repair import _closest_empty_assessment_state, _grounded_judgment
from tests.test_builder2_mixed_partial_resume import _production_mixed_partial_state
from tests.test_builder2_tournament import _candidate, _judgment


def _unrecoverable_closest_state() -> Dict[str, Any]:
    state = _closest_empty_assessment_state()
    closest_id = PRODUCTION_CLOSEST_CANDIDATE_ID
    for candidate_id, record in (state.get("candidates") or {}).items():
        if not isinstance(record, dict):
            continue
        prototype_id = record.get("prototypeId")
        if prototype_id in {"closest", "think_small"}:
            record["validationStatus"] = "accepted"
            record["status"] = "accepted"
            record["judgeStatus"] = "pending"
            record.pop("judgeFailure", None)
    normal = state["judgeResponseLedgerByCandidate"][closest_id][0]
    normal["responseFingerprint"] = PRODUCTION_SOURCE_RESPONSE_FINGERPRINT
    normal["parsedResponseFingerprint"] = PRODUCTION_SOURCE_PARSED_RESPONSE_FINGERPRINT
    pending = state["candidates"][closest_id]["pendingJudgeRepair"]
    pending["sourceResponseFingerprint"] = PRODUCTION_SOURCE_RESPONSE_FINGERPRINT
    pending["sourceParsedResponseFingerprint"] = PRODUCTION_SOURCE_PARSED_RESPONSE_FINGERPRINT
    pending["repairDispatched"] = True
    pending["repairOutcomeUnrecoverable"] = True
    pending["lifecycleStage"] = REPAIR_STAGE_REPAIR_RESPONSE_UNRECOVERABLE
    state["metrics"] = {"judgeRepairCalls": 2}
    state["jobId"] = PRODUCTION_JOB_ID
    state["tournamentId"] = PRODUCTION_TOURNAMENT_ID
    state["status"] = "failed"
    state["failureStage"] = "mixed_partial_reasoning"
    return state


def _apply_resolution(state: Dict[str, Any]) -> Dict[str, Any]:
    return apply_judge_unavailable_resolution(
        state,
        candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID,
        expected_job_id=PRODUCTION_JOB_ID,
        expected_tournament_id=PRODUCTION_TOURNAMENT_ID,
        expected_source_judgment_id=PRODUCTION_SOURCE_JUDGMENT_ID,
        expected_source_response_fingerprint=PRODUCTION_SOURCE_RESPONSE_FINGERPRINT,
        expected_source_parsed_response_fingerprint=PRODUCTION_SOURCE_PARSED_RESPONSE_FINGERPRINT,
    )


class TestUnavailableResolutionGuards(unittest.TestCase):
    def test_unrecoverable_blocks_automatic_resume(self) -> None:
        state = _unrecoverable_closest_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertFalse(plan["resumeEligible"])
        self.assertEqual(plan["executorRejectionReason"], "builder2_judge_repair_response_unavailable")

    def test_dry_run_zero_mutations(self) -> None:
        state = _unrecoverable_closest_state()
        before = copy.deepcopy(state)
        assessment = assess_judge_unavailable_resolution(
            state,
            candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID,
            expected_job_id=PRODUCTION_JOB_ID,
            expected_tournament_id=PRODUCTION_TOURNAMENT_ID,
            expected_source_judgment_id=PRODUCTION_SOURCE_JUDGMENT_ID,
            expected_source_response_fingerprint=PRODUCTION_SOURCE_RESPONSE_FINGERPRINT,
            expected_source_parsed_response_fingerprint=PRODUCTION_SOURCE_PARSED_RESPONSE_FINGERPRINT,
        )
        self.assertTrue(assessment["resolutionEligible"])
        self.assertFalse(assessment["stateMutated"])
        self.assertEqual(state, before)

    def test_apply_requires_explicit_apply_env(self) -> None:
        enable_memory_store()
        try:
            state = _unrecoverable_closest_state()
            save_tournament_state(state["jobId"], copy.deepcopy(state))
            with patch("engine.builder2_judge_unavailable_resolution.redis_configured", return_value=True), patch.dict(
                "os.environ",
                {"BUILDER2_JUDGE_UNAVAILABLE_RESOLUTION_JOB_ID": PRODUCTION_JOB_ID},
                clear=False,
            ):
                dry = run_judge_unavailable_resolution(
                    job_id=PRODUCTION_JOB_ID,
                    candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID,
                    apply=False,
                )
            self.assertTrue(dry.get("ok"))
            self.assertTrue(dry.get("dryRun"))
            stored = load_tournament_state(PRODUCTION_JOB_ID)
            assert stored is not None
            self.assertFalse(has_operator_judgment_unavailable_resolution(stored, PRODUCTION_CLOSEST_CANDIDATE_ID))
        finally:
            disable_memory_store()

    def test_apply_validates_fingerprints(self) -> None:
        state = _unrecoverable_closest_state()
        assessment = assess_judge_unavailable_resolution(
            state,
            candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID,
            expected_source_response_fingerprint="bad-fingerprint",
        )
        self.assertFalse(assessment["resolutionEligible"])

    def test_accepted_judgment_blocks_resolution(self) -> None:
        state = _unrecoverable_closest_state()
        from engine.builder2_accepted_judgment_store import persist_accepted_judgment

        judgment = _grounded_judgment(PRODUCTION_CLOSEST_CANDIDATE_ID)
        persist_accepted_judgment(
            state,
            candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID,
            prototype_id="closest",
            judgment_id="judge-accepted",
            judgment=judgment,
            total=70,
            scores=dict(judgment["scores"]),
        )
        assessment = assess_judge_unavailable_resolution(state, candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID)
        self.assertFalse(assessment["resolutionEligible"])

    def test_salvageable_repair_blocks_resolution(self) -> None:
        state = _unrecoverable_closest_state()
        repaired = copy.deepcopy(_grounded_judgment(PRODUCTION_CLOSEST_CANDIDATE_ID, eligible=False))
        append_judge_response_attempt(
            state,
            candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID,
            judgment_id="judge-repair-salvage-block",
            call_type="repair",
            response_text=json.dumps(repaired, ensure_ascii=False),
            parsed=repaired,
        )
        assessment = assess_judge_unavailable_resolution(state, candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID)
        self.assertFalse(assessment["resolutionEligible"])

    def test_winner_blocks_resolution(self) -> None:
        state = _unrecoverable_closest_state()
        state["winnerCandidateId"] = "cand-1-think_small-1-test"
        assessment = assess_judge_unavailable_resolution(state, candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID)
        self.assertFalse(assessment["resolutionEligible"])


class TestUnavailableResolutionApply(unittest.TestCase):
    def test_apply_creates_no_fabricated_judgment(self) -> None:
        state = _unrecoverable_closest_state()
        strategy_before = copy.deepcopy(state["strategyFoundation"])
        creator_before = copy.deepcopy(state["candidates"][PRODUCTION_CLOSEST_CANDIDATE_ID]["creatorOutput"])
        ledger_before = copy.deepcopy(state["judgeResponseLedgerByCandidate"])
        result = _apply_resolution(state)
        self.assertTrue(result["resolutionApplied"])
        self.assertIsNone(state["candidates"][PRODUCTION_CLOSEST_CANDIDATE_ID].get("judgmentId"))
        self.assertNotIn(PRODUCTION_CLOSEST_CANDIDATE_ID, state.get("acceptedJudgments") or {})
        self.assertEqual(state["strategyFoundation"], strategy_before)
        self.assertEqual(state["candidates"][PRODUCTION_CLOSEST_CANDIDATE_ID]["creatorOutput"], creator_before)
        self.assertEqual(state["judgeResponseLedgerByCandidate"], ledger_before)

    def test_apply_marks_excluded_from_winner(self) -> None:
        state = _unrecoverable_closest_state()
        _apply_resolution(state)
        self.assertEqual(state["candidates"][PRODUCTION_CLOSEST_CANDIDATE_ID]["validationStatus"], "judge_unavailable")
        with self.assertRaises(Builder2TournamentError):
            select_global_winner(state)

    def test_repeated_apply_idempotent(self) -> None:
        state = _unrecoverable_closest_state()
        first = _apply_resolution(state)
        snapshot = copy.deepcopy(state)
        second = _apply_resolution(state)
        self.assertTrue(first["resolutionApplied"])
        self.assertTrue(second["alreadyResolved"])
        self.assertFalse(second["resolutionApplied"])
        self.assertEqual(state, snapshot)


class TestUnavailableResolutionPlanner(unittest.TestCase):
    def test_post_resolution_planner_for_closest(self) -> None:
        state = _unrecoverable_closest_state()
        _apply_resolution(state)
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        closest = plan["resumePlanByPrototype"]["closest"]
        self.assertTrue(plan["resumeEligible"])
        self.assertEqual(closest["creatorAction"], "reuse")
        self.assertEqual(closest["judgeAction"], "resolved_unavailable")
        self.assertFalse(closest["normalJudgeCallRequired"])
        self.assertFalse(closest["repairJudgeCallRequired"])
        self.assertTrue(closest["excludedFromWinnerSelection"])
        self.assertTrue(closest["operatorResolutionApplied"])

    def test_post_resolution_call_counts(self) -> None:
        state = _unrecoverable_closest_state()
        _apply_resolution(state)
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertEqual(plan["remainingCreatorNormalCalls"], 4)
        self.assertEqual(plan["remainingJudgeNormalCalls"], 5)
        self.assertEqual(plan["requiredJudgeRepairCalls"], 0)
        self.assertEqual(plan["totalPaidCallsBeforeWinner"], 9)
        self.assertEqual(plan["conditionalWinnerNormalCalls"], 1)
        self.assertEqual(plan["minimumAdditionalPaidReasoningCalls"], 9)
        self.assertEqual(plan["maximumAdditionalPaidReasoningCallsWithoutFutureRepairs"], 10)
        self.assertEqual(plan["perInvocationCallLimit"], 3)
        self.assertEqual(plan["resumePlanByPrototype"]["think_small"]["judgeAction"], "dispatch")

    def test_inspect_reports_operator_fields(self) -> None:
        state = _unrecoverable_closest_state()
        before = inspect_judge_repair_response(state, candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID)
        self.assertTrue(before["operatorResolutionEligible"])
        self.assertFalse(before["operatorResolutionAlreadyApplied"])
        _apply_resolution(state)
        after = inspect_judge_repair_response(state, candidate_id=PRODUCTION_CLOSEST_CANDIDATE_ID)
        self.assertTrue(after["operatorResolutionAlreadyApplied"])


class TestUnavailableTournamentCompletion(unittest.TestCase):
    def _five_judged_state(self, *, eligible: bool) -> Dict[str, Any]:
        state = _unrecoverable_closest_state()
        _apply_resolution(state)
        for prototype_id in [
            "think_small",
            "winning_card",
            "summer_fan",
            "forgot",
            "greenpeace_essential_pairing",
        ]:
            candidate_id = f"cand-1-{prototype_id}-1-test"
            if prototype_id not in {"think_small"}:
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
            from engine.builder2_accepted_judgment_store import persist_accepted_judgment

            judgment = _judgment(candidate_id, eligible=eligible)
            persist_accepted_judgment(
                state,
                candidate_id=candidate_id,
                prototype_id=prototype_id,
                judgment_id=f"judge-{candidate_id}",
                judgment=judgment,
                total=70,
                scores=dict(judgment["scores"]),
            )
        return state

    def test_five_judgments_allow_winner_selection(self) -> None:
        state = self._five_judged_state(eligible=True)
        self.assertTrue(is_tournament_ready_for_winner_selection(state))
        winner = select_global_winner(state)
        self.assertNotEqual(winner, PRODUCTION_CLOSEST_CANDIDATE_ID)

    def test_all_five_ineligible_stops_without_winner(self) -> None:
        state = self._five_judged_state(eligible=False)
        self.assertTrue(is_tournament_ready_for_winner_selection(state))
        with self.assertRaises(Builder2TournamentError) as ctx:
            select_global_winner(state)
        self.assertEqual(ctx.exception.args[0], "builder2_no_factually_eligible_candidate")


class TestUnavailableChunkedResume(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_resume_skips_closest_after_resolution(self) -> None:
        state = _unrecoverable_closest_state()
        _apply_resolution(state)
        save_tournament_state(state["jobId"], copy.deepcopy(state))
        calls = {"closest": 0}

        def judge_side_effect(**kwargs: Any):
            candidate_id = str(kwargs["candidate_id"])
            if "closest" in candidate_id:
                calls["closest"] += 1
            judgment = _grounded_judgment(candidate_id)
            return f"judge-{candidate_id}", judgment, 70, dict(judgment["scores"])

        with patch("engine.builder2_complete_ad_reasoning_resume.judge_candidate", side_effect=judge_side_effect), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease",
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.redis_configured",
            return_value=True,
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.video_job_get_raw",
            return_value={},
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=lambda **kwargs: (
                f"cand-1-{kwargs['prototype_id']}-1-new",
                _candidate(kwargs["prototype_id"]),
            ),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                max_calls=3,
                acquire_lease=False,
            )
        self.assertTrue(report.get("ok"))
        self.assertEqual(calls["closest"], 0)


if __name__ == "__main__":
    unittest.main()
