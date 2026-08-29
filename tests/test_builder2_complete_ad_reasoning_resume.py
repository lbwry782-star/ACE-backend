"""
Builder2 controlled complete-ad reasoning-only resume tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import (
    CALL_BUDGET_EXHAUSTED,
    ControlledReasoningCallBudget,
    GREENPEACE_PROTOTYPE,
    run_controlled_complete_ad_reasoning_resume,
    validate_controlled_complete_ad_preconditions,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_new_format_config import BUILDER2_NEW_FORMAT_VERSION
from engine.builder2_resume_contract import BUILDER2_RESUME_CONTRACT_VERSION
from engine.builder2_tournament_completion_gate import accepted_creator_count, accepted_judgment_count
from engine.builder2_tournament_contracts import Builder2TournamentError, TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_winner_persistence import is_valid_persisted_winner_development, persist_winner_development_atomically
from engine.builder2_winner_preservation_contract import (
    PARSED_WINNER_RESPONSE_KEY,
    build_server_owned_winner_source_reference,
    process_winner_development_response,
)
from tests.builder2_methodology_fixtures import methodology_winner_extras
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_tournament import _candidate, _judgment, _strategy, _winner_plan_from_prompt


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _missing_greenpeace_state(*, forgot_id: str = "cand-1-forgot-1-test") -> Dict[str, Any]:
    state = _six_prototype_state(judged=5, creators=5)
    state["jobId"] = "job-controlled-reasoning-resume"
    state["tournamentId"] = "tournament-controlled-resume"
    state["builder2ResumeContractVersion"] = BUILDER2_RESUME_CONTRACT_VERSION
    state["builder2NewFormatVersion"] = BUILDER2_NEW_FORMAT_VERSION
    state["productDescription"] = "Controlled resume product"
    state["contentLanguage"] = "he"
    state["schemaVersion"] = TOURNAMENT_STATE_SCHEMA_VERSION
    state.setdefault("rounds", [{"roundIndex": 1, "prototypeIds": state["initialActivePrototypeIds"]}])
    state["provisionalWinnerCandidateId"] = forgot_id
    state["winnerCandidateId"] = None
    for candidate_id, record in state["candidates"].items():
        if forgot_id in candidate_id:
            record["totalScore"] = 90
        else:
            record["totalScore"] = min(int(record.get("totalScore") or 0), 70)
    return state


def _parsed_forgot_winner(candidate_id: str = "cand-1-forgot-1-test") -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("forgot")
    plan = _winner_plan_from_prompt("")
    plan.update(
        methodology_winner_extras(
            headline_decision="omit",
            winning_candidate=candidate,
            strategy=strategy,
        )
    )
    plan["headline"] = ""
    plan["headlineText"] = ""
    plan["headlineCoreKeyword"] = ""
    plan["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
    plan["prototypeId"] = "forgot"
    return {
        "parsed": plan,
        "candidateId": candidate_id,
        "prototypeId": "forgot",
    }


def _make_greenpeace_candidate() -> Tuple[str, Dict[str, Any]]:
    candidate_id = "cand-1-greenpeace_essential_pairing-1-resume"
    return candidate_id, _candidate(GREENPEACE_PROTOTYPE)


def _make_greenpeace_judgment(candidate_id: str, *, total: int = 60) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
    judgment = _judgment(candidate_id, total_hint=total, eligible=True)
    scores = dict(judgment["scores"])
    return f"judge-{candidate_id}", judgment, total, scores


class TestControlledReasoningPreconditions(unittest.TestCase):
    def test_five_five_missing_greenpeace_passes(self) -> None:
        ok, reason = validate_controlled_complete_ad_preconditions(_missing_greenpeace_state())
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_six_six_idempotent_passes(self) -> None:
        state = _six_prototype_state(judged=6, creators=6)
        state["jobId"] = "job-idempotent"
        state["tournamentId"] = "tournament-idempotent"
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_degraded_terminal_uri_lev_passes(self) -> None:
        from tests.test_builder2_tournament_terminal_slots import _uri_lev_production_state

        ok, reason = validate_controlled_complete_ad_preconditions(_uri_lev_production_state())
        self.assertTrue(ok)
        self.assertIsNone(reason)


class TestControlledReasoningCallBudget(unittest.TestCase):
    def test_fourth_call_blocked_before_submission(self) -> None:
        budget = ControlledReasoningCallBudget(max_calls=3)
        budget.record("builder2_creator")
        budget.record("builder2_judge")
        budget.record("builder2_winner")
        with self.assertRaises(Builder2TournamentError) as ctx:
            budget.assert_can_call("builder2_winner")
        self.assertIn(CALL_BUDGET_EXHAUSTED, str(ctx.exception))


class TestControlledReasoningResumeFlow(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _run_with_mocks(
        self,
        state: Dict[str, Any],
        *,
        greenpeace_total: int = 60,
        winner_plan: Dict[str, Any] | None = None,
        creator_raises: Exception | None = None,
        judge_raises: Exception | None = None,
        winner_raises: Exception | None = None,
    ) -> Dict[str, Any]:
        gp_id, gp_candidate = _make_greenpeace_candidate()

        def creator_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any]]:
            if creator_raises:
                raise creator_raises
            return gp_id, deepcopy(gp_candidate)

        def judge_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
            if judge_raises:
                raise judge_raises
            return _make_greenpeace_judgment(kwargs["candidate_id"], total=greenpeace_total)

        def winner_side_effect(**kwargs: Any) -> Dict[str, Any]:
            if winner_raises:
                raise winner_raises
            candidate = kwargs.get("winning_candidate") or _candidate("forgot")
            candidate_id = str(kwargs.get("candidate_id") or "cand-1-forgot-1-test")
            strategy_obj = _strategy(language="he")
            raw = deepcopy(winner_plan) if winner_plan is not None else _winner_plan_from_prompt("")
            if winner_plan is None:
                raw.update(
                    methodology_winner_extras(
                        headline_decision="omit",
                        winning_candidate=candidate,
                        strategy=strategy_obj,
                    )
                )
            raw["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
            raw["prototypeId"] = _clean(kwargs.get("prototype_id")) or candidate.get("prototypeId")
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
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.release_job_lease"
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=creator_side_effect,
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=judge_side_effect,
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
            side_effect=winner_side_effect,
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )
        self._creator_mock = creator_mock
        self._judge_mock = judge_mock
        self._winner_mock = winner_mock
        return report

    def test_missing_greenpeace_creator_and_judge_only(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        report = self._run_with_mocks(state, greenpeace_total=60)
        self.assertTrue(report["ok"])
        self.assertEqual(report["creatorCallsThisRun"], 1)
        self.assertEqual(report["judgeCallsThisRun"], 1)
        self.assertEqual(report["winnerCallsThisRun"], 0)
        self.assertEqual(report["totalReasoningCallsThisRun"], 2)
        self.assertEqual(report["acceptedCreatorCount"], 6)
        self.assertEqual(report["acceptedJudgmentCount"], 6)
        self._creator_mock.assert_called_once()
        self._judge_mock.assert_called_once()
        self._winner_mock.assert_not_called()

    def test_existing_five_creators_not_regenerated(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        original_ids = set(state["acceptedCreatorCandidates"].keys())
        report = self._run_with_mocks(state)
        self.assertTrue(report["ok"])
        self.assertTrue(original_ids.issubset(set(state["acceptedCreatorCandidates"].keys())))

    def test_winner_reused_when_forgot_remains_winner(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        report = self._run_with_mocks(state, greenpeace_total=60)
        self.assertTrue(report["winnerDevelopmentReused"])
        self.assertTrue(report["winnerDevelopmentAccepted"])
        self.assertEqual(report["finalWinnerCandidateId"], "cand-1-forgot-1-test")

    def test_winner_not_reused_when_winner_changes(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        report = self._run_with_mocks(state, greenpeace_total=99)
        self.assertFalse(report["winnerDevelopmentReused"])
        self.assertEqual(report["winnerCallsThisRun"], 1)
        self.assertEqual(report["totalReasoningCallsThisRun"], 3)
        self.assertNotEqual(report["finalWinnerCandidateId"], "cand-1-forgot-1-test")

    def test_stops_before_media(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        report = self._run_with_mocks(state)
        self.assertTrue(report["stoppedBeforeMedia"])
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)
        self.assertEqual(report["nextStage"], "media_prerequisite_validation")

    def test_creator_failure_stops_before_judge(self) -> None:
        state = _missing_greenpeace_state()
        report = self._run_with_mocks(
            state,
            creator_raises=Builder2TournamentError("builder2_creator_validation_failed"),
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "creator_generation")
        self._judge_mock.assert_not_called()
        self._winner_mock.assert_not_called()

    def test_judge_failure_stops_before_winner(self) -> None:
        state = _missing_greenpeace_state()
        report = self._run_with_mocks(
            state,
            judge_raises=Builder2TournamentError("builder2_judge_invalid_response"),
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "judge_generation")
        self._winner_mock.assert_not_called()

    def test_second_execution_zero_model_calls(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        working: Dict[str, Any] = deepcopy(state)

        def save_side_effect(job_id: str, tournament_state: Dict[str, Any]) -> None:
            working.clear()
            working.update(deepcopy(tournament_state))

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=lambda **kwargs: _make_greenpeace_candidate(),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=lambda **kwargs: _make_greenpeace_judgment(kwargs["candidate_id"], total=60),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
            side_effect=save_side_effect,
        ):
            from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume

            first = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )
            self.assertTrue(first["ok"])
            second = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(working),
                acquire_lease=False,
            )
        self.assertTrue(second["ok"])
        self.assertEqual(second["totalReasoningCallsThisRun"], 0)
        winner_mock.assert_not_called()

    def test_same_job_id_preserved(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        report = self._run_with_mocks(state)
        self.assertEqual(report["jobId"], state["jobId"])

    def test_six_way_incomplete_blocks_winner_selection(self) -> None:
        state = _missing_greenpeace_state()
        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=lambda **kwargs: _make_greenpeace_candidate(),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=Builder2TournamentError("builder2_judge_invalid_response"),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.select_global_winner",
        ) as winner_select, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureStage"], "judge_generation")
        winner_select.assert_not_called()

    @patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={})
    @patch("engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=False)
    def test_execution_lease_blocks_concurrent_resume(self, _lease: Any, _raw: Any) -> None:
        state = _missing_greenpeace_state()
        report = run_controlled_complete_ad_reasoning_resume(
            job_id=state["jobId"],
            tournament_state=state,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureReason"], "builder2_complete_ad_reasoning_resume_lease_unavailable")


class TestPostReasoningInspectReadOnly(unittest.TestCase):
    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    @patch("engine.builder2_tournament_store.save_tournament_state")
    def test_post_reasoning_inspector_zero_writes(
        self,
        save_state: Any,
        _job_raw: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        state = _six_prototype_state(judged=6, creators=6)
        state["jobId"] = "job-post-inspect"
        state["reasoningComplete"] = True
        state["mediaStarted"] = False
        state["progressStage"] = "media_prerequisite_validation"
        state["winnerCandidateId"] = "cand-1-forgot-1-test"
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_complete_ad_resume("job-post-inspect")
        self.assertTrue(report["ok"])
        self.assertEqual(report["acceptedCreatorCount"], 6)
        self.assertEqual(report["acceptedJudgmentCount"], 6)
        self.assertFalse(report["mediaStarted"])
        self.assertEqual(report["startImageCalls"], 0)
        self.assertEqual(report["runwaySubmissionCount"], 0)
        self.assertFalse(report["finalVideoAvailable"])
        self.assertEqual(report["redisMutations"], 0)
        save_state.assert_not_called()


class TestBuilder1IsolationReasoningResume(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
