"""
Builder2 Reasoning Resume executor tests for terminal-slot degraded tournaments — mocks only.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import (
    run_controlled_complete_ad_reasoning_resume,
    validate_controlled_complete_ad_preconditions,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_complete_ad_resume_plan import resolve_complete_ad_canonical_resume_plan
from engine.builder2_tournament_completion_gate import is_tournament_ready_for_winner_selection
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_manager import select_global_winner
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_persistence import is_valid_persisted_winner_development
from engine.builder2_winner_preservation_contract import (
    build_server_owned_winner_source_reference,
    process_winner_development_response,
)
from tests.builder2_methodology_fixtures import methodology_winner_extras
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt
from tests.test_builder2_tournament_terminal_slots import _uri_lev_production_state


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _winner_side_effect(**kwargs: Any) -> Dict[str, Any]:
    candidate = kwargs.get("winning_candidate") or _candidate("forgot")
    candidate_id = str(kwargs.get("candidate_id") or "cand-1-forgot-1-uri")
    strategy_obj = _strategy(language="he")
    raw = _winner_plan_from_prompt("")
    raw.update(
        methodology_winner_extras(
            headline_decision="omit",
            winning_candidate=candidate,
            strategy=strategy_obj,
        )
    )
    raw["advertisingClosure"] = deepcopy(candidate["advertisingClosure"])
    raw["prototypeId"] = _clean(kwargs.get("prototype_id")) or "forgot"
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


class TestDegradedTerminalReasoningResumePreconditions(unittest.TestCase):
    def test_uri_lev_shape_passes_controlled_preconditions(self) -> None:
        state = _uri_lev_production_state()
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok, reason)
        self.assertIsNone(reason)

    def test_inspector_executor_would_accept_uri_lev(self) -> None:
        state = _uri_lev_production_state()
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertTrue(plan["executorWouldAcceptState"])
        self.assertIsNone(plan["executorRejectionReason"])
        self.assertEqual(plan["resolvedResumeStage"], "winner_selection")


class TestDegradedTerminalReasoningResumeExecutor(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _run_uri_lev(
        self,
        state: Dict[str, Any],
        *,
        max_calls: int = 1,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Any, Any, Any]:
        working: Dict[str, Any] = deepcopy(state)

        def save_side_effect(_job_id: str, tournament_state: Dict[str, Any]) -> None:
            working.clear()
            working.update(deepcopy(tournament_state))

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
            side_effect=_winner_side_effect,
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
            side_effect=save_side_effect,
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                max_calls=max_calls,
                stop_before_media=True,
                acquire_lease=False,
            )
        return report, working, creator_mock, judge_mock, winner_mock

    def test_uri_lev_winner_selection_and_development(self) -> None:
        state = _uri_lev_production_state()
        report, working, creator_mock, judge_mock, winner_mock = self._run_uri_lev(state)

        self.assertTrue(report["ok"])
        self.assertNotEqual(report.get("failureReason"), "builder2_complete_ad_reasoning_resume_unexpected_partial_state")
        self.assertEqual(report["creatorCallsThisRun"], 0)
        self.assertEqual(report["judgeCallsThisRun"], 0)
        self.assertEqual(report["winnerCallsThisRun"], 1)
        self.assertEqual(report["totalReasoningCallsThisRun"], 1)
        self.assertTrue(report["strategyReused"])
        self.assertEqual(report["finalWinnerCandidateId"], "cand-1-forgot-1-uri")
        self.assertEqual(report["finalWinnerPrototypeId"], "forgot")
        self.assertEqual(report["finalWinnerScore"], 87)
        self.assertTrue(report["stoppedBeforeMedia"])
        self.assertEqual(report["nextStage"], "media_prerequisite_validation")
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)
        creator_mock.assert_not_called()
        judge_mock.assert_not_called()
        winner_mock.assert_called_once()
        self.assertEqual(working.get("winnerCandidateId"), "cand-1-forgot-1-uri")
        self.assertTrue(working.get("winnerSelectionFinal"))
        self.assertTrue(is_valid_persisted_winner_development(working))

    def test_uri_lev_idempotent_second_run(self) -> None:
        state = _uri_lev_production_state()
        first, working, _, _, winner_mock = self._run_uri_lev(state)
        self.assertTrue(first["ok"])
        winner_mock.reset_mock()

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
        ) as winner_mock2, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            second = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(working),
                max_calls=1,
                stop_before_media=True,
                acquire_lease=False,
            )

        self.assertTrue(second["ok"])
        self.assertEqual(second["totalReasoningCallsThisRun"], 0)
        self.assertEqual(second["finalWinnerCandidateId"], "cand-1-forgot-1-uri")
        creator_mock.assert_not_called()
        judge_mock.assert_not_called()
        winner_mock2.assert_not_called()

    def test_five_terminal_one_missing_not_winner_selection_ready(self) -> None:
        state = _uri_lev_production_state()
        state["candidates"].pop("cand-1-winning_card-1-uri", None)
        self.assertFalse(is_tournament_ready_for_winner_selection(state))
        plan = resolve_complete_ad_canonical_resume_plan(state, read_only=True)
        self.assertNotEqual(plan["resolvedResumeStage"], "winner_selection")
        ok, reason = validate_controlled_complete_ad_preconditions(state)
        self.assertFalse(ok)
        self.assertIn("unexpected_partial_state", reason or "")

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.select_global_winner",
        ) as winner_select, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
        ) as winner_dev, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                max_calls=1,
                stop_before_media=True,
                acquire_lease=False,
            )
        winner_select.assert_not_called()
        winner_dev.assert_not_called()
        self.assertFalse(report["ok"])

    def test_single_eligible_candidate_wins(self) -> None:
        state = _uri_lev_production_state()
        for candidate_id in list(state["candidates"].keys()):
            if candidate_id.endswith("summer_fan-1-uri"):
                state["candidates"][candidate_id]["eligible"] = False
        self.assertEqual(select_global_winner(state), "cand-1-forgot-1-uri")
        report, working, creator_mock, judge_mock, winner_mock = self._run_uri_lev(state)
        self.assertTrue(report["ok"])
        self.assertEqual(report["finalWinnerCandidateId"], "cand-1-forgot-1-uri")
        self.assertEqual(report["winnerCallsThisRun"], 1)
        creator_mock.assert_not_called()
        judge_mock.assert_not_called()
        winner_mock.assert_called_once()
        self.assertTrue(is_valid_persisted_winner_development(working))

    def test_zero_eligible_fails_without_winner_development(self) -> None:
        state = _uri_lev_production_state()
        for _candidate_id, record in state["candidates"].items():
            if record.get("validationStatus") == "accepted":
                record["eligible"] = False
        ok, _reason = validate_controlled_complete_ad_preconditions(state)
        self.assertTrue(ok)

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
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
        self.assertFalse(report["ok"])
        self.assertEqual(report["failureReason"], "builder2_no_factually_eligible_candidate")
        self.assertEqual(report["winnerCallsThisRun"], 0)
        winner_mock.assert_not_called()


class TestDegradedTerminalInspectExecutorAlignment(unittest.TestCase):
    @patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_complete_ad_resume_inspect._read_raw")
    @patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={})
    @patch("engine.builder2_tournament_store.save_tournament_state")
    def test_inspector_matches_executor_preconditions(
        self,
        save_state: Any,
        _job_raw: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        state = _uri_lev_production_state()
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_complete_ad_resume(state["jobId"])
        self.assertTrue(report["ok"])
        self.assertTrue(report["executorWouldAcceptState"])
        self.assertIsNone(report["executorRejectionReason"])
        self.assertEqual(report["resolvedResumeStage"], "winner_selection")
        save_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
