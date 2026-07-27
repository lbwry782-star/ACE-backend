"""
Builder2 reasoning resume and creator terminal failure observability tests — mocks only.
"""
from __future__ import annotations

import logging
import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_complete_ad_reasoning_resume import (
    GREENPEACE_PROTOTYPE,
    main as reasoning_resume_main,
    run_controlled_complete_ad_reasoning_resume,
)
from engine.builder2_creator import generate_creator_candidate
from engine.builder2_tournament_contracts import Builder2TournamentError, TOURNAMENT_STATE_SCHEMA_VERSION
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY
from tests.test_builder2_complete_ad_reasoning_resume import (
    _make_greenpeace_candidate,
    _missing_greenpeace_state,
    _parsed_forgot_winner,
)
from tests.test_builder2_tournament import _candidate, _strategy


def _resume_state() -> Dict[str, Any]:
    state = _missing_greenpeace_state()
    state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
    return state


class TestCreatorFailureObservability(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _generate(self, *, llm_side_effect: Any) -> None:
        strategy = _strategy(language="he")
        state = {"jobId": "job-obs", "tournamentId": "tournament-obs"}
        with patch("engine.builder2_creator._invoke_creator_model", side_effect=llm_side_effect):
            with self.assertRaises(Builder2TournamentError):
                generate_creator_candidate(
                    product_name="ACE Product",
                    product_description="desc",
                    language="he",
                    strategy_foundation=strategy,
                    prototype_id=GREENPEACE_PROTOTYPE,
                    round_index=1,
                    attempt_number=1,
                    runway_mode="gen4.5",
                    state=state,
                    single_attempt_only=True,
                )

    def test_openai_exception_logs_traceback(self) -> None:
        with self.assertLogs("engine.builder2_creator", level="ERROR") as logs:
            self._generate(llm_side_effect=RuntimeError("connection reset"))

        joined = "\n".join(logs.output)
        self.assertIn("BUILDER2_CREATOR_OPENAI_FAILED", joined)
        self.assertIn("exceptionClass=RuntimeError", joined)
        self.assertIn("Traceback", joined)

    def test_empty_output_text_produces_terminal_diagnostic(self) -> None:
        with self.assertLogs("engine.builder2_creator", level="ERROR") as logs:
            self._generate(llm_side_effect=lambda **kwargs: ("", None))

        joined = "\n".join(logs.output)
        self.assertIn("BUILDER2_CREATOR_PARSE_FAILED", joined)
        self.assertIn("BUILDER2_CREATOR_REJECTED", joined)
        self.assertIn("builder2_creator_empty_response", joined)

    def test_malformed_json_produces_parsing_diagnostic(self) -> None:
        with self.assertLogs("engine.builder2_creator", level="ERROR") as logs:
            self._generate(llm_side_effect=lambda **kwargs: ("not json at all", None))

        joined = "\n".join(logs.output)
        self.assertIn("BUILDER2_CREATOR_PARSE_FAILED", joined)
        self.assertIn("builder2_creator_malformed_response", joined)

    def test_validation_rejection_reports_precise_reason(self) -> None:
        broken = _candidate(GREENPEACE_PROTOTYPE)
        broken["schemaVersion"] = "wrong"
        with self.assertLogs("engine.builder2_creator", level="ERROR") as logs:
            self._generate(
                llm_side_effect=lambda **kwargs: (
                    __import__("json").dumps(broken),
                    None,
                )
            )

        joined = "\n".join(logs.output)
        self.assertIn("BUILDER2_CREATOR_REJECTED", joined)
        self.assertIn("builder2_creator_schema_invalid:schemaVersion", joined)


class TestReasoningResumeFailureObservability(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_openai_exception_surfaces_resume_terminal_logs(self) -> None:
        state = _resume_state()
        working = deepcopy(state)
        original_creators = deepcopy(working["acceptedCreatorCandidates"])
        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state"
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=Builder2TournamentError("builder2_creator_openai_failed:APIConnectionError:timeout"),
        ):
            with self.assertLogs("engine.builder2_complete_ad_reasoning_resume", level="ERROR") as logs:
                report = run_controlled_complete_ad_reasoning_resume(
                    job_id=state["jobId"],
                    tournament_state=working,
                    acquire_lease=False,
                )

        self.assertFalse(report["ok"])
        joined = "\n".join(logs.output)
        self.assertIn("BUILDER2_COMPLETE_AD_REASONING_RESUME_TERMINAL_FAILURE", joined)
        self.assertEqual(working["acceptedCreatorCandidates"], original_creators)
        self.assertNotIn(GREENPEACE_PROTOTYPE, {rec.get("prototypeId") for rec in working["acceptedCreatorCandidates"].values()})

    def test_main_emits_final_reasoning_resume_failed(self) -> None:
        env = {
            "BUILDER2_COMPLETE_AD_REASONING_RESUME_JOB_ID": "job-obs-main",
            "BUILDER2_COMPLETE_AD_REASONING_RESUME_MAX_CALLS": "3",
            "BUILDER2_COMPLETE_AD_REASONING_RESUME_STOP_BEFORE_MEDIA": "true",
        }
        with patch.dict(os.environ, env, clear=False), patch(
            "engine.builder2_complete_ad_reasoning_resume.run_controlled_complete_ad_reasoning_resume",
            return_value={"ok": False, "failureStage": "creator_generation", "failureReason": "builder2_creator_empty_response"},
        ):
            with self.assertLogs("engine.builder2_complete_ad_reasoning_resume", level="ERROR") as logs:
                exit_code = reasoning_resume_main()

        self.assertEqual(exit_code, 1)
        self.assertTrue(any("BUILDER2_REASONING_RESUME_FAILED" in line for line in logs.output))

    def test_no_media_calls_on_creator_failure(self) -> None:
        state = _resume_state()
        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state"
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=Builder2TournamentError("builder2_creator_empty_response"),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate"
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate"
        ) as winner_mock:
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )

        self.assertFalse(report["ok"])
        judge_mock.assert_not_called()
        winner_mock.assert_not_called()
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)


class TestBuilder1IsolationObservability(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
