"""
Builder2 Creator advertisingClosure.durationSeconds normalization and offline recovery tests.
"""
from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple
from unittest.mock import patch

from engine.builder2_complete_ad_contract import (
    normalize_creator_advertising_closure_execution_metadata,
    resolve_canonical_creator_end_card_duration_seconds,
    validate_creator_complete_ad_fields,
)
from engine.builder2_complete_ad_creator_recovery import (
    can_offline_revalidate_rejected_creator,
    persist_rejected_creator_parsed_response,
)
from engine.builder2_complete_ad_reasoning_resume import (
    GREENPEACE_PROTOTYPE,
    run_controlled_complete_ad_reasoning_resume,
)
from engine.builder2_complete_ad_resume_inspect import inspect_builder2_complete_ad_resume
from engine.builder2_creator_normalization import normalize_creator_candidate
from engine.builder2_new_format_config import resolve_builder2_end_card_duration_seconds
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY
from tests.test_builder2_complete_ad_flow import _six_prototype_state
from tests.test_builder2_complete_ad_reasoning_resume import (
    _make_greenpeace_judgment,
    _missing_greenpeace_state,
    _parsed_forgot_winner,
)
from tests.test_builder2_tournament import _candidate, _strategy


def _canonical_duration() -> float:
    return resolve_canonical_creator_end_card_duration_seconds()


class TestCreatorDurationNormalization(unittest.TestCase):
    def test_missing_duration_set_from_canonical_source(self) -> None:
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"].pop("durationSeconds", None)
        normalized, resolved = normalize_creator_candidate(
            candidate,
            assigned_prototype_id=GREENPEACE_PROTOTYPE,
            prototype_display_name=GREENPEACE_PROTOTYPE,
            strategy_foundation=_strategy(language="he"),
            job_id="job-duration",
            candidate_id="cand-duration",
        )
        self.assertEqual(normalized["advertisingClosure"]["durationSeconds"], _canonical_duration())
        self.assertIn("advertisingClosure.durationSeconds", resolved)

    def test_numeric_string_normalized(self) -> None:
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = "7"
        normalized, _ = normalize_creator_advertising_closure_execution_metadata(
            candidate,
            job_id="job-duration",
            candidate_id="cand-duration",
            prototype_id=GREENPEACE_PROTOTYPE,
        )
        self.assertEqual(normalized["advertisingClosure"]["durationSeconds"], _canonical_duration())

    def test_seconds_suffix_normalized_when_parseable(self) -> None:
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = "7 seconds"
        normalized, changed = normalize_creator_advertising_closure_execution_metadata(
            candidate,
            job_id="job-duration",
            candidate_id="cand-duration",
            prototype_id=GREENPEACE_PROTOTYPE,
        )
        self.assertTrue(changed)
        self.assertEqual(normalized["advertisingClosure"]["durationSeconds"], _canonical_duration())

    def test_wrong_model_duration_replaced_by_canonical(self) -> None:
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = 10
        slogan_before = candidate["advertisingClosure"]["sloganText"]
        normalized, changed = normalize_creator_advertising_closure_execution_metadata(
            candidate,
            job_id="job-duration",
            candidate_id="cand-duration",
            prototype_id=GREENPEACE_PROTOTYPE,
        )
        self.assertTrue(changed)
        self.assertEqual(normalized["advertisingClosure"]["durationSeconds"], _canonical_duration())
        self.assertEqual(normalized["advertisingClosure"]["sloganText"], slogan_before)

    def test_creative_fields_not_modified(self) -> None:
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = 10
        candidate["advertisingClosure"]["sloganText"] = "קרוב יותר ממה שחשבת"
        candidate["semanticBridge"]["keyWordOrConcept"] = "closeness"
        before = deepcopy(candidate)
        normalized, _ = normalize_creator_candidate(
            candidate,
            assigned_prototype_id=GREENPEACE_PROTOTYPE,
            prototype_display_name=GREENPEACE_PROTOTYPE,
            strategy_foundation=_strategy(language="he"),
        )
        self.assertEqual(normalized["advertisingClosure"]["sloganText"], before["advertisingClosure"]["sloganText"])
        self.assertEqual(normalized["semanticBridge"]["keyWordOrConcept"], before["semanticBridge"]["keyWordOrConcept"])
        self.assertEqual(normalized["conceptSummary"], before["conceptSummary"])

    def test_methodology_validation_still_active_after_duration_fix(self) -> None:
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = 10
        candidate["advertisingClosure"]["productNameText"] = "Wrong Product"
        normalized, _ = normalize_creator_candidate(
            candidate,
            assigned_prototype_id=GREENPEACE_PROTOTYPE,
            prototype_display_name=GREENPEACE_PROTOTYPE,
            strategy_foundation=_strategy(language="he"),
        )
        with self.assertRaises(Builder2TournamentError):
            validate_creator_complete_ad_fields(
                normalized,
                strategy_foundation=_strategy(language="he"),
                assigned_prototype_id=GREENPEACE_PROTOTYPE,
                product_name="ACE Product",
            )

    def test_canonical_source_matches_end_card_config(self) -> None:
        self.assertEqual(
            resolve_canonical_creator_end_card_duration_seconds(),
            resolve_builder2_end_card_duration_seconds(),
        )


class TestOfflineGreenpeaceRecovery(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def _rejected_greenpeace_state(self, *, duration: Any = 10) -> Dict[str, Any]:
        state = _missing_greenpeace_state()
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = duration
        candidate["advertisingClosure"]["productNameText"] = "ACE Product"
        candidate_id = "cand-1-greenpeace_essential_pairing-1-33964989"
        persist_rejected_creator_parsed_response(
            state,
            candidate_id=candidate_id,
            prototype_id=GREENPEACE_PROTOTYPE,
            round_index=1,
            attempt_number=1,
            parsed=candidate,
            failure_reason="builder2_creator_validation_failed:advertisingClosure.durationSeconds",
        )
        return state

    def test_rejected_greenpeace_revalidates_after_normalization(self) -> None:
        state = self._rejected_greenpeace_state(duration=10)
        candidate_id = "cand-1-greenpeace_essential_pairing-1-33964989"
        ok, reason = can_offline_revalidate_rejected_creator(
            state,
            candidate_id=candidate_id,
            product_name="ACE Product",
        )
        self.assertTrue(ok, reason)

    def test_inspector_marks_offline_revalidatable(self) -> None:
        state = self._rejected_greenpeace_state(duration=10)
        with patch("engine.builder2_complete_ad_resume_inspect.redis_configured", return_value=True), patch(
            "engine.builder2_complete_ad_resume_inspect._read_raw",
            return_value=deepcopy(state),
        ), patch("engine.builder2_complete_ad_resume_inspect.video_job_get_raw", return_value={}):
            report = inspect_builder2_complete_ad_resume(state["jobId"])
        self.assertTrue(report["rejectedCreatorResponseAvailable"])
        self.assertTrue(report["rejectedCreatorOfflineRevalidatable"])

    def test_offline_recovery_zero_creator_openai_calls(self) -> None:
        state = self._rejected_greenpeace_state(duration=10)
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()

        def judge_side_effect(**kwargs: Any) -> Tuple[str, Dict[str, Any], int, Dict[str, int]]:
            return _make_greenpeace_judgment(kwargs["candidate_id"], total=60)

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
        ) as creator_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=judge_side_effect,
        ) as judge_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.develop_builder2_winning_candidate",
        ) as winner_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["creatorCallsThisRun"], 0)
        creator_mock.assert_not_called()
        judge_mock.assert_called_once()
        winner_mock.assert_not_called()

    def test_failure_report_preserves_five_five_counts(self) -> None:
        state = _missing_greenpeace_state()
        broken = _candidate(GREENPEACE_PROTOTYPE)
        broken["schemaVersion"] = "wrong"
        candidate_id = "cand-1-greenpeace_essential_pairing-1-33964989"
        persist_rejected_creator_parsed_response(
            state,
            candidate_id=candidate_id,
            prototype_id=GREENPEACE_PROTOTYPE,
            round_index=1,
            attempt_number=1,
            parsed=broken,
            failure_reason="builder2_creator_schema_invalid:schemaVersion",
        )

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.generate_creator_candidate",
            side_effect=Builder2TournamentError("builder2_creator_schema_invalid:schemaVersion"),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["acceptedCreatorCount"], 5)
        self.assertEqual(report["acceptedJudgmentCount"], 5)


class TestRejectedCreatorCallAccounting(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()

    def test_http200_rejected_creator_counts_one_reasoning_call(self) -> None:
        state = _missing_greenpeace_state()
        broken = _candidate(GREENPEACE_PROTOTYPE)
        broken["schemaVersion"] = "wrong"

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ), patch(
            "engine.builder2_creator._invoke_creator_model",
            return_value=(json.dumps(broken), None),
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["creatorCallsThisRun"], 1)
        self.assertEqual(report["totalReasoningCallsThisRun"], 1)
        self.assertEqual(report["acceptedCreatorCount"], 5)
        self.assertEqual(report["acceptedJudgmentCount"], 5)

    def test_stop_before_media_blocks_media_calls(self) -> None:
        state = _missing_greenpeace_state()
        state[PARSED_WINNER_RESPONSE_KEY] = _parsed_forgot_winner()
        candidate = _candidate(GREENPEACE_PROTOTYPE)
        candidate["advertisingClosure"]["durationSeconds"] = 10
        candidate_id = "cand-1-greenpeace_essential_pairing-1-recover"
        persist_rejected_creator_parsed_response(
            state,
            candidate_id=candidate_id,
            prototype_id=GREENPEACE_PROTOTYPE,
            round_index=1,
            attempt_number=1,
            parsed=candidate,
            failure_reason="builder2_creator_validation_failed:advertisingClosure.durationSeconds",
        )

        with patch("engine.builder2_complete_ad_reasoning_resume.video_job_get_raw", return_value={}), patch(
            "engine.builder2_complete_ad_reasoning_resume.acquire_job_lease", return_value=True
        ), patch("engine.builder2_complete_ad_reasoning_resume.release_job_lease"), patch(
            "engine.builder2_complete_ad_reasoning_resume.judge_candidate",
            side_effect=lambda **kwargs: _make_greenpeace_judgment(kwargs["candidate_id"], total=60),
        ), patch(
            "engine.builder2_complete_ad_reasoning_resume.save_tournament_state",
        ):
            report = run_controlled_complete_ad_reasoning_resume(
                job_id=state["jobId"],
                tournament_state=deepcopy(state),
                acquire_lease=False,
                stop_before_media=True,
            )

        self.assertTrue(report["ok"])
        self.assertEqual(report["creatorCallsThisRun"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)


class TestBuilder1IsolationDuration(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
