"""
Builder2 Advertising Closure tests — mocks only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_advertising_closure_contract import (
    build_closure_from_winner_plan,
    get_advertising_closure_status,
    headline_decision_allows_runway_scene_text,
    judge_advertising_completion_passes,
    normalize_advertising_closure,
    set_advertising_closure_status,
    validate_advertising_closure_delivery,
    validate_advertising_closure_methodology,
    validate_advertising_closure_object,
    validate_judge_advertising_completion_assessment,
    validate_silent_visual_understanding,
    validate_slogan_text,
    validate_strategic_understanding,
)
from engine.builder2_advertising_closure_proposal import generate_advertising_closure_proposal
from engine.builder2_closure_render import ClosureRenderResult
from engine.builder2_advertising_closure_resume import approve_persisted_proposal, run_one_advertising_closure_resume
from engine.builder2_advertising_closure_resume_guard import AdvertisingClosureResumeGuard
from engine.builder2_headline_decision_contract import get_normalized_headline_decision, headline_decision_is_omit
from engine.builder2_methodology_validation import collect_judge_methodology_structural_errors, validate_judge_methodology
from engine.builder2_tournament_contracts import Builder2TournamentError
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store
from engine.builder2_winner_persistence import persist_winner_development_atomically
from tests.builder2_methodology_fixtures import methodology_judgment_extras, methodology_winner_extras
from tests.test_builder2_media_resume import HISTORICAL_JOB_ID, _media_ready_state, _mock_start_image_data_uri
from tests.test_builder2_tournament import _candidate, _strategy, _winner_plan_from_prompt
from tests.test_builder2_winner_resume import _historical_judged_state


def _summer_fan_plan(*, headline_decision: str = "omit", with_closure: bool = False) -> Dict[str, Any]:
    strategy = _strategy(language="he")
    candidate = _candidate("summer_fan")
    plan = _winner_plan_from_prompt("")
    plan.update(methodology_winner_extras(headline_decision=headline_decision, winning_candidate=candidate, strategy=strategy))
    plan["productNameResolved"] = "ACE Product"
    plan["language"] = "he"
    plan["headlineDecision"] = {"decision": headline_decision, "reasonSource": "not_required"}
    if headline_decision == "omit":
        plan["headlineForm"] = "none"
        plan["headline"] = ""
        plan["headlineText"] = ""
    else:
        plan["headlineForm"] = "direct"
        plan["headline"] = "ACE Product closer"
        plan["headlineText"] = "ACE Product closer"
        plan["headlineTextRemainder"] = "closer"
    if with_closure:
        plan["advertisingClosure"] = {
            "required": True,
            "productNameText": "ACE Product",
            "sloganText": "קרוב יותר לבחירה הנכונה",
            "language": "he",
            "presentationMode": "end_card",
            "durationSeconds": 1.5,
            "headlineSource": "advertising_closure_role",
            "noLogo": True,
        }
    return plan


def _completed_state(
    *,
    with_closure: bool = False,
    approved: bool = False,
    rendered: bool = False,
    clear_creator_closure: bool = False,
) -> Dict[str, Any]:
    state = _media_ready_state(job_id=HISTORICAL_JOB_ID)
    plan = _summer_fan_plan(with_closure=with_closure)
    if clear_creator_closure:
        plan.pop("advertisingClosure", None)
        state.pop("advertisingClosure", None)
        state.pop("advertisingClosureStatus", None)
        state.pop("advertisingClosureSource", None)
    state["winnerDevelopmentPlan"] = plan
    media = state.setdefault("mediaResume", {})
    media["finalPublicUrl"] = "https://example.com/raw-runway.mp4"
    media["runwayVideoUrl"] = "https://example.com/raw-runway.mp4"
    media["downloadedVideoPath"] = "https://example.com/raw-runway.mp4"
    media["mediaResumeStatus"] = "completed"
    if with_closure:
        state["advertisingClosure"] = plan["advertisingClosure"]
        set_advertising_closure_status(state, "approved" if approved else "proposed")
    if rendered:
        media["finalVideoWithClosureUrl"] = "https://example.com/final-with-closure.mp4"
        media["finalPublicUrl"] = "https://example.com/final-with-closure.mp4"
        media["advertisingClosureRendered"] = True
        media["actualFinalVideoDurationSeconds"] = 13.51
        media["advertisingClosureStatus"] = "completed"
        state["advertisingClosureStatus"] = "completed"
    return state


class TestAdvertisingClosureContract(unittest.TestCase):
    def test_mandatory_for_final_delivery(self) -> None:
        plan = _summer_fan_plan()
        ok, missing = validate_advertising_closure_delivery(winner_plan=plan, tournament_state={})
        self.assertFalse(ok)
        self.assertIn("winnerDevelopmentPlan.advertisingClosure", missing)

    def test_omit_still_requires_closure(self) -> None:
        plan = _summer_fan_plan(headline_decision="omit")
        with self.assertRaises(Builder2TournamentError):
            validate_advertising_closure_methodology(plan, require_present=True)

    def test_use_carries_headline_into_closure(self) -> None:
        plan = _summer_fan_plan(headline_decision="use")
        closure = build_closure_from_winner_plan(plan)
        self.assertEqual(closure["sloganText"], "closer")

    def test_product_name_required_in_closure(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_advertising_closure_object({"required": True, "productNameText": "", "sloganText": "Valid slogan"})

    def test_slogan_required(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_slogan_text(slogan="", product_name="ACE Product")

    def test_slogan_word_limit_excludes_product_name(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_slogan_text(
                slogan="one two three four five six seven eight",
                product_name="ACE Product",
            )

    def test_generic_slogans_rejected(self) -> None:
        with self.assertRaises(Builder2TournamentError):
            validate_slogan_text(slogan="The best choice", product_name="ACE Product")

    def test_omit_suppresses_runway_scene_text_only(self) -> None:
        plan = _summer_fan_plan(headline_decision="omit")
        self.assertTrue(headline_decision_is_omit(get_normalized_headline_decision(plan)))
        self.assertFalse(headline_decision_allows_runway_scene_text(plan))

    def test_three_validations(self) -> None:
        plan = _summer_fan_plan(with_closure=True)
        judgment = methodology_judgment_extras()
        self.assertTrue(validate_silent_visual_understanding(winner_plan=plan, winning_judgment=judgment))
        self.assertTrue(validate_strategic_understanding(winner_plan=plan, winning_judgment=judgment))
        ok_missing_status, missing_status = validate_advertising_closure_delivery(winner_plan=plan, tournament_state={})
        self.assertFalse(ok_missing_status)
        self.assertIn("advertisingClosureStatus.approved_or_completed", missing_status)
        ok_completed, missing_completed = validate_advertising_closure_delivery(
            winner_plan=plan,
            tournament_state={"advertisingClosureStatus": "completed", "mediaResume": {}},
        )
        self.assertFalse(ok_completed)
        self.assertIn("mediaResume.finalVideoWithClosureUrl", missing_completed)


class TestJudgeAdvertisingCompletion(unittest.TestCase):
    def test_judge_pillar_required(self) -> None:
        judgment = methodology_judgment_extras()
        validate_judge_methodology(judgment, candidate=_candidate("summer_fan"))
        self.assertTrue(judge_advertising_completion_passes(judgment))

    def test_missing_closure_makes_future_candidate_ineligible(self) -> None:
        judgment = methodology_judgment_extras()
        judgment["eligible"] = True
        judgment["advertisingCompletionAssessment"]["functionsAsAdvertisement"] = False
        with self.assertRaises(Builder2TournamentError):
            validate_judge_advertising_completion_assessment(judgment)

    def test_structural_errors_include_assessment(self) -> None:
        judgment = methodology_judgment_extras()
        del judgment["advertisingCompletionAssessment"]
        errors = collect_judge_methodology_structural_errors(judgment, candidate=_candidate("summer_fan"))
        self.assertIn("builder2_judge_validation_failed:advertisingCompletionAssessment", errors)


class TestAdvertisingClosureResume(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_store()

    def tearDown(self) -> None:
        disable_memory_store()
        AdvertisingClosureResumeGuard.end()

    def test_proposal_only_one_call_maximum(self) -> None:
        state = _completed_state(clear_creator_closure=True)
        calls = {"count": 0}

        def _llm(**kwargs: Any) -> Dict[str, Any]:
            calls["count"] += 1
            return {
                "productNameText": "ACE Product",
                "sloganText": "קרוב יותר לבחירה",
                "language": "he",
                "presentationMode": "end_card",
                "durationSeconds": 1.5,
            }

        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            proposal_only=True,
            llm_client=_llm,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["advertisingClosureCalls"], 1)
        self.assertEqual(calls["count"], 1)

    def test_proposal_only_zero_runway_image_ffmpeg(self) -> None:
        state = _completed_state()
        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            proposal_only=True,
            llm_client=lambda **kwargs: {
                "productNameText": "ACE Product",
                "sloganText": "קרוב יותר לבחירה",
                "language": "he",
                "presentationMode": "end_card",
                "durationSeconds": 1.5,
            },
        )
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["closureFfmpegCalls"], 0)

    def test_persisted_proposal_is_reused(self) -> None:
        state = _completed_state(with_closure=True)
        calls = {"count": 0}
        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            proposal_only=True,
            llm_client=lambda **kwargs: calls.__setitem__("count", calls["count"] + 1) or {},
        )
        self.assertTrue(report["mediaReused"])
        self.assertEqual(report["advertisingClosureCalls"], 0)
        self.assertEqual(calls["count"], 0)

    def test_approval_makes_zero_model_calls(self) -> None:
        state = _completed_state(with_closure=True)
        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            approve=True,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["totalReasoningCalls"], 0)
        self.assertEqual(report["advertisingClosureStatus"], "approved")

    def test_render_only_zero_reasoning_and_runway(self) -> None:
        state = _completed_state(with_closure=True, approved=True)
        approve_persisted_proposal(state)
        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            render_only=True,
            render_endcard=lambda *args, **kwargs: ClosureRenderResult(
                public_url="https://example.com/final-with-closure.mp4",
                local_path="/tmp/final-with-closure.mp4",
                measured_duration_seconds=13.51,
                output_token="tok" * 8,
                input_fingerprint="abc",
            ),
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["totalReasoningCalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["closureFfmpegCalls"], 1)
        self.assertEqual(report["resolvedVideoUrl"], "https://example.com/final-with-closure.mp4")

    def test_completed_closure_artifact_reused(self) -> None:
        state = _completed_state(with_closure=True, approved=True, rendered=True)
        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            render_only=True,
            render_endcard=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ffmpeg must not run")),
        )
        self.assertTrue(report["mediaReused"])
        self.assertEqual(report["closureFfmpegCalls"], 0)

    def test_historical_winner_identity_unchanged(self) -> None:
        state = _completed_state()
        report = run_one_advertising_closure_resume(
            job_id=HISTORICAL_JOB_ID,
            tournament_state=deepcopy(state),
            proposal_only=True,
            llm_client=lambda **kwargs: {
                "productNameText": "ACE Product",
                "sloganText": "קרוב יותר לבחירה",
                "language": "he",
                "presentationMode": "end_card",
                "durationSeconds": 1.5,
            },
        )
        self.assertEqual(report["winnerPrototypeId"], "summer_fan")
        self.assertEqual(report["winnerCandidateId"], state["winnerDevelopmentCandidateId"])


class TestRunwayPromptPolicy(unittest.TestCase):
    def test_runway_prompt_remains_text_free(self) -> None:
        plan = _summer_fan_plan(headline_decision="omit", with_closure=True)
        prompt = str(plan.get("videoPrompt") or plan.get("videoPromptCore") or "")
        self.assertNotIn("title card", prompt.lower())
        self.assertNotIn("caption", prompt.lower())


class TestBuilder1Isolation(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
