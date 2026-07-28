"""
Builder2 finalization recovery eligibility — dual recovery basis tests.
"""
from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_closure_render import Builder2ClosureRenderError, ClosureRenderResult
from engine.builder2_media_finalization_contract import (
    assess_recoverable_failed_finalization_state,
    evaluate_finalization_recovery_eligibility,
    finalization_recovery_eligible,
)
from engine.builder2_media_finalization_resume import (
    _persist_finalization_failure_state,
    run_finalization_preflight,
    run_one_media_finalization_resume,
)
from engine.builder2_media_finalization_state_inspect import inspect_builder2_media_finalization_state
from tests.test_builder2_media_finalization_failure_inspect import (
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    RAW_RUNWAY,
    _false_completion_state,
    _job_raw,
)
from tests.test_builder2_media_finalization import _valid_closure_result


def _production_stranded_state() -> Dict[str, Any]:
    state = deepcopy(_false_completion_state(with_valid_closure=False))
    state["status"] = "media_finalization_incomplete"
    state["mediaContinuationRequired"] = True
    state["advertisingClosureStatus"] = "failed"
    media = state["mediaResume"]
    media["mediaResumeStatus"] = "finalization_failed"
    media["advertisingClosureStatus"] = "failed"
    media["advertisingClosureRendered"] = False
    media["actualFinalVideoDurationSeconds"] = None
    media.pop("headlineArtifactUrl", None)
    return state


class TestLegacyFalseCompletionEligibility(unittest.TestCase):
    def test_legacy_false_completion_remains_eligible(self) -> None:
        state = _false_completion_state(with_valid_closure=False)
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(evaluation["eligible"])
        self.assertTrue(evaluation["legacyFalseCompletionConfirmed"])
        self.assertFalse(evaluation["recoverableFailedFinalizationConfirmed"])
        self.assertEqual(evaluation["recoveryEligibilityBasis"], "legacy_false_completion")


class TestRecoverableFailedFinalizationEligibility(unittest.TestCase):
    def test_failed_finalization_state_becomes_eligible(self) -> None:
        state = _production_stranded_state()
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(evaluation["eligible"])
        self.assertFalse(evaluation["legacyFalseCompletionConfirmed"])
        self.assertTrue(evaluation["recoverableFailedFinalizationConfirmed"])
        self.assertEqual(evaluation["recoveryEligibilityBasis"], "failed_finalization_state")

    def test_production_shaped_state_regression(self) -> None:
        state = _production_stranded_state()
        assessment = assess_recoverable_failed_finalization_state(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(assessment.recoverable)
        self.assertEqual(assessment.recovery_basis, "failed_finalization_state")
        eligible, missing = finalization_recovery_eligible(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(eligible)
        self.assertEqual(missing, [])

    def test_does_not_require_status_completed(self) -> None:
        state = _production_stranded_state()
        self.assertEqual(state["status"], "media_finalization_incomplete")
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(evaluation["eligible"])
        self.assertFalse(evaluation["falseCompletionConfirmed"])

    def test_failed_recovery_remains_eligible_for_retry(self) -> None:
        state = _production_stranded_state()
        report: Dict[str, Any] = {
            "failureStage": "concatenation",
            "failureReason": "builder2_closure_ffmpeg_failed",
            "originalFailureStage": "concatenation",
            "originalFailureCode": "builder2_closure_ffmpeg_failed",
            "originalFailureClass": "Builder2ClosureRenderError",
        }
        _persist_finalization_failure_state(state, report=report, headline_failure=False)
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertTrue(evaluation["eligible"])
        self.assertEqual(state["mediaResume"]["finalizationFailureStage"], "concatenation")
        self.assertEqual(state["mediaResume"]["finalizationFailureCode"], "builder2_closure_ffmpeg_failed")
        self.assertEqual(state["mediaResume"]["finalizationFailureClass"], "Builder2ClosureRenderError")

    def test_second_failed_retry_remains_eligible(self) -> None:
        state = _production_stranded_state()
        for stage in ("concatenation", "duration_probe"):
            report = {
                "failureStage": stage,
                "failureReason": "builder2_closure_ffmpeg_failed",
                "originalFailureStage": stage,
                "originalFailureCode": "builder2_closure_ffmpeg_failed",
                "originalFailureClass": "Builder2ClosureRenderError",
            }
            _persist_finalization_failure_state(state, report=report, headline_failure=False)
            evaluation = evaluate_finalization_recovery_eligibility(
                state=state,
                plan=state["winnerDevelopmentPlan"],
                job_video_url=HEADLINE_URL,
            )
            self.assertTrue(evaluation["eligible"], msg=f"retry after {stage} should remain eligible")


class TestRecoveryEligibilityBlocking(unittest.TestCase):
    def test_valid_closure_inclusive_final_blocks_recovery(self) -> None:
        state = _false_completion_state(with_valid_closure=True)
        state["mediaResume"]["advertisingClosureRendered"] = True
        state["mediaResume"]["actualFinalVideoDurationSeconds"] = 12.01
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=CLOSURE_URL,
        )
        self.assertFalse(evaluation["eligible"])
        self.assertTrue(evaluation["recoveryBlockedByValidFinal"])
        self.assertIn("validClosureAlreadyPresent", evaluation["missing"])

    def test_completed_publication_blocks_recovery(self) -> None:
        state = _false_completion_state(with_valid_closure=True)
        state["mediaResume"]["advertisingClosureRendered"] = True
        state["mediaResume"]["advertisingClosureStatus"] = "completed"
        state["mediaResume"]["actualFinalVideoDurationSeconds"] = 12.01
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=CLOSURE_URL,
        )
        self.assertFalse(evaluation["eligible"])

    def test_missing_raw_and_headline_blocks_recovery(self) -> None:
        state = _production_stranded_state()
        media = state["mediaResume"]
        for key in (
            "rawRunwayVideoUrl",
            "rawRunwayVideoPath",
            "runwayVideoUrl",
            "downloadedVideoPath",
            "headlineArtifactUrl",
            "finalPublicUrl",
            "finalVideoWithClosureUrl",
            "finalVideoPath",
        ):
            media.pop(key, None)
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url="",
        )
        self.assertFalse(evaluation["eligible"])
        self.assertTrue(evaluation["recoveryBlockedByMissingIntermediate"])

    def test_invalid_winner_blocks_recovery(self) -> None:
        state = _production_stranded_state()
        state.pop("winnerDevelopmentAcceptedAt", None)
        state.pop("winnerDevelopmentCandidateId", None)
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertFalse(evaluation["eligible"])
        self.assertIn("winnerDevelopmentAccepted", evaluation["missing"])

    def test_missing_closure_data_blocks_recovery(self) -> None:
        state = _production_stranded_state()
        state.pop("advertisingClosure", None)
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertFalse(evaluation["eligible"])
        self.assertIn("advertisingClosure", evaluation["missing"])

    def test_conflicting_publication_evidence_blocks_recovery(self) -> None:
        state = _production_stranded_state()
        media = state["mediaResume"]
        media["advertisingClosureRendered"] = True
        media["advertisingClosureStatus"] = "completed"
        media["actualFinalVideoDurationSeconds"] = None
        evaluation = evaluate_finalization_recovery_eligibility(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
        )
        self.assertFalse(evaluation["eligible"])
        blocking = evaluation["recoverableFailedFinalizationBlockingConditions"]
        self.assertIn("conflictingPublicationEvidence", blocking)

    def test_active_lease_blocks_failed_finalization_basis(self) -> None:
        state = _production_stranded_state()
        assessment = assess_recoverable_failed_finalization_state(
            state=state,
            plan=state["winnerDevelopmentPlan"],
            job_video_url=HEADLINE_URL,
            active_finalization_lease=True,
        )
        self.assertFalse(assessment.recoverable)
        self.assertIn("activeFinalizationLeasePresent", assessment.blocking_conditions)


class TestPreflightFromFailedFinalization(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    def test_preflight_from_failed_state_no_redis_or_publication(
        self,
        pipeline: Any,
        build_config: Any,
    ) -> None:
        def _ok(**kwargs: Any) -> None:
            kwargs["report"]["ok"] = True
            kwargs["report"]["readyForFinalizationRecovery"] = True

        pipeline.side_effect = _ok
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        state = _production_stranded_state()
        report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["eligibleForFinalizationRecovery"])
        self.assertEqual(report["recoveryEligibilityBasis"], "failed_finalization_state")
        self.assertFalse(report["legacyFalseCompletionConfirmed"])
        self.assertTrue(report["recoverableFailedFinalizationConfirmed"])
        self.assertTrue(report["preflight"])
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["publicationCalls"], 0)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["runwayPollingCalls"], 0)


class TestSuccessfulRecoveryPersistence(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.video_job_mark_done")
    @patch("engine.builder2_media_finalization_resume.save_tournament_state")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease")
    def test_successful_recovery_from_failed_state_marks_completed(
        self,
        _release: Any,
        _lease: Any,
        _redis: Any,
        read_raw: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        pipeline: Any,
        save_state: Any,
        mark_done: Any,
    ) -> None:
        state = _production_stranded_state()
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))

        def _render(**kwargs: Any) -> ClosureRenderResult:
            st = kwargs["state"]
            media = st.setdefault("mediaResume", {})
            media.update(
                {
                    "headlineReconstructionCompleted": True,
                    "headlineArtifactSource": "deterministic_local_reconstruction_from_raw_runway",
                    "finalVideoWithClosureUrl": CLOSURE_URL,
                    "finalPublicUrl": CLOSURE_URL,
                    "advertisingClosureRendered": True,
                    "actualFinalVideoDurationSeconds": 12.01,
                    "advertisingClosureStatus": "completed",
                }
            )
            return _valid_closure_result()

        pipeline.side_effect = _render
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertTrue(report["ok"])
        saved = save_state.call_args[0][1]
        self.assertEqual(saved["status"], "completed")
        self.assertFalse(saved["mediaContinuationRequired"])
        self.assertEqual(saved["mediaResume"]["mediaResumeStatus"], "completed")
        self.assertEqual(saved["mediaResume"]["advertisingClosureStatus"], "completed")
        self.assertTrue(saved["mediaResume"]["advertisingClosureRendered"])
        mark_done.assert_called_once()

    @patch("engine.builder2_media_finalization_resume.save_tournament_state")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease")
    @patch("engine.builder2_media_finalization_resume.video_job_mark_done")
    def test_failed_publication_does_not_mark_completion(
        self,
        mark_done: Any,
        _release: Any,
        _lease: Any,
        _redis: Any,
        read_raw: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        pipeline: Any,
        save_state: Any,
    ) -> None:
        state = _production_stranded_state()
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))

        def _render(**kwargs: Any) -> ClosureRenderResult:
            st = kwargs["state"]
            media = st.setdefault("mediaResume", {})
            media.update(
                {
                    "finalVideoWithClosureUrl": HEADLINE_URL,
                    "finalPublicUrl": HEADLINE_URL,
                    "advertisingClosureRendered": False,
                    "actualFinalVideoDurationSeconds": None,
                    "advertisingClosureStatus": "failed",
                }
            )
            return _valid_closure_result(public_url=HEADLINE_URL, measured_duration_seconds=10.042)

        pipeline.side_effect = _render
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertFalse(report["ok"])
        mark_done.assert_not_called()
        if save_state.called:
            saved = save_state.call_args[0][1]
            self.assertNotEqual(saved.get("status"), "completed")


class TestStateInspectorEligibilityReporting(unittest.TestCase):
    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_inspector_reports_failed_finalization_basis(
        self,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = _production_stranded_state()
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertTrue(report["currentEligibility"])
        self.assertEqual(report["currentEligibilityReason"], "recoverable_failed_finalization_state")
        self.assertFalse(report["falseCompletionConfirmed"])
        self.assertTrue(report["recoverableFailedFinalizationConfirmed"])
        self.assertEqual(report["recoveryEligibilityBasis"], "failed_finalization_state")
        self.assertEqual(report["recommendedNextAction"], "run_finalization_preflight")
        self.assertEqual(report["publicationEvidenceClassification"], "proven_not_published")


if __name__ == "__main__":
    unittest.main()
