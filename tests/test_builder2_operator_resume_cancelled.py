"""
Builder2 operator-only cancelled-job resume tests — offline/mocked only.
"""
from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder2_job_cancellation import request_builder2_job_cancellation
from engine.builder2_job_ownership import ownership_fields_for_job_create
from engine.builder2_operator_resume_cancelled import (
    CLASSIFICATION_FINAL_OUTPUT_AVAILABLE,
    CLASSIFICATION_UNSAFE,
    classify_cancelled_builder2_job,
    reactivate_cancelled_builder2_job,
    run_operator_resume_cancelled_job,
)
from engine.builder2_resume_service import request_builder2_resume
from engine.builder2_tournament_recovery import (
    disable_memory_recovery,
    enable_memory_recovery,
    register_recoverable_job,
    requeue_recoverable_job,
    set_memory_job_hash,
)
from engine.builder2_tournament_store import (
    disable_memory_store,
    enable_memory_store,
    load_tournament_state,
    new_tournament_state,
    save_tournament_state,
)
from engine.video_jobs_redis import disable_memory_jobs, enable_memory_jobs, video_job_get_raw
from tests.test_builder2_media_resume import _media_ready_state
from tests.test_builder2_tournament import _strategy


def _builder2_job_hash(job_id: str, **extra: str) -> Dict[str, str]:
    fields = ownership_fields_for_job_create(MagicMock(headers={}), {"productDescription": "desc"})
    fields.update(
        {
            "status": "running",
            "product_name": "Product",
            "product_description": "A product description for testing.",
            "public_base_url": "https://example.test",
            "canResume": "1",
            "builder": "builder2",
        }
    )
    fields.update({k: str(v) for k, v in extra.items()})
    set_memory_job_hash(job_id, fields)
    return fields


class _OperatorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        enable_memory_jobs()
        enable_memory_store()
        enable_memory_recovery()
        os.environ["BUILDER2_TOURNAMENT_ENABLED"] = "true"

    def tearDown(self) -> None:
        disable_memory_jobs()
        disable_memory_store()
        disable_memory_recovery()
        os.environ.pop("BUILDER2_TOURNAMENT_ENABLED", None)


class TestNormalCancelledStillBlocked(_OperatorTestBase):
    def test_media_resume_still_rejected(self) -> None:
        job_id = "op-normal-media-block"
        state = _media_ready_state(job_id=job_id)
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        from engine.builder2_media_resume import run_one_media_resume

        report = run_one_media_resume(job_id=job_id, tournament_state=deepcopy(state))
        self.assertEqual(report.get("failureReason"), "builder2_job_cancelled")

    def test_reasoning_resume_still_rejected(self) -> None:
        job_id = "op-normal-reasoning-block"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="s")
        state["strategyFoundation"] = _strategy()
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        from engine.builder2_complete_ad_reasoning_resume import run_controlled_complete_ad_reasoning_resume

        report = run_controlled_complete_ad_reasoning_resume(job_id=job_id, tournament_state=state, acquire_lease=False)
        self.assertEqual(report.get("failureReason"), "builder2_job_cancelled")

    def test_recovery_still_skips_cancelled(self) -> None:
        job_id = "op-recovery-skip"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="s")
        state["strategyFoundation"] = _strategy()
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        register_recoverable_job(job_id)
        self.assertFalse(requeue_recoverable_job(job_id))

    def test_durable_resume_still_rejected(self) -> None:
        job_id = "op-durable-block"
        state = _media_ready_state(job_id=job_id)
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1", canResume="0")
        result = request_builder2_resume(job_id)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("error"), "builder2_job_cancelled")


class TestOperatorClassification(_OperatorTestBase):
    def test_refuses_paid_call_outcome_unknown(self) -> None:
        job_id = "op-lyria-unknown"
        state = _media_ready_state(job_id=job_id)
        media = state.setdefault("mediaResume", {})
        media["musicGenerationStatus"] = "paid_call_outcome_unknown"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        request_builder2_job_cancellation(job_id)

        classification = classify_cancelled_builder2_job(job_id)
        self.assertEqual(classification.get("classification"), CLASSIFICATION_UNSAFE)
        self.assertFalse(classification.get("safe"))
        self.assertIn("lyria", (classification.get("paidCallOutcomeUnknown") or "").lower())

    @patch("engine.builder2_complete_ad_reasoning_resume.run_controlled_complete_ad_reasoning_resume")
    def test_operator_reasoning_incomplete_reactivates_and_delegates(self, reasoning_mock: MagicMock) -> None:
        job_id = "op-reasoning-resume"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="s")
        state["strategyFoundation"] = _strategy()
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)

        reasoning_mock.return_value = {"ok": True, "strategyReused": True}
        report = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(report.get("reactivated"))
        self.assertEqual(report.get("resumeType"), "reasoning")
        reasoning_mock.assert_called_once()
        raw = video_job_get_raw(job_id) or {}
        self.assertEqual(raw.get("cancelRequested"), "0")
        self.assertNotEqual(raw.get("status"), "cancelled")
        self.assertEqual(raw.get("operatorResumedFromCancelled"), "1")
        self.assertEqual(raw.get("previousCancelReason"), "frontend_refresh")

    @patch("engine.builder2_media_resume.run_one_media_resume")
    def test_operator_media_incomplete_reactivates_and_delegates(self, media_mock: MagicMock) -> None:
        job_id = "op-media-resume"
        state = _media_ready_state(job_id=job_id)
        state["mediaContinuationRequired"] = True
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)

        media_mock.return_value = {"ok": True, "mediaReused": True, "runwaySubmissionCalls": 0}
        report = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(report.get("reactivated"))
        self.assertEqual(report.get("resumeType"), "media")
        media_mock.assert_called_once()

    @patch("engine.builder2_media_resume.run_one_media_resume")
    def test_runway_artifact_reused_no_resubmit(self, media_mock: MagicMock) -> None:
        job_id = "op-runway-reuse"
        state = _media_ready_state(job_id=job_id)
        state["mediaContinuationRequired"] = True
        media = state.setdefault("mediaResume", {})
        media["runwayTaskId"] = "task-existing"
        media["runwayVideoUrl"] = "https://runway.test/v.mp4"
        media["runwayStatus"] = "SUCCEEDED"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)

        media_mock.return_value = {"ok": True, "runwaySubmissionCalls": 0, "runwayTaskCreatedCount": 0}
        report = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(report.get("ok"))
        classification = report.get("classification") or {}
        self.assertEqual(classification.get("resumeType"), "media")
        media_mock.assert_called_once()

    @patch("engine.builder2_media_resume.run_one_media_resume")
    def test_lyria_succeeded_reused(self, media_mock: MagicMock) -> None:
        job_id = "op-lyria-reuse"
        state = _media_ready_state(job_id=job_id)
        state["mediaContinuationRequired"] = True
        media = state.setdefault("mediaResume", {})
        media["runwayTaskId"] = "task-1"
        media["runwayVideoUrl"] = "https://runway.test/v.mp4"
        media["runwayStatus"] = "SUCCEEDED"
        media["musicGenerationStatus"] = "succeeded"
        media["musicArtifactUrl"] = "https://example.test/music/a.mp3"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)

        media_mock.return_value = {"ok": True, "lyriaCalls": 0, "lyriaReused": True}
        report = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(report.get("ok"))
        resume_report = report.get("resumeReport") or {}
        self.assertEqual(resume_report.get("lyriaCalls"), 0)

    def test_already_completed_no_delegate_paid_calls(self) -> None:
        job_id = "op-already-done"
        state = _media_ready_state(job_id=job_id)
        media = state.setdefault("mediaResume", {})
        media["finalPublicUrl"] = "https://example.test/final.mp4"
        media["mediaResumeStatus"] = "completed"
        save_tournament_state(job_id, state)
        _builder2_job_hash(
            job_id,
            status="cancelled",
            cancelRequested="1",
            video_url="https://example.test/final.mp4",
        )
        with patch("engine.builder2_media_resume.run_one_media_resume") as media_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.run_controlled_complete_ad_reasoning_resume"
        ) as reasoning_mock:
            report = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(report.get("ok"))
        self.assertEqual((report.get("classification") or {}).get("classification"), CLASSIFICATION_FINAL_OUTPUT_AVAILABLE)
        media_mock.assert_not_called()
        reasoning_mock.assert_not_called()
        resume_report = report.get("resumeReport") or {}
        self.assertTrue(resume_report.get("paidCallsSkipped"))

    @patch("engine.builder2_media_resume.run_one_media_resume")
    def test_second_operator_invocation_idempotent(self, media_mock: MagicMock) -> None:
        job_id = "op-idempotent"
        state = _media_ready_state(job_id=job_id)
        state["mediaContinuationRequired"] = True
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)
        media_mock.return_value = {"ok": True}

        first = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(first.get("reactivated"))
        raw_after = dict(video_job_get_raw(job_id) or {})

        second = run_operator_resume_cancelled_job(job_id)
        self.assertTrue(second.get("ok"))
        raw_second = video_job_get_raw(job_id) or {}
        self.assertEqual(raw_after.get("operatorResumedAt"), raw_second.get("operatorResumedAt"))
        self.assertFalse(second.get("reactivated"))
        self.assertEqual(media_mock.call_count, 2)

    def test_paid_unknown_operator_refuses_zero_delegate(self) -> None:
        job_id = "op-refuse-unknown"
        state = _media_ready_state(job_id=job_id)
        media = state.setdefault("mediaResume", {})
        media["musicGenerationStatus"] = "paid_call_outcome_unknown"
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="cancelled", cancelRequested="1")
        with patch("engine.builder2_media_resume.run_one_media_resume") as media_mock, patch(
            "engine.builder2_complete_ad_reasoning_resume.run_controlled_complete_ad_reasoning_resume"
        ) as reasoning_mock:
            report = run_operator_resume_cancelled_job(job_id)
        self.assertFalse(report.get("ok"))
        self.assertFalse(report.get("reactivated"))
        media_mock.assert_not_called()
        reasoning_mock.assert_not_called()


class TestBuilder1Unchanged(_OperatorTestBase):
    def test_not_builder2_job_rejected(self) -> None:
        job_id = "builder1-op"
        set_memory_job_hash(
            job_id,
            {
                "status": "cancelled",
                "cancelRequested": "1",
                "product_name": "P",
                "product_description": "D",
                "public_base_url": "https://example.test",
            },
        )
        classification = classify_cancelled_builder2_job(job_id)
        self.assertEqual(classification.get("failureReason"), "not_builder2_job")


class TestReactivationFields(_OperatorTestBase):
    def test_reactivation_preserves_audit_fields(self) -> None:
        job_id = "op-audit"
        state = new_tournament_state(job_id=job_id, language="he", active_prototype_ids=["think_small"], random_seed="s")
        state["strategyFoundation"] = _strategy()
        save_tournament_state(job_id, state)
        _builder2_job_hash(job_id, status="running")
        request_builder2_job_cancellation(job_id)

        classification = classify_cancelled_builder2_job(job_id)
        self.assertTrue(classification.get("safe"))
        result = reactivate_cancelled_builder2_job(job_id, classification=classification)
        self.assertTrue(result.get("reactivated"))

        raw = video_job_get_raw(job_id) or {}
        self.assertEqual(raw.get("previousCancelReason"), "frontend_refresh")
        self.assertEqual(raw.get("cancelRequested"), "0")
        self.assertTrue(raw.get("operatorResumedAt"))
        tournament = load_tournament_state(job_id)
        self.assertTrue((tournament or {}).get("operatorResumedFromCancelled"))


if __name__ == "__main__":
    unittest.main()
