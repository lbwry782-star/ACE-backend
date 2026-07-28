"""
Builder2 media-finalization failure inspector tests — mocks only.
"""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_media_finalization_failure_inspect import (
    _sanitize_report_for_output,
    inspect_builder2_media_finalization_failure,
    main,
)
from engine.builder2_tournament_contracts import WINNER_PLAN_SCHEMA_VERSION
from engine.builder2_winner_preservation_contract import SERVER_OWNED_WINNER_SOURCE_KEY


JOB_ID = "d6425b71-c612-4fcd-a3cf-8c30db88ca52"
RAW_RUNWAY = "https://runway.example.com/raw-task.mp4"
HEADLINE_URL = "https://ace.example.com/api/video-headline/abc123deadbeefcafe0123456789abcd"
CLOSURE_URL = "https://ace.example.com/api/builder2-final-video/closuretoken0123456789abcdef"
BROKEN_HEADLINE_FINAL_URL = (
    "https://ace-backend-k1p6.onrender.com/api/video-headline/42228511edd94fa18eccedf4d39db8e0"
)


def verified_final_publication_media_fields(**overrides: Any) -> Dict[str, Any]:
    return {
        "finalPublicationVerificationAccepted": True,
        "finalPublicationDurableStorageConfirmed": True,
        "finalPublicationBackendKind": "persistent_disk",
        "finalPublicationReferencePresent": True,
        "finalPublicationUploadedByteCount": 1028987,
        "headlineReconstructionCompleted": True,
        "finalDurationAccepted": True,
        **overrides,
    }


def _false_completion_state(*, with_valid_closure: bool = False) -> Dict[str, Any]:
    candidate_id = "cand-1-forgot-1-2ab7377c"
    final_url = CLOSURE_URL if with_valid_closure else HEADLINE_URL
    plan = {
        "schemaVersion": WINNER_PLAN_SCHEMA_VERSION,
        "prototypeId": "forgot",
        "structureType": "continuous_event",
        "headlineDecision": {"decision": "use", "reasonSource": "judge", "reason": "Required."},
        "headlineText": "SECRET HEADLINE TEXT",
        "headlineTextRemainder": "remember the moment",
        "productNameResolved": "Forgot Product",
        "advertisingClosure": {
            "required": True,
            "productNameText": "Forgot Product",
            "sloganText": "SECRET SLOGAN TEXT",
            "language": "he",
            "presentationMode": "end_card",
            "durationSeconds": 2.0,
            "noLogo": True,
        },
        SERVER_OWNED_WINNER_SOURCE_KEY: {
            "sourceCandidateId": candidate_id,
            "sourcePrototypeId": "forgot",
        },
    }
    return {
        "jobId": JOB_ID,
        "tournamentId": "9bbcd5d6-cca4-4f76-b01f-d434f95f5580",
        "status": "completed",
        "reasoningComplete": True,
        "mediaStarted": True,
        "mediaContinuationRequired": False,
        "winnerDevelopmentAcceptedAt": "2026-05-19T00:00:00+00:00",
        "winnerDevelopmentCandidateId": candidate_id,
        "winnerDevelopmentPrototypeId": "forgot",
        "winnerDevelopmentPlan": plan,
        "advertisingClosure": dict(plan["advertisingClosure"]),
        "advertisingClosureStatus": "completed",
        "candidates": {
            candidate_id: {
                "candidateId": candidate_id,
                "prototypeId": "forgot",
                "totalScore": 88,
            }
        },
        "mediaResume": {
            "mediaResumeStatus": "completed",
            "progressStage": "completed",
            "startImageStatus": "completed",
            "startImageArtifact": "data:image/png;base64,REDACTED_SHOULD_NOT_PRINT",
            "runwayTaskId": "task-secret-123",
            "runwaySubmissionStatus": "submitted",
            "runwayStatus": "SUCCEEDED",
            "runwayVideoUrl": RAW_RUNWAY,
            "rawRunwayVideoUrl": RAW_RUNWAY,
            "rawRunwayVideoPath": RAW_RUNWAY,
            "downloadedVideoPath": RAW_RUNWAY,
            "rawRunwayDurationSeconds": 10.042,
            "finalPublicUrl": final_url,
            "finalVideoPath": final_url,
            "finalVideoWithClosureUrl": final_url,
            "finalVideoDurationSeconds": 12.0 if with_valid_closure else 10.042,
            "endCardDurationSeconds": 2.0,
            "advertisingClosureStatus": "completed",
            "advertisingClosureRendered": with_valid_closure,
            "actualFinalVideoDurationSeconds": 12.01 if with_valid_closure else None,
            "closureRenderedAt": "2026-05-20T00:00:00+00:00",
            "ffmpegStatus": "completed",
            "headlineArtifactUrl": HEADLINE_URL if with_valid_closure else None,
            **(verified_final_publication_media_fields() if with_valid_closure else {}),
            "callCounters": {
                "startImageCalls": 1,
                "startImageGeneratedCount": 1,
                "runwaySubmissionCalls": 1,
                "runwayTaskCreatedCount": 1,
                "runwayPollingCalls": 1,
            },
        },
    }


def _job_raw(*, video_url: str) -> Dict[str, str]:
    return {
        "status": "done",
        "video_url": video_url,
        "marketing_text": "SECRET MARKETING",
    }


class TestBuilder2MediaFinalizationFailureInspect(unittest.TestCase):
    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    @patch("engine.builder2_tournament_store.load_tournament_state")
    @patch("engine.builder2_tournament_store.save_tournament_state")
    @patch("engine.builder2_tournament_store._write_raw")
    def test_read_raw_used_without_load_or_writes(
        self,
        write_raw: Any,
        save_state: Any,
        load_state: Any,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = deepcopy(_false_completion_state())
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertTrue(report["ok"])
        read_raw.assert_called_once_with(JOB_ID)
        load_state.assert_not_called()
        save_state.assert_not_called()
        write_raw.assert_not_called()
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["inspectionCallCounts"]["redisMutations"], 0)

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_false_completion_without_valid_closure(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertTrue(report["falseCompletionDetected"])
        self.assertEqual(report["persistedCompletionStatus"], "completed")
        self.assertEqual(report["effectiveCompletionStatus"], "incomplete")
        self.assertFalse(report["completionContractAudit"]["completionContractSatisfied"])
        self.assertIn(
            "job_video_url_points_to_headline_artifact_not_closure_inclusive_final",
            report["falseCompletionReasons"],
        )

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_headline_only_final_url_not_accepted_as_closure(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        audit = report["completionContractAudit"]
        self.assertFalse(audit["closureInclusiveArtifactPresent"])
        self.assertFalse(audit["jobVideoUrlPointsToClosureArtifact"])
        graph = report["artifactIdentityGraph"]
        self.assertTrue(graph["jobMarkedDoneViaHeadlineArtifact"])

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_valid_closure_inclusive_final_accepted(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        state = _false_completion_state(with_valid_closure=True)
        state["mediaResume"]["finalVideoWithClosureUrl"] = CLOSURE_URL
        state["mediaResume"]["finalPublicUrl"] = CLOSURE_URL
        state["mediaResume"]["finalVideoPath"] = CLOSURE_URL
        state["mediaResume"]["headlineArtifactUrl"] = HEADLINE_URL
        state["mediaResume"]["headlineArtifactUrl"] = HEADLINE_URL
        state["mediaResume"]["advertisingClosureRendered"] = True
        state["mediaResume"]["actualFinalVideoDurationSeconds"] = 12.01
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=CLOSURE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertFalse(report["falseCompletionDetected"])
        self.assertTrue(report["completionContractAudit"]["completionContractSatisfied"])

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_missing_closure_status_detected(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        state = _false_completion_state(with_valid_closure=False)
        state["mediaResume"].pop("advertisingClosureStatus", None)
        state["mediaResume"].pop("closureRenderedAt", None)
        state["advertisingClosureStatus"] = "rendering"
        read_raw.return_value = deepcopy(state)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertFalse(report["completionContractAudit"]["closureRendered"])
        self.assertFalse(report["completionContractAudit"]["completionContractSatisfied"])

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_raw_and_headline_artifacts_distinguished(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        graph = report["artifactIdentityGraph"]
        relation = graph["pairwiseRelations"].get("rawRunwayArtifact__headlineOverlayArtifact") or graph[
            "pairwiseRelations"
        ].get("headlineOverlayArtifact__rawRunwayArtifact")
        self.assertEqual(relation, "different")
        job_raw_relation = graph["pairwiseRelations"].get("rawRunwayArtifact__jobVideoUrl") or graph[
            "pairwiseRelations"
        ].get("jobVideoUrl__rawRunwayArtifact")
        self.assertEqual(job_raw_relation, "different")

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_urls_and_tokens_redacted_from_output(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        safe = _sanitize_report_for_output(report)
        rendered = json.dumps(safe, ensure_ascii=False)
        self.assertNotIn("abc123deadbeefcafe0123456789abcd", rendered)
        self.assertNotIn("data:image/png", rendered)
        self.assertNotIn("task-secret-123", rendered)
        self.assertNotIn("SECRET HEADLINE TEXT", rendered)
        self.assertNotIn("SECRET SLOGAN TEXT", rendered)
        self.assertNotIn("SECRET MARKETING", rendered)

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_reuse_reported_for_existing_artifacts(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertTrue(report["startImageCanBeReused"])
        self.assertTrue(report["runwaySubmissionCanBeReused"])
        self.assertTrue(report["runwayPollingCanBeSkipped"])
        self.assertTrue(report["rawRunwayArtifactCanBeReused"])
        self.assertTrue(report["headlineArtifactCanBeReused"])
        self.assertTrue(report["advertisingClosureCanBeReused"])

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_recovery_reported_zero_openai_and_runway_submission(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertFalse(report["recoveryRequiresOpenAI"])
        self.assertFalse(report["recoveryRequiresRunwaySubmission"])
        self.assertFalse(report["recoveryRequiresRunwayPolling"])
        self.assertTrue(report["recoveryRequiresFFmpeg"])
        self.assertTrue(report["recoveryRequiresPublication"])

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.video_jobs_redis.get_redis")
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_zero_redis_writes(
        self,
        read_raw: Any,
        job_get_raw: Any,
        get_redis: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        redis_client = MagicMock()
        get_redis.return_value = redis_client
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        self.assertTrue(report["ok"])
        self.assertEqual(report["redisMutations"], 0)
        for method_name in ("hset", "set", "expire", "lpush", "pipeline"):
            getattr(redis_client, method_name).assert_not_called()

    @patch("engine.builder2_media_finalization_failure_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_failure_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_failure_inspect._read_raw")
    def test_inspection_call_counts_zero_paid_operations(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_failure(JOB_ID)
        counts = report["inspectionCallCounts"]
        self.assertEqual(counts["openAICalls"], 0)
        self.assertEqual(counts["imageCalls"], 0)
        self.assertEqual(counts["runwaySubmissionCalls"], 0)
        self.assertEqual(counts["runwayPollingCalls"], 0)
        self.assertEqual(counts["ffmpegCalls"], 0)
        self.assertEqual(counts["publicationCalls"], 0)

    def test_inspect_module_has_no_paid_or_media_execution(self) -> None:
        import inspect as inspect_module

        import engine.builder2_media_finalization_failure_inspect as mod

        source = inspect_module.getsource(mod)
        for forbidden in (
            "import openai",
            "from openai",
            "import subprocess",
            "subprocess.run(",
            "generate_builder2_start_image(",
            "submit_builder2_runway_task(",
            "postprocess_video_headline(",
            "append_advertising_closure_endcard(",
            "load_tournament_state(",
            "save_tournament_state(",
            "video_job_mark_done(",
        ):
            self.assertNotIn(forbidden, source)

    @patch.dict("os.environ", {"BUILDER2_MEDIA_FINALIZATION_FAILURE_INSPECT_JOB_ID": JOB_ID}, clear=False)
    @patch("engine.builder2_media_finalization_failure_inspect.inspect_builder2_media_finalization_failure")
    def test_main_uses_env_job_id(self, inspect_fn: Any) -> None:
        inspect_fn.return_value = {"ok": True, "inspectionCompleted": True}
        code = main()
        self.assertEqual(code, 0)
        inspect_fn.assert_called_once_with(JOB_ID)

    @patch.dict("os.environ", {}, clear=True)
    def test_main_missing_job_id_exits_one(self) -> None:
        self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
