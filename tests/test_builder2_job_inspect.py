"""
Builder2 completed-job inspector tests — mocks only.
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from engine.builder2_job_inspect import (
    DEFAULT_JOB_INSPECT_ID,
    _SENSITIVE_OUTPUT_KEYS,
    inspect_builder2_completed_job,
)


HISTORICAL_JOB_ID = DEFAULT_JOB_INSPECT_ID
HISTORICAL_CANDIDATE_ID = "cand-1-summer_fan-1-57f415ca"
JOB_VIDEO_URL = "https://example.com/public/ace-video.mp4"
MEDIA_PUBLIC_URL = "https://example.com/public/media-final.mp4"


def _completed_job_record(*, video_url: str = JOB_VIDEO_URL) -> Dict[str, Any]:
    return {
        "status": "done",
        "videoUrl": video_url,
        "marketingText": "Hidden marketing copy.",
        "publicBaseUrl": "https://example.com",
        "error": "",
    }


def _completed_tournament_state(*, final_public_url: str = MEDIA_PUBLIC_URL) -> Dict[str, Any]:
    return {
        "status": "completed",
        "winnerDevelopmentCandidateId": HISTORICAL_CANDIDATE_ID,
        "winnerDevelopmentPrototypeId": "summer_fan",
        "mediaContinuationRequired": False,
        "mediaResume": {
            "mediaResumeStatus": "completed",
            "progressStage": "completed",
            "finalPublicUrl": final_public_url,
            "finalVideoPath": final_public_url,
            "runwayTaskId": "task-secret-123",
            "runwayVideoUrl": "https://runway.internal/output.mp4",
            "mediaCompletedAt": "2026-05-20T10:00:00+00:00",
        },
    }


class TestBuilder2JobInspectCompletedJob(unittest.TestCase):
    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_completed_job_with_job_video_url(
        self,
        job_get: Any,
        load_state: Any,
        raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        job_get.return_value = _completed_job_record()
        raw_reader.return_value = {
            "status": "done",
            "video_url": JOB_VIDEO_URL,
            "enqueued_ts": "1716192000",
            "last_progress_ts": "1716195600",
        }
        load_state.return_value = _completed_tournament_state()

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)

        self.assertTrue(report["ok"])
        self.assertTrue(report["jobExists"])
        self.assertTrue(report["jobCompleted"])
        self.assertEqual(report["resolvedVideoUrl"], JOB_VIDEO_URL)
        self.assertEqual(report["resolvedVideoUrlSource"], "job_video_url")
        self.assertTrue(report["downloadableCandidate"])
        self.assertTrue(report["frontendRecoveryLikely"])
        self.assertFalse(report["backendDeliveryIncomplete"])

    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_completed_job_with_only_media_final_public_url(
        self,
        job_get: Any,
        load_state: Any,
        raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        job_get.return_value = {"status": "done", "videoUrl": "", "marketingText": "", "publicBaseUrl": "https://example.com", "error": ""}
        raw_reader.return_value = {"status": "done", "video_url": ""}
        load_state.return_value = _completed_tournament_state()

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)

        self.assertEqual(report["resolvedVideoUrl"], MEDIA_PUBLIC_URL)
        self.assertEqual(report["resolvedVideoUrlSource"], "media_final_public_url")
        self.assertTrue(report["finalPublicUrlPresent"])
        self.assertTrue(report["frontendRecoveryLikely"])

    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_precedence_prefers_user_facing_job_url(
        self,
        job_get: Any,
        load_state: Any,
        raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        job_get.return_value = _completed_job_record(video_url=JOB_VIDEO_URL)
        raw_reader.return_value = {"status": "done", "video_url": JOB_VIDEO_URL}
        load_state.return_value = _completed_tournament_state(final_public_url=MEDIA_PUBLIC_URL)

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)

        self.assertEqual(report["resolvedVideoUrl"], JOB_VIDEO_URL)
        self.assertEqual(report["resolvedVideoUrlSource"], "job_video_url")

    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_completed_job_without_url_reports_delivery_incomplete(
        self,
        job_get: Any,
        load_state: Any,
        raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        job_get.return_value = {"status": "done", "videoUrl": "", "marketingText": "", "publicBaseUrl": "", "error": ""}
        raw_reader.return_value = {"status": "done", "video_url": ""}
        load_state.return_value = {
            "status": "completed",
            "winnerDevelopmentCandidateId": HISTORICAL_CANDIDATE_ID,
            "winnerDevelopmentPrototypeId": "summer_fan",
            "mediaResume": {"mediaResumeStatus": "completed", "finalPublicUrl": ""},
        }

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)

        self.assertTrue(report["jobCompleted"])
        self.assertIsNone(report["resolvedVideoUrl"])
        self.assertFalse(report["frontendRecoveryLikely"])
        self.assertTrue(report["backendDeliveryIncomplete"])
        self.assertIn("job.video_url", report["missingVideoUrlPaths"])

    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw", return_value=None)
    @patch("engine.builder2_job_inspect.load_tournament_state", return_value=None)
    @patch("engine.builder2_job_inspect.video_job_get", return_value=None)
    def test_missing_job_is_reported_safely(
        self,
        _job_get: Any,
        _load_state: Any,
        _raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        report = inspect_builder2_completed_job("missing-job-id")

        self.assertTrue(report["ok"])
        self.assertFalse(report["jobExists"])
        self.assertFalse(report["tournamentExists"])
        self.assertFalse(report["frontendRecoveryLikely"])

    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_tournament_winner_identity_is_reported(
        self,
        job_get: Any,
        load_state: Any,
        raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        job_get.return_value = _completed_job_record()
        raw_reader.return_value = {"status": "done", "video_url": JOB_VIDEO_URL}
        load_state.return_value = _completed_tournament_state()

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)

        self.assertEqual(report["winnerCandidateId"], HISTORICAL_CANDIDATE_ID)
        self.assertEqual(report["winnerPrototypeId"], "summer_fan")


class TestBuilder2JobInspectSafety(unittest.TestCase):
    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_sensitive_fields_are_not_printed(
        self,
        job_get: Any,
        load_state: Any,
        raw_reader: Any,
        _redis_cfg: Any,
    ) -> None:
        job_get.return_value = _completed_job_record()
        raw_reader.return_value = {
            "status": "done",
            "video_url": JOB_VIDEO_URL,
            "marketing_text": "Secret copy",
            "user_id": "user-secret",
            "session_id": "session-secret",
        }
        load_state.return_value = _completed_tournament_state()

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)
        output = json.dumps({k: v for k, v in report.items() if k not in _SENSITIVE_OUTPUT_KEYS})

        self.assertNotIn("Secret copy", output)
        self.assertNotIn("user-secret", output)
        self.assertNotIn("session-secret", output)
        self.assertNotIn("task-secret-123", output)
        self.assertNotIn("runway.internal", output)
        self.assertTrue(report["marketingTextPresent"])
        self.assertTrue(report["runwayTaskIdPresent"])
        self.assertTrue(report["runwayVideoUrlPresent"])
        self.assertTrue(report["ownershipFieldsPresent"])

    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.video_jobs_redis.get_redis")
    @patch("engine.builder2_job_inspect.load_tournament_state")
    @patch("engine.builder2_job_inspect.video_job_get")
    def test_inspector_performs_no_redis_writes(
        self,
        job_get: Any,
        load_state: Any,
        get_redis: Any,
        _redis_cfg: Any,
    ) -> None:
        redis_client = MagicMock()
        redis_client.hgetall.return_value = {"status": "done", "video_url": JOB_VIDEO_URL}
        get_redis.return_value = redis_client
        job_get.return_value = _completed_job_record()
        load_state.return_value = _completed_tournament_state()

        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)

        self.assertTrue(report["ok"])
        self.assertEqual(report["redisMutations"], 0)
        for method_name in ("hset", "set", "expire", "lpush", "pipeline"):
            method = getattr(redis_client, method_name)
            method.assert_not_called()
            if method_name == "pipeline":
                pipeline = method.return_value
                pipeline.execute.assert_not_called()


class TestBuilder2JobInspectGuarantees(unittest.TestCase):
    @patch("engine.builder2_job_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_job_inspect._read_job_hash_raw", return_value={"status": "done", "video_url": JOB_VIDEO_URL})
    @patch("engine.builder2_job_inspect.load_tournament_state", return_value=_completed_tournament_state())
    @patch("engine.builder2_job_inspect.video_job_get", return_value=_completed_job_record())
    def test_zero_external_calls(self, *_mocks: Any) -> None:
        report = inspect_builder2_completed_job(HISTORICAL_JOB_ID)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["runwayCalls"], 0)
        self.assertEqual(report["ffmpegCalls"], 0)

    @patch.dict(os.environ, {"BUILDER1_QUALITY_MODEL": "gpt-5.6-sol"}, clear=True)
    def test_builder1_unaffected(self) -> None:
        from engine.builder1_planning_profile import quality_model

        self.assertEqual(quality_model(), "gpt-5.6-sol")


class TestBuilder2JobInspectCli(unittest.TestCase):
    @patch("engine.builder2_job_inspect.inspect_builder2_completed_job")
    @patch.dict(os.environ, {"BUILDER2_JOB_INSPECT_ID": HISTORICAL_JOB_ID}, clear=False)
    def test_cli_module_entry(self, inspect_fn: Any) -> None:
        inspect_fn.return_value = {"ok": True, "jobId": HISTORICAL_JOB_ID}
        from engine.builder2_job_inspect import main

        with patch("builtins.print") as print_mock:
            code = main()
        self.assertEqual(code, 0)
        inspect_fn.assert_called_once_with(HISTORICAL_JOB_ID)
        printed = print_mock.call_args[0][0]
        payload = json.loads(printed)
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
