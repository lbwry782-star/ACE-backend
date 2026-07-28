"""
Builder2 media finalization state inspector tests — mocks only.
"""
from __future__ import annotations

import json
import os
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_media_finalization_state_inspect import (
    inspect_builder2_media_finalization_state,
    main,
)
from tests.test_builder2_media_finalization_failure_inspect import (
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    RAW_RUNWAY,
    _false_completion_state,
    _job_raw,
)


def _post_failed_recovery_state() -> Dict[str, Any]:
    state = deepcopy(_false_completion_state(with_valid_closure=False))
    state["status"] = "media_finalization_incomplete"
    state["mediaContinuationRequired"] = True
    state["advertisingClosureStatus"] = "failed"
    state["mediaResume"]["mediaResumeStatus"] = "finalization_failed"
    state["mediaResume"]["advertisingClosureStatus"] = "failed"
    state["mediaResume"]["advertisingClosureRendered"] = False
    state["mediaResume"]["actualFinalVideoDurationSeconds"] = None
    state["mediaResume"].pop("headlineArtifactUrl", None)
    return state


class TestBuilder2MediaFinalizationStateInspect(unittest.TestCase):
    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    @patch("engine.builder2_tournament_store.save_tournament_state")
    @patch("engine.builder2_tournament_store.load_tournament_state")
    def test_no_redis_mutations_or_lease(
        self,
        load_state: Any,
        save_state: Any,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        save_state.assert_not_called()
        load_state.assert_not_called()
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["leaseOperations"], 0)

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_legacy_false_completion_eligible(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertTrue(report["currentEligibility"])
        self.assertTrue(report["falseCompletionConfirmed"])
        self.assertTrue(report["eligibilityConditionResults"]["falseCompletionProven"])

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_failed_recovery_state_is_eligible(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = _post_failed_recovery_state()
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertTrue(report["currentEligibility"])
        self.assertFalse(report["falseCompletionConfirmed"])
        self.assertTrue(report["recoverableFailedFinalizationConfirmed"])
        self.assertEqual(report["currentEligibilityReason"], "recoverable_failed_finalization_state")
        self.assertEqual(report["recoveryEligibilityBasis"], "failed_finalization_state")
        self.assertFalse(report["falseCompletionConditionResults"]["persistedStatusCompleted"])
        self.assertTrue(report["stateChangedFromKnownLegacyPattern"])
        self.assertIn("persistedTournamentStatus", report["changedConditionNames"])

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_distinguishes_headline_from_closure_routes(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertEqual(report["finalVideoWithClosureRouteFamily"], "api/video-headline")
        self.assertEqual(report["rawRunwayRouteFamily"], "runway-artifact")
        self.assertTrue(report["finalUrlEqualsLegacyHeadlineArtifact"])
        self.assertFalse(report["finalUrlEqualsRawRunway"])

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_publication_not_persisted_after_failed_recovery(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = _post_failed_recovery_state()
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertEqual(report["publicationEvidenceClassification"], "proven_not_published")
        self.assertFalse(report["publicationCompletedPersisted"])
        self.assertFalse(report["advertisingClosureRendered"])
        self.assertTrue(report["rawRunwayRecoverable"])

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_recommends_preflight_from_failed_state(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = _post_failed_recovery_state()
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertEqual(report["recommendedNextAction"], "run_finalization_preflight")

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_no_full_urls_or_creative_text(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        blob = json.dumps(report)
        self.assertNotIn(HEADLINE_URL, blob)
        self.assertNotIn(RAW_RUNWAY, blob)
        self.assertNotIn("SECRET", blob)
        self.assertNotIn("Forgot Product", blob)
        self.assertNotIn("finalPublicUrl", report)
        self.assertNotIn("finalVideoWithClosureUrl", report)
        self.assertNotIn("jobVideoUrl", report)

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_include_final_url_flag_adds_only_requested_fields(
        self,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
    ) -> None:
        state = deepcopy(_false_completion_state(with_valid_closure=True))
        read_raw.return_value = state
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID, include_final_url=True)
        self.assertEqual(report["finalPublicUrl"], CLOSURE_URL)
        self.assertEqual(report["finalVideoWithClosureUrl"], CLOSURE_URL)
        self.assertEqual(report["jobVideoUrl"], HEADLINE_URL)
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["leaseOperations"], 0)
        self.assertEqual(report["publicationCalls"], 0)
        blob = json.dumps(report)
        self.assertNotIn(RAW_RUNWAY, blob)
        self.assertNotIn("SECRET", blob)
        self.assertNotIn("Forgot Product", blob)

    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_INCLUDE_FINAL_URL": "true",
        },
        clear=False,
    )
    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_include_final_url_env_flag(
        self,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertEqual(report["jobVideoUrl"], HEADLINE_URL)
        self.assertEqual(report["finalVideoWithClosureUrl"], HEADLINE_URL)
        self.assertEqual(report["finalPublicUrl"], HEADLINE_URL)

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_include_final_url_omits_raw_runway_job_video_url(
        self,
        read_raw: Any,
        job_get_raw: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = deepcopy(_false_completion_state(with_valid_closure=False))
        job_get_raw.return_value = _job_raw(video_url=RAW_RUNWAY)
        report = inspect_builder2_media_finalization_state(JOB_ID, include_final_url=True)
        self.assertIsNone(report["jobVideoUrl"])
        self.assertNotIn(RAW_RUNWAY, json.dumps(report))

    @patch("engine.builder2_media_finalization_state_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_state_inspect.video_job_get_raw")
    @patch("engine.builder2_media_finalization_state_inspect._read_raw")
    def test_precise_eligibility_after_failed_recovery(self, read_raw: Any, job_get_raw: Any, _redis: Any) -> None:
        read_raw.return_value = _post_failed_recovery_state()
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        report = inspect_builder2_media_finalization_state(JOB_ID)
        self.assertEqual(report["likelyMutationSourceFunction"], "run_one_media_finalization_resume.save_tournament_state_on_render_failure")
        self.assertEqual(report["minimalFutureStateRepairFields"], [])

    @patch.dict("os.environ", {"BUILDER2_MEDIA_FINALIZATION_STATE_INSPECT_JOB_ID": JOB_ID}, clear=False)
    @patch("engine.builder2_media_finalization_state_inspect.inspect_builder2_media_finalization_state")
    def test_cli_emits_json_and_exits_zero(self, inspect_fn: Any) -> None:
        inspect_fn.return_value = {"ok": True, "inspectionCompleted": True, "currentEligibility": False}
        with patch("builtins.print") as print_mock:
            code = main()
        self.assertEqual(code, 0)
        payload = json.loads(print_mock.call_args[0][0])
        self.assertTrue(payload["inspectionCompleted"])


class TestBuilder2ReportingObservabilitySemantics(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_cli_booleans_truthful_in_emitted_json(self, run_one: Any) -> None:
        import io

        run_one.return_value = {"jobId": JOB_ID, "ok": False, "preflight": True, "failureStage": "eligibility"}
        buffer = io.StringIO()

        def _write(data: str) -> int:
            buffer.write(data)
            return len(data)

        with patch("sys.stdout.write", side_effect=_write):
            with patch("sys.stdout.flush"):
                with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                    from engine.builder2_media_finalization_resume import main as resume_main

                    resume_main()
        payload = json.loads(buffer.getvalue().strip())
        self.assertTrue(payload["cliReportConstructionAccepted"])
        self.assertTrue(payload["cliJsonSerializationAccepted"])
        self.assertTrue(payload["cliStdoutWriteAttempted"])
        self.assertNotIn("cliStdoutWriteAccepted", payload)
        self.assertNotIn("cliDoneLogAttempted", payload)
        joined = "\n".join(captured.output)
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_REPORT_EMITTED", joined)
        self.assertIn("stdoutWriteAccepted=True", joined)


if __name__ == "__main__":
    unittest.main()
