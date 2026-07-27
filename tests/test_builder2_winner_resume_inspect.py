"""
Builder2 Winner-development resume inspect — read-only safety and sanitization tests.
"""
from __future__ import annotations

import io
import json
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_winner_preservation_contract import PARSED_WINNER_RESPONSE_KEY
from engine.builder2_winner_resume_inspect import inspect_builder2_winner_resume, main
from tests.test_builder2_tournament import _winner_plan_from_prompt


def _winner_failure_state(*, headline: Any = None, headline_text: str = "SECRET HEADLINE") -> Dict[str, Any]:
    winner_id = "cand-1-forgot-1-2ab7377c"
    judgment_id = "judgment-forgot"
    plan = _winner_plan_from_prompt("")
    plan["headline"] = headline
    plan["headlineText"] = headline_text
    plan["headlineCoreKeyword"] = "forgot"
    plan["headlineForm"] = "statement"
    plan["headlineDecision"] = {
        "decision": "use",
        "reasonSource": "judge",
        "reason": "Headline is required for this concept.",
    }
    return {
        "jobId": "d6425b71-c612-4fcd-a3cf-8c30db88ca52",
        "status": "failed",
        "failureStage": "winner_development",
        "failureReason": "builder2_tournament_invalid_field:headline",
        "canResume": True,
        "reasoningComplete": False,
        "mediaStarted": False,
        "winnerCandidateId": winner_id,
        "winnerDevelopmentCandidateId": winner_id,
        "winnerDevelopmentPaidCallRecorded": True,
        "metrics": {
            "winnerDevelopmentCalls": 1,
            "winnerNormalCalls": 1,
            "winnerRepairCalls": 0,
            "winnerRetryCalls": 0,
            "totalReasoningCalls": 13,
        },
        PARSED_WINNER_RESPONSE_KEY: {
            "candidateId": winner_id,
            "parsed": plan,
        },
        "candidates": {
            winner_id: {
                "candidateId": winner_id,
                "prototypeId": "forgot",
                "judgmentId": judgment_id,
                "totalScore": 88,
            }
        },
        "judgments": {
            judgment_id: {
                "judgmentId": judgment_id,
                "candidateId": winner_id,
                "totalScore": 88,
                "judgment": {
                    "headlineNecessityAssessment": {
                        "headlineNeeded": True,
                        "visualWouldWorkWithoutHeadline": False,
                        "headlineRecommended": True,
                        "notes": "Long private judge notes that must never appear in inspect output. "
                        * 3,
                    }
                },
            }
        },
    }


class TestBuilder2WinnerResumeInspect(unittest.TestCase):
    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    @patch("engine.builder2_tournament_store.load_tournament_state")
    @patch("engine.builder2_tournament_store.save_tournament_state")
    @patch("engine.builder2_tournament_store._write_raw")
    @patch("engine.builder2_tournament_recovery.mark_job_queued")
    @patch("engine.builder2_execution_lease.acquire_job_lease", return_value=False)
    def test_read_raw_used_not_load_tournament_state(
        self,
        _lease: Any,
        _queue: Any,
        write_raw: Any,
        save_state: Any,
        load_state: Any,
        read_raw: Any,
        _redis: Any,
    ) -> None:
        read_raw.return_value = deepcopy(_winner_failure_state())
        report = inspect_builder2_winner_resume("d6425b71-c612-4fcd-a3cf-8c30db88ca52")
        self.assertTrue(report["ok"])
        read_raw.assert_called_once_with("d6425b71-c612-4fcd-a3cf-8c30db88ca52")
        load_state.assert_not_called()
        save_state.assert_not_called()
        write_raw.assert_not_called()
        _queue.assert_not_called()
        _lease.assert_not_called()
        self.assertEqual(report["redisMutations"], 0)

    def test_inspect_module_has_no_paid_or_media_imports(self) -> None:
        import inspect as inspect_module
        import engine.builder2_winner_resume_inspect as mod

        source = inspect_module.getsource(mod)
        for forbidden in (
            "openai",
            "runway",
            "ffmpeg",
            "builder2_image",
            "media_pipeline",
            "load_tournament_state",
            "save_tournament_state",
            "_write_raw",
        ):
            self.assertNotIn(forbidden, source, msg=f"unexpected reference to {forbidden}")

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_non_empty_headline_text_never_printed(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_winner_failure_state(headline="SECRET HEADLINE"))
        report = inspect_builder2_winner_resume("d6425b71-c612-4fcd-a3cf-8c30db88ca52")
        payload = json.dumps(report)
        self.assertNotIn("SECRET HEADLINE", payload)
        self.assertTrue(report["parsedWinnerHeadline"]["valuePresent"])
        self.assertGreater(report["parsedWinnerHeadline"]["characterCount"], 0)
        self.assertNotIn("value", report["parsedWinnerHeadline"])

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_null_and_empty_headline_distinguishable(self, read_raw: Any, _redis: Any) -> None:
        null_state = _winner_failure_state(headline=None, headline_text="")
        null_state[PARSED_WINNER_RESPONSE_KEY]["parsed"]["headlineText"] = None
        read_raw.return_value = deepcopy(null_state)
        null_report = inspect_builder2_winner_resume("job-null-headline")
        self.assertEqual(null_report["parsedWinnerHeadline"]["value"], None)
        self.assertEqual(null_report["parsedWinnerHeadlineText"]["value"], None)

        empty_state = _winner_failure_state(headline="", headline_text="")
        read_raw.return_value = deepcopy(empty_state)
        empty_report = inspect_builder2_winner_resume("job-empty-headline")
        self.assertEqual(empty_report["parsedWinnerHeadline"]["value"], "")
        self.assertEqual(empty_report["parsedWinnerHeadlineText"]["value"], "")

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_headline_decision_safe_metadata(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_winner_failure_state())
        report = inspect_builder2_winner_resume("job-headline-decision")
        decision = report["parsedWinnerHeadlineDecision"]
        self.assertEqual(decision["decision"], "use")
        self.assertEqual(decision["reasonSource"], "judge")
        self.assertTrue(decision["reasonPresent"])
        self.assertIn("reason", decision)
        self.assertNotIn("Long private", json.dumps(decision))

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_headline_necessity_assessment_sanitized(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_winner_failure_state())
        report = inspect_builder2_winner_resume("job-headline-necessity")
        assessment = report["winningJudgmentHeadlineNecessityAssessment"]
        self.assertTrue(assessment["headlineNeeded"])
        self.assertFalse(assessment["visualWouldWorkWithoutHeadline"])
        self.assertTrue(assessment["notesPresent"])
        self.assertTrue(assessment.get("notesRedacted"))
        self.assertNotIn("Long private judge notes", json.dumps(report))

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_winner_paid_call_and_metrics(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_winner_failure_state())
        report = inspect_builder2_winner_resume("job-metrics")
        paid = report["winnerDevelopmentPaidCallRecorded"]
        self.assertTrue(paid["keyExists"])
        self.assertEqual(paid["valueType"], "bool")
        self.assertTrue(paid["value"])
        self.assertEqual(report["winnerMetrics"]["winnerDevelopmentCalls"], 1)
        self.assertFalse(report["winnerDevelopmentPlanExists"])
        self.assertFalse(report["winnerDevelopmentAccepted"])

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_missing_tournament_safe_error(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = None
        report = inspect_builder2_winner_resume("missing-job")
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"], "builder2_winner_resume_inspect_tournament_not_found")
        self.assertEqual(report["redisMutations"], 0)

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    @patch("engine.builder2_tournament_store.load_tournament_state")
    def test_no_index_backfill_or_load(self, load_state: Any, read_raw: Any, _redis: Any) -> None:
        state = _winner_failure_state()
        state["acceptedJudgments"] = {}
        read_raw.return_value = deepcopy(state)
        report = inspect_builder2_winner_resume("job-no-backfill")
        self.assertTrue(report["ok"])
        load_state.assert_not_called()

    @patch("engine.builder2_winner_resume_inspect.redis_configured", return_value=True)
    @patch("engine.builder2_winner_resume_inspect._read_raw")
    def test_main_exit_codes(self, read_raw: Any, _redis: Any) -> None:
        read_raw.return_value = deepcopy(_winner_failure_state())
        with patch.dict("os.environ", {"BUILDER2_WINNER_RESUME_INSPECT_JOB_ID": "job-main-ok"}):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main()
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])

        read_raw.return_value = None
        with patch.dict("os.environ", {"BUILDER2_WINNER_RESUME_INSPECT_JOB_ID": "job-main-missing"}):
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                code = main()
            self.assertEqual(code, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])

    def test_builder1_files_unchanged(self) -> None:
        import glob
        import os

        root = os.path.dirname(os.path.dirname(__file__))
        builder1_paths = glob.glob(os.path.join(root, "engine", "builder1*.py"))
        self.assertTrue(builder1_paths, "expected builder1 modules to exist")
        for path in builder1_paths:
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("builder2_winner_resume_inspect", source)
            self.assertNotIn("BUILDER2_WINNER_RESUME_INSPECT_JOB_ID", source)


if __name__ == "__main__":
    unittest.main()
