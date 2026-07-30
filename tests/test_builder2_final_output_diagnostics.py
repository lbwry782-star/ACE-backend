"""
Builder2 final-output diagnostics tests — mocks only.
"""
from __future__ import annotations

import json
import unittest
from copy import deepcopy
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_final_output_diagnostics import (
    build_builder2_media_diagnostic_fields,
    classify_builder2_media_diagnostic_phase,
    collect_media_resume_contract_missing_fields,
    inspect_builder2_final_output,
    resolve_safe_durable_final_public_output,
)
from engine.builder2_final_output_inspect import main as final_output_inspect_main
from engine.builder2_media_resume_contract_inspect import inspect_media_resume_contract
from engine.builder2_tournament_store import disable_memory_store, enable_memory_store, save_tournament_state
from engine.builder2_winner_offline_salvage import attempt_offline_winner_development_salvage
from engine.builder2_winner_persistence import compute_winner_development_plan_fingerprint
from tests.test_builder2_media_finalization_failure_inspect import CLOSURE_URL, _false_completion_state
from tests.test_builder2_winner_offline_salvage import (
    _current_job_shaped_state,
    _winning_card_winner_id,
)


def _completed_state(*, media_continuation_required: bool = False) -> Dict[str, Any]:
    state = deepcopy(_false_completion_state(with_valid_closure=True))
    state["jobId"] = "job-final-output-completed"
    state["mediaContinuationRequired"] = media_continuation_required
    state["winnerDevelopmentAccepted"] = True
    state["mediaContinuationRequired"] = media_continuation_required
    plan = state["winnerDevelopmentPlan"]
    state["winnerDevelopmentPlanFingerprint"] = compute_winner_development_plan_fingerprint(plan)
    return state


class TestBuilder2FinalOutputDiagnostics(unittest.TestCase):
    def test_completed_job_classified_without_resume_missing_fields(self) -> None:
        state = _completed_state(media_continuation_required=False)
        diagnostic = build_builder2_media_diagnostic_fields(state)
        self.assertTrue(diagnostic["mediaCompleted"])
        self.assertFalse(diagnostic["mediaResumeNeeded"])
        self.assertEqual(diagnostic["mediaDiagnosticPhase"], "completed")
        self.assertEqual(diagnostic["mediaResumeMissingFields"], [])
        self.assertFalse(diagnostic["mediaResumeReady"])
        self.assertEqual(diagnostic["mediaResumeBlockedReason"], "media_already_completed")
        self.assertTrue(diagnostic["finalOutputAvailable"])

    def test_completed_media_contract_inspector_does_not_flag_continuation(self) -> None:
        state = _completed_state(media_continuation_required=False)
        report = inspect_media_resume_contract(state)
        self.assertTrue(report["mediaCompleted"])
        self.assertFalse(report["mediaResumeNeeded"])
        self.assertEqual(report["mediaResumeMissingFields"], [])
        self.assertFalse(report["mediaContinuationRequired"])
        self.assertFalse(report["mediaResumeReady"])
        self.assertEqual(report["mediaResumeBlockedReason"], "media_already_completed")

    def test_incomplete_ready_job_retains_readiness_validation(self) -> None:
        enable_memory_store()
        try:
            state = _current_job_shaped_state()
            winner_id = _winning_card_winner_id(state)
            winner_rec = state["candidates"][winner_id]
            save_tournament_state(state["jobId"], state)
            attempt_offline_winner_development_salvage(
                state,
                winner_candidate_id=winner_id,
                prototype_id="winning_card",
                strategy_foundation=state["strategyFoundation"],
                winning_candidate=winner_rec["creatorOutput"],
                winning_judgment=state["judgments"][winner_rec["judgmentId"]]["judgment"],
                job_id=state["jobId"],
            )
            save_tournament_state(state["jobId"], state)
            report = inspect_media_resume_contract(state)
            self.assertFalse(report["mediaCompleted"])
            self.assertTrue(report["mediaResumeNeeded"])
            self.assertEqual(report["mediaDiagnosticPhase"], "ready_to_resume")
            self.assertTrue(report["mediaResumeReady"])
            self.assertEqual(report["mediaResumeMissingFields"], [])
        finally:
            disable_memory_store()

    def test_incomplete_not_ready_reports_missing_fields(self) -> None:
        state = _current_job_shaped_state()
        missing = collect_media_resume_contract_missing_fields(state)
        self.assertIn("mediaContinuationRequired", missing)
        self.assertEqual(classify_builder2_media_diagnostic_phase(state), "incomplete_not_ready")

    @patch("engine.builder2_final_output_inspect.load_tournament_state")
    def test_final_output_inspector_returns_safe_public_route(self, load_state) -> None:
        from io import StringIO

        state = _completed_state()
        load_state.return_value = state
        buffer = StringIO()
        with patch.dict("os.environ", {"BUILDER2_FINAL_OUTPUT_INSPECT_JOB_ID": state["jobId"]}, clear=False), patch(
            "sys.stdout", buffer
        ):
            exit_code = final_output_inspect_main()
        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["finalOutputAvailable"])
        self.assertTrue(payload["mediaCompleted"])
        self.assertFalse(payload["mediaResumeNeeded"])
        self.assertIn("/api/builder2-final-video/", payload["durableFinalPublicPath"])
        self.assertTrue(payload["finalVideoToken"])
        self.assertNotIn("runway", json.dumps(payload).lower())
        self.assertNotIn("X-Amz", json.dumps(payload))

    def test_signed_storage_url_is_not_exposed(self) -> None:
        state = _completed_state()
        signed = (
            "https://bucket.s3.amazonaws.com/private/final.mp4"
            "?X-Amz-Signature=secretvalue&X-Amz-Credential=AKIAFAKEKEY"
        )
        state["mediaResume"]["finalPublicUrl"] = signed
        state["mediaResume"]["finalVideoWithClosureUrl"] = CLOSURE_URL
        safe = resolve_safe_durable_final_public_output(state)
        self.assertIn("/api/builder2-final-video/", safe["durableFinalPublicPath"] or "")
        self.assertNotIn("AKIA", json.dumps(safe))
        self.assertNotIn("secretvalue", json.dumps(safe))

    @patch("engine.builder2_media_resume.execute_builder2_media_pipeline")
    def test_completed_inspectors_are_read_only_and_do_not_invoke_pipeline(self, pipeline) -> None:
        state = _completed_state()
        inspect_media_resume_contract(state)
        inspect_builder2_final_output(state)
        pipeline.assert_not_called()

    def test_builder1_unchanged(self) -> None:
        import glob
        import os

        root = os.path.dirname(os.path.dirname(__file__))
        for path in glob.glob(os.path.join(root, "engine", "builder1*.py")):
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("builder2_final_output_diagnostics", source)
            self.assertNotIn("builder2_final_output_inspect", source)


if __name__ == "__main__":
    unittest.main()
