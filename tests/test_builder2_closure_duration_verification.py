"""Builder2 closure final-duration verification and preflight diagnostics."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    FinalDurationVerificationDiagnostics,
    build_final_duration_verification_diagnostics,
    verify_builder2_final_video_duration,
)
from engine.builder2_media_finalization_resume import (
    _apply_closure_duration_diagnostics,
    _execute_finalization_render_pipeline,
    _initial_report,
    _record_ffprobe_call,
)


class TestClosureDurationFormula(unittest.TestCase):
    def test_measured_visual_plus_end_card_expected(self) -> None:
        diag = verify_builder2_final_video_duration(
            12.042,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
        )
        self.assertAlmostEqual(diag.calculated_expected_final_duration_seconds, 12.042, places=3)
        self.assertAlmostEqual(diag.measured_closure_output_duration_seconds, 12.042, places=3)

    def test_frame_rounding_accepted(self) -> None:
        verify_builder2_final_video_duration(
            12.067,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
        )

    def test_source_only_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                10.042,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_not_longer_than_visual")

    def test_end_card_only_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                2.0,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_not_longer_than_visual")

    def test_duplicated_visual_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                22.084,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_duplicated_visual")

    def test_insufficient_gain_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                11.5,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_missing_end_card")

    def test_approximately_two_second_gain_accepted(self) -> None:
        verify_builder2_final_video_duration(
            12.033,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
        )

    def test_configured_twelve_metadata_does_not_override_measured_visual(self) -> None:
        diag = verify_builder2_final_video_duration(
            12.042,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
            expected_final_seconds=12.0,
        )
        self.assertAlmostEqual(diag.calculated_expected_final_duration_seconds, 12.042, places=3)

    def test_rejected_duration_preserved_in_diagnostics(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                11.0,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        exc = ctx.exception
        self.assertIsNotNone(exc.duration_diagnostics)
        assert exc.duration_diagnostics is not None
        self.assertAlmostEqual(exc.duration_diagnostics.measured_closure_output_duration_seconds, 11.0, places=3)
        self.assertAlmostEqual(
            exc.duration_diagnostics.calculated_expected_final_duration_seconds,
            12.042,
            places=3,
        )
        self.assertTrue(exc.duration_diagnostics.final_duration_verification_failure_code)


class TestClosureDurationDiagnosticsSafety(unittest.TestCase):
    def test_diagnostics_exclude_paths_and_creative_text(self) -> None:
        diag = build_final_duration_verification_diagnostics(
            11.5,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
            failure_code="builder2_media_final_duration_missing_end_card",
        )
        payload = json.dumps(diag.to_report_dict())
        self.assertNotIn("/", payload)
        self.assertNotIn("\\", payload)
        self.assertNotIn("SECRET", payload)

    def test_failure_report_includes_bounds_and_delta(self) -> None:
        report = _initial_report(job_id="job-1", preflight=True)
        diagnostics = FinalDurationVerificationDiagnostics(
            measured_closure_output_duration_seconds=11.0,
            measured_closure_source_duration_seconds=10.042,
            configured_visual_duration_seconds=10.0,
            configured_end_card_duration_seconds=2.0,
            configured_final_duration_seconds=12.0,
            calculated_expected_final_duration_seconds=12.042,
            accepted_final_duration_lower_bound_seconds=11.692,
            accepted_final_duration_upper_bound_seconds=12.392,
            final_duration_delta_seconds=-1.042,
            final_duration_verification_failure_code="builder2_media_final_duration_missing_end_card",
        )
        exc = Builder2ClosureRenderError(
            "builder2_media_final_duration_missing_end_card",
            stage="duration_verification",
            duration_diagnostics=diagnostics,
            closure_ffmpeg_execution_accepted=True,
            closure_output_file_created=True,
            closure_output_file_size_bytes=12345,
            closure_ffprobe_calls=2,
        )
        _apply_closure_duration_diagnostics(report, exc)
        self.assertTrue(report["closureFfmpegExecutionAccepted"])
        self.assertEqual(report["measuredFinalDurationSeconds"], 11.0)
        self.assertAlmostEqual(report["calculatedExpectedFinalDurationSeconds"], 12.042, places=3)
        self.assertIsNotNone(report["acceptedFinalDurationLowerBoundSeconds"])
        self.assertIsNotNone(report["finalDurationDeltaSeconds"])


class TestFfprobeCounterAccuracy(unittest.TestCase):
    def test_counters_aggregate_actual_subprocess_calls(self) -> None:
        report = _initial_report(job_id="job-1", preflight=True)
        _record_ffprobe_call(report, category="raw_runway")
        _record_ffprobe_call(report, category="headline")
        _record_ffprobe_call(report, category="final_closure")
        _record_ffprobe_call(report, category="final_closure")
        self.assertEqual(report["rawRunwayFfprobeCalls"], 1)
        self.assertEqual(report["headlineFfprobeCalls"], 1)
        self.assertEqual(report["finalClosureFfprobeCalls"], 2)
        self.assertEqual(report["totalFfprobeSubprocessCalls"], 4)
        self.assertEqual(report["ffprobeCalls"], 4)


class TestPreflightDurationFailureReporting(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    def test_preflight_preserves_rejected_duration_and_counters(
        self,
        mock_source: Any,
        mock_render: Any,
    ) -> None:
        from engine.builder2_media_finalization_resume import _execute_finalization_render_pipeline

        tmp = Path(tempfile.mkdtemp())
        source = tmp / "headline.mp4"
        source.write_bytes(b"fake")
        mock_source.return_value = type(
            "Decision",
            (),
            {
                "failure_reason": "",
                "failure_stage": "",
                "closure_input_path": source,
                "local_headline_render_required": False,
                "source_kind": "legacy_headline_artifact",
                "legacy_headline_diagnostics": None,
                "raw_runway_diagnostics": None,
                "to_report_dict": lambda self: {"selectedFinalizationSourceKind": "legacy_headline_artifact"},
            },
        )()

        diagnostics = build_final_duration_verification_diagnostics(
            11.0,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
            failure_code="builder2_media_final_duration_missing_end_card",
        )
        mock_render.side_effect = Builder2ClosureRenderError(
            "builder2_media_final_duration_missing_end_card",
            stage="duration_verification",
            duration_diagnostics=diagnostics,
            closure_ffmpeg_execution_accepted=True,
            closure_output_file_created=True,
            closure_output_file_size_bytes=999,
            closure_ffprobe_calls=2,
        )

        state: Dict[str, Any] = {
            "reasoningComplete": True,
            "winnerDevelopmentPlan": {"headlineDecision": {"decision": "include"}},
            "advertisingClosure": {
                "required": True,
                "productNameText": "P",
                "sloganText": "S",
                "durationSeconds": 2.0,
            },
            "mediaResume": {},
        }
        report = _initial_report(job_id="job-1", preflight=True)

        with patch(
            "engine.builder2_media_finalization_resume._probe_duration",
            return_value=10.042,
        ):
            _execute_finalization_render_pipeline(
                job_id="job-1",
                state=state,
                plan=state["winnerDevelopmentPlan"],
                job_video_url="https://example.com/h",
                report=report,
                preflight=True,
                public_base_url="https://ace.example.com",
            )

        self.assertEqual(report["failureStage"], "duration_verification")
        self.assertEqual(report["measuredFinalDurationSeconds"], 11.0)
        self.assertAlmostEqual(report["calculatedExpectedFinalDurationSeconds"], 12.042, places=3)
        self.assertTrue(report["closureFfmpegExecutionAccepted"])
        self.assertEqual(report["headlineFfprobeCalls"], 1)
        self.assertEqual(report["finalClosureFfprobeCalls"], 2)
        self.assertEqual(report["ffprobeCalls"], 3)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["redisMutations"], 0)


if __name__ == "__main__":
    unittest.main()
