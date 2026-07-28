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
    render_builder2_advertising_closure_endcard,
    verify_builder2_final_video_duration,
)
from engine.builder2_media_finalization_resume import (
    _apply_closure_duration_diagnostics,
    _execute_finalization_render_pipeline,
    _initial_report,
    _record_ffprobe_call,
)
from engine.builder2_new_format_config import resolve_builder2_effective_closure_segment_duration_seconds


class TestEffectiveClosureSegmentContract(unittest.TestCase):
    def test_effective_segment_is_configured_two_seconds(self) -> None:
        self.assertAlmostEqual(resolve_builder2_effective_closure_segment_duration_seconds(), 2.0, places=3)
        self.assertAlmostEqual(resolve_builder2_effective_closure_segment_duration_seconds(3.0), 2.0, places=3)

    def test_production_shape_calculates_twelve_not_thirteen(self) -> None:
        diag = verify_builder2_final_video_duration(
            12.042,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=3.0,
        )
        self.assertAlmostEqual(diag.effective_closure_segment_duration_seconds, 2.0, places=3)
        self.assertAlmostEqual(diag.calculated_expected_final_duration_seconds, 12.042, places=3)
        self.assertNotAlmostEqual(diag.calculated_expected_final_duration_seconds, 13.042, places=3)

    def test_frame_rounding_accepted(self) -> None:
        verify_builder2_final_video_duration(
            12.045,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
        )

    def test_thirteen_second_output_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                13.034,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=3.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_excessive_closure_gain")

    def test_three_second_gain_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                13.042,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_excessive_closure_gain")

    def test_two_second_gain_accepted(self) -> None:
        diag = verify_builder2_final_video_duration(
            12.033,
            visual_duration_seconds=10.042,
            end_card_duration_seconds=2.0,
        )
        self.assertAlmostEqual(diag.actual_closure_gain_seconds, 1.991, places=3)
        self.assertTrue(diag.closure_gain_accepted)

    def test_source_only_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                10.042,
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
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_excessive_closure_gain")

    def test_insufficient_gain_rejected(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                11.5,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=2.0,
            )
        self.assertEqual(ctx.exception.args[0], "builder2_media_final_duration_missing_end_card")

    def test_rejected_duration_preserved_in_diagnostics(self) -> None:
        with self.assertRaises(Builder2ClosureRenderError) as ctx:
            verify_builder2_final_video_duration(
                13.034,
                visual_duration_seconds=10.042,
                end_card_duration_seconds=3.0,
            )
        exc = ctx.exception
        assert exc.duration_diagnostics is not None
        self.assertAlmostEqual(exc.duration_diagnostics.measured_closure_output_duration_seconds, 13.034, places=3)
        self.assertAlmostEqual(
            exc.duration_diagnostics.calculated_expected_final_duration_seconds,
            12.042,
            places=3,
        )
        self.assertFalse(exc.duration_diagnostics.closure_gain_accepted)


class TestClosureFfmpegCommandConstruction(unittest.TestCase):
    def test_card_and_concat_use_two_second_authoritative_segment(self) -> None:
        from types import SimpleNamespace

        captured: list[tuple[list[str], str, str]] = []

        def runner(cmd: list[str], stage: str, category: str) -> None:
            captured.append((cmd, stage, category))

        tmp = Path(tempfile.mkdtemp())
        source = tmp / "source.mp4"
        source.write_bytes(b"fake")
        out = tmp / "out.mp4"
        real_is_file = Path.is_file
        real_stat = Path.stat

        def fake_is_file(self: Path) -> bool:
            if self.name == "out.mp4":
                return True
            return real_is_file(self)

        def fake_stat(self: Path):
            if self.name == "out.mp4":
                return SimpleNamespace(st_size=1234, st_mode=33188)
            return real_stat(self)

        with patch("engine.builder2_closure_render._ffprobe_duration_seconds", side_effect=[10.042, 12.042]), patch(
            "engine.builder2_closure_render._input_has_audio",
            return_value=False,
        ), patch(
            "engine.builder2_closure_render._default_font_path",
            return_value="font.ttf",
        ), patch(
            "engine.builder2_closure_render._ffmpeg_bin",
            return_value="ffmpeg",
        ), patch(
            "engine.builder2_closure_render._filter_path_for_ffmpeg",
            side_effect=lambda p: str(p),
        ), patch(
            "engine.builder2_closure_render._path_for_token",
            return_value=out,
        ), patch(
            "pathlib.Path.replace",
            return_value=None,
        ), patch(
            "pathlib.Path.is_file",
            fake_is_file,
        ), patch(
            "pathlib.Path.stat",
            fake_stat,
        ), patch(
            "engine.builder2_closure_render.verify_builder2_final_video_duration",
            side_effect=lambda measured, **kwargs: build_final_duration_verification_diagnostics(
                measured,
                visual_duration_seconds=10.042,
            ),
        ):
            render_builder2_advertising_closure_endcard(
                str(source),
                "https://ace.example.com",
                product_name="Product",
                slogan="Slogan",
                language="en",
                duration_seconds=3.0,
                publish=False,
                output_path=out,
                ffmpeg_runner=runner,
            )

        card_cmd = next(cmd for cmd, _stage, category in captured if category == "ffmpeg_card")
        concat_cmd = next(cmd for cmd, _stage, category in captured if category == "ffmpeg_concat")
        card_input = "".join(card_cmd)
        filter_complex = concat_cmd[concat_cmd.index("-filter_complex") + 1]
        self.assertIn("d=2.0", card_input)
        self.assertIn("-t", card_cmd)
        self.assertIn("2.000000", card_cmd)
        self.assertIn("trim=duration=2.000000", filter_complex)
        self.assertNotIn("d=3.0", card_input)
        self.assertNotIn("trim=duration=3", filter_complex)


class TestClosureDurationDiagnosticsSafety(unittest.TestCase):
    def test_diagnostics_exclude_paths_and_creative_text(self) -> None:
        diag = build_final_duration_verification_diagnostics(
            13.034,
            visual_duration_seconds=10.042,
            failure_code="builder2_media_final_duration_excessive_closure_gain",
        )
        payload = json.dumps(diag.to_report_dict())
        self.assertNotIn("/", payload)
        self.assertNotIn("\\", payload)
        self.assertNotIn("SECRET", payload)

    def test_failure_report_includes_gain_and_effective_segment(self) -> None:
        report = _initial_report(job_id="job-1", preflight=True)
        diagnostics = FinalDurationVerificationDiagnostics(
            measured_closure_output_duration_seconds=13.034,
            measured_closure_source_duration_seconds=10.042,
            configured_visual_duration_seconds=10.0,
            configured_end_card_duration_seconds=2.0,
            effective_closure_segment_duration_seconds=2.0,
            configured_final_duration_seconds=12.0,
            calculated_expected_final_duration_seconds=12.042,
            accepted_final_duration_lower_bound_seconds=11.692,
            accepted_final_duration_upper_bound_seconds=12.392,
            final_duration_delta_seconds=0.992,
            actual_closure_gain_seconds=2.992,
            closure_gain_accepted=False,
            final_duration_verification_failure_code="builder2_media_final_duration_excessive_closure_gain",
        )
        exc = Builder2ClosureRenderError(
            "builder2_media_final_duration_excessive_closure_gain",
            stage="duration_verification",
            duration_diagnostics=diagnostics,
            closure_ffmpeg_execution_accepted=True,
            closure_output_file_created=True,
            closure_output_file_size_bytes=12345,
            closure_ffprobe_calls=2,
        )
        _apply_closure_duration_diagnostics(report, exc)
        self.assertAlmostEqual(report["effectiveClosureSegmentDurationSeconds"], 2.0, places=3)
        self.assertAlmostEqual(report["calculatedExpectedFinalDurationSeconds"], 12.042, places=3)
        self.assertAlmostEqual(report["actualClosureGainSeconds"], 2.992, places=3)
        self.assertFalse(report["closureGainAccepted"])


class TestFfprobeCounterAccuracy(unittest.TestCase):
    def test_counters_aggregate_actual_subprocess_calls(self) -> None:
        report = _initial_report(job_id="job-1", preflight=True)
        _record_ffprobe_call(report, category="raw_runway")
        _record_ffprobe_call(report, category="headline")
        _record_ffprobe_call(report, category="final_closure")
        _record_ffprobe_call(report, category="final_closure")
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
            13.034,
            visual_duration_seconds=10.042,
            failure_code="builder2_media_final_duration_excessive_closure_gain",
        )
        mock_render.side_effect = Builder2ClosureRenderError(
            "builder2_media_final_duration_excessive_closure_gain",
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
                "durationSeconds": 3.0,
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
        self.assertAlmostEqual(report["calculatedExpectedFinalDurationSeconds"], 12.042, places=3)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["redisMutations"], 0)


if __name__ == "__main__":
    unittest.main()
