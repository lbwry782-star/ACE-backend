"""
Builder2 media finalization resume CLI contract tests.
"""
from __future__ import annotations

import ast
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_closure_render import (
    ClosureRenderResult,
    build_final_duration_verification_diagnostics,
    render_builder2_advertising_closure_endcard,
)
from engine.builder2_media_finalization_resume import (
    emit_media_finalization_resume_report,
    main,
    run_finalization_preflight,
)
from engine.builder2_new_format_config import resolve_builder2_effective_closure_segment_duration_seconds
from tests.test_builder2_media_finalization_failure_inspect import (
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    _false_completion_state,
    _job_raw,
)


def _success_closure_result(**overrides: Any) -> ClosureRenderResult:
    measured = overrides.get("measured_duration_seconds", 12.042)
    diagnostics = build_final_duration_verification_diagnostics(
        measured,
        visual_duration_seconds=10.042,
        effective_closure_segment_duration_seconds=2.0,
    )
    return ClosureRenderResult(
        public_url=overrides.get("public_url", CLOSURE_URL),
        local_path=overrides.get("local_path", "/tmp/out.mp4"),
        measured_duration_seconds=measured,
        output_token=overrides.get("output_token", "tok" * 8),
        input_fingerprint=overrides.get("input_fingerprint", "abc"),
        duration_diagnostics=diagnostics,
    )


def _defective_main_without_unconditional_report(*, job_id: str, preflight: bool) -> int:
    """Pre-fix control flow: no finally block, no guaranteed JSON/DONE emission."""
    import engine.builder2_media_finalization_resume as finalization_resume

    report = finalization_resume.run_one_media_finalization_resume(job_id=job_id, preflight=preflight)
    return 0 if report.get("ok") else 1


class TestBuilder2MediaFinalizationCliContract(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_successful_preflight_prints_one_json_object(self, run_one: Any) -> None:
        run_one.return_value = {
            "jobId": JOB_ID,
            "ok": True,
            "preflight": True,
            "readyForFinalizationRecovery": True,
        }
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue().strip())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["readyForFinalizationRecovery"])

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_successful_preflight_logs_done(self, run_one: Any) -> None:
        run_one.return_value = {"jobId": JOB_ID, "ok": True, "preflight": True}
        with patch("sys.stdout", io.StringIO()):
            with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                main()
        joined = "\n".join(captured.output)
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_RESUME_DONE", joined)

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_exit_code_zero_only_after_report_emission(self, run_one: Any) -> None:
        run_one.return_value = {"jobId": JOB_ID, "ok": True, "preflight": True}
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 0)
        self.assertTrue(buffer.getvalue().strip().startswith("{"))

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_rejected_duration_prints_json_and_exits_nonzero(self, run_one: Any) -> None:
        run_one.return_value = {
            "jobId": JOB_ID,
            "ok": False,
            "preflight": True,
            "failureStage": "duration_verification",
            "failureReason": "builder2_media_final_duration_out_of_tolerance",
            "measuredFinalDurationSeconds": 13.034,
        }
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failureStage"], "duration_verification")

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_ffmpeg_failure_prints_json_and_exits_nonzero(self, run_one: Any) -> None:
        run_one.return_value = {
            "jobId": JOB_ID,
            "ok": False,
            "preflight": True,
            "failureStage": "card_generation",
            "failureReason": "builder2_closure_ffmpeg_failed",
            "safeFfmpegReturnCode": 1,
        }
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["failureStage"], "card_generation")

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_ffprobe_failure_prints_json_and_exits_nonzero(self, run_one: Any) -> None:
        run_one.return_value = {
            "jobId": JOB_ID,
            "ok": False,
            "preflight": True,
            "failureStage": "duration_probe",
            "failureReason": "builder2_closure_ffprobe_failed",
        }
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["failureReason"], "builder2_closure_ffprobe_failed")

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_unexpected_exception_converted_to_sanitized_report(self, run_one: Any) -> None:
        run_one.side_effect = RuntimeError("boom")
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failureReason"], "builder2_media_finalization_unexpected_internal_error")
        self.assertNotIn("boom", buffer.getvalue())

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_stdout_is_explicitly_flushed(self, run_one: Any) -> None:
        run_one.return_value = {"jobId": JOB_ID, "ok": True, "preflight": True}
        with patch("builtins.print") as print_mock:
            with patch("sys.stdout.flush") as flush_mock:
                main()
        print_mock.assert_called()
        self.assertTrue(print_mock.call_args.kwargs.get("flush"))
        flush_mock.assert_called()

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_mocked_closure_renderer_returns_normally_to_runner(self, run_one: Any) -> None:
        run_one.return_value = {
            "jobId": JOB_ID,
            "ok": True,
            "preflight": True,
            "closureRenderAccepted": True,
            "readyForFinalizationRecovery": True,
        }
        code = main()
        self.assertEqual(code, 0)
        run_one.assert_called_once_with(job_id=JOB_ID, preflight=True)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_job_id_emits_json_and_exits_nonzero(self) -> None:
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertFalse(payload["ok"])


class TestBuilder2MediaFinalizationCliRegression(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_defective_pattern_exits_without_report(self, run_one: Any) -> None:
        def _abort_after_closure_start(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            logging.getLogger("engine.builder2_closure_render").info(
                "BUILDER2_CLOSURE_ENDCARD start jobId=%s",
                JOB_ID,
            )
            raise SystemExit(0)

        run_one.side_effect = _abort_after_closure_start
        with self.assertRaises(SystemExit) as raised:
            _defective_main_without_unconditional_report(job_id=JOB_ID, preflight=True)
        self.assertEqual(raised.exception.code, 0)

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_fixed_main_emits_report_after_closure_start_system_exit(self, run_one: Any) -> None:
        def _abort_after_closure_start(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            logging.getLogger("engine.builder2_closure_render").info(
                "BUILDER2_CLOSURE_ENDCARD start jobId=%s",
                JOB_ID,
            )
            raise SystemExit(0)

        run_one.side_effect = _abort_after_closure_start
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                code = main()
        self.assertEqual(code, 0)
        self.assertTrue(buffer.getvalue().strip().startswith("{"))
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_RESUME_DONE", "\n".join(captured.output))


class TestBuilder2MediaFinalizationCliLibraryBoundaries(unittest.TestCase):
    def test_closure_helpers_do_not_raise_system_exit(self) -> None:
        source = Path("engine/builder2_closure_render.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                if isinstance(func, ast.Name) and func.id == "SystemExit":
                    self.fail("builder2_closure_render must not raise SystemExit")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"exit", "_exit"}:
                    self.fail("builder2_closure_render must not terminate the process")

    def test_finalization_resume_helpers_do_not_raise_system_exit(self) -> None:
        source = Path("engine/builder2_media_finalization_resume.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in ast.walk(module):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                continue
        body = module.body
        main_guard_idx = next(
            i
            for i, node in enumerate(body)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        library_nodes = body[:main_guard_idx]
        library_source = ast.unparse(ast.Module(body=library_nodes, type_ignores=[]))
        self.assertNotIn("SystemExit(", library_source)
        self.assertNotIn("sys.exit(", library_source)
        self.assertNotIn("os._exit(", library_source)

    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    def test_preflight_does_not_mutate_redis_or_call_paid_services(
        self,
        _redis: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        source_decision: Any,
        closure_render: Any,
    ) -> None:
        from engine.builder2_media_finalization_source import FinalizationSourceDecision

        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        headline_path = Path("headline.mp4")
        source_decision.return_value = FinalizationSourceDecision(
            source_kind="legacy_headline_artifact",
            closure_input_path=headline_path,
        )
        closure_render.return_value = _success_closure_result()
        state = _false_completion_state(with_valid_closure=False)
        state["advertisingClosure"]["durationSeconds"] = 3.0
        with patch("engine.builder2_media_finalization_resume._probe_duration", return_value=10.042):
            with patch("engine.builder2_media_finalization_resume.save_tournament_state") as save_state:
                with patch("engine.builder2_media_finalization_resume.video_job_mark_done") as mark_done:
                    report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertTrue(report["ok"])
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["redisMutations"], 0)
        save_state.assert_not_called()
        mark_done.assert_not_called()
        closure_render.assert_called_once()
        passed_duration = closure_render.call_args.kwargs["duration_seconds"]
        self.assertEqual(passed_duration, 3.0)
        self.assertAlmostEqual(
            resolve_builder2_effective_closure_segment_duration_seconds(passed_duration),
            2.0,
            places=3,
        )

    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume.resolve_finalization_source_decision")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    def test_expected_duration_remains_approximately_12_042(
        self,
        _redis: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        source_decision: Any,
        closure_render: Any,
    ) -> None:
        from engine.builder2_media_finalization_source import FinalizationSourceDecision

        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        source_decision.return_value = FinalizationSourceDecision(
            source_kind="legacy_headline_artifact",
            closure_input_path=Path("headline.mp4"),
        )
        closure_render.return_value = _success_closure_result(measured_duration_seconds=12.042)
        state = _false_completion_state(with_valid_closure=False)
        state["advertisingClosure"]["durationSeconds"] = 3.0
        with patch("engine.builder2_media_finalization_resume._probe_duration", return_value=10.042):
            report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertAlmostEqual(report["calculatedExpectedFinalDurationSeconds"], 12.042, places=3)


class TestBuilder2MediaFinalizationCliModuleEntry(unittest.TestCase):
    def test_module_boundary_raises_system_exit_after_main(self) -> None:
        source = Path("engine/builder2_media_finalization_resume.py").read_text(encoding="utf-8")
        self.assertIn("raise SystemExit(main())", source)
        self.assertNotIn("sys.exit(main())", source)

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_main_env_interface_exercised(self, run_one: Any) -> None:
        run_one.return_value = {
            "jobId": JOB_ID,
            "ok": True,
            "preflight": True,
            "readyForFinalizationRecovery": True,
        }
        code = main()
        self.assertEqual(code, 0)
        run_one.assert_called_once_with(job_id=JOB_ID, preflight=True)


class TestBuilder2ClosureRenderReturnsNormally(unittest.TestCase):
    def test_successful_mocked_render_returns_typed_result(self) -> None:
        def _fake_runner(cmd: list[str], stage: str, category: str) -> None:
            _ = (stage, category)
            target = Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"video")

        with patch("engine.builder2_closure_render._ffprobe_duration_seconds", side_effect=[10.042, 12.042]):
            with patch("engine.builder2_closure_render._input_has_audio", return_value=False):
                with patch("engine.builder2_closure_render._default_font_path", return_value="/fonts/default.ttf"):
                    with patch("engine.builder2_closure_render._ffmpeg_bin", return_value="/usr/bin/ffmpeg"):
                        with patch("pathlib.Path.is_file", return_value=True):
                            with patch("pathlib.Path.read_bytes", return_value=b"video"):
                                with patch("pathlib.Path.write_text"):
                                    with patch("pathlib.Path.replace"):
                                        result = render_builder2_advertising_closure_endcard(
                                                "file:///tmp/in.mp4",
                                                "https://ace.example.com",
                                                product_name="Product",
                                                slogan="Slogan",
                                                duration_seconds=3.0,
                                                job_id=JOB_ID,
                                                publish=False,
                                                output_path=Path(tempfile.gettempdir()) / "builder2_closure_cli_out.mp4",
                                                ffmpeg_runner=_fake_runner,
                                            )
        self.assertIsInstance(result, ClosureRenderResult)
        self.assertAlmostEqual(result.measured_duration_seconds, 12.042, places=3)


if __name__ == "__main__":
    unittest.main()
