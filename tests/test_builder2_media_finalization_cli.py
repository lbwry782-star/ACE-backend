"""
Builder2 media finalization resume CLI contract tests.
"""
from __future__ import annotations

import ast
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from engine.builder2_closure_render import (
    Builder2ClosureRenderError,
    ClosureRenderResult,
    build_final_duration_verification_diagnostics,
    render_builder2_advertising_closure_endcard,
)
from engine.builder2_media_finalization_reporting import (
    build_minimal_fallback_report,
    emit_fail_safe_media_finalization_report,
    json_safe_value,
    preserve_original_failure,
    sanitize_media_finalization_report,
)
from engine.builder2_media_finalization_resume import (
    emit_media_finalization_resume_report,
    main,
    run_finalization_preflight,
    run_one_media_finalization_resume,
)
from engine.builder2_new_format_config import resolve_builder2_effective_closure_segment_duration_seconds
from tests.test_builder2_media_finalization_failure_inspect import (
    CLOSURE_URL,
    HEADLINE_URL,
    JOB_ID,
    _false_completion_state,
    _job_raw,
)
from tests.builder2_preflight_test_helpers import patch_accepted_web_storage_capability


def _success_closure_result(**overrides: Any) -> ClosureRenderResult:
    measured = overrides.get("measured_duration_seconds", 13.542)
    diagnostics = build_final_duration_verification_diagnostics(
        measured,
        visual_duration_seconds=10.042,
        effective_closure_segment_duration_seconds=3.5,
    )
    return ClosureRenderResult(
        public_url=overrides.get("public_url", ""),
        local_path=overrides.get("local_path", "/tmp/out.mp4"),
        measured_duration_seconds=measured,
        output_token=overrides.get("output_token", "tok" * 8),
        input_fingerprint=overrides.get("input_fingerprint", "abc"),
        duration_diagnostics=diagnostics,
    )


def _mock_closure_render_writes_output(*_args: Any, **kwargs: Any) -> ClosureRenderResult:
    output_path = kwargs["output_path"]
    output_path.write_bytes(b"video")
    measured = kwargs.get("measured_duration_seconds", 13.542)
    return _success_closure_result(
        local_path=str(output_path),
        public_url="",
        measured_duration_seconds=measured,
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
        with patch("sys.stdout.write") as write_mock:
            with patch("sys.stdout.flush") as flush_mock:
                main()
        write_mock.assert_called()
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

    @patch_accepted_web_storage_capability()
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
        _capability: Any,
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
        closure_render.side_effect = _mock_closure_render_writes_output
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
            3.5,
            places=3,
        )

    @patch_accepted_web_storage_capability()
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
        _capability: Any,
    ) -> None:
        from engine.builder2_media_finalization_source import FinalizationSourceDecision

        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")
        source_decision.return_value = FinalizationSourceDecision(
            source_kind="legacy_headline_artifact",
            closure_input_path=Path("headline.mp4"),
        )
        closure_render.side_effect = lambda *args, **kwargs: _mock_closure_render_writes_output(
            *args,
            measured_duration_seconds=13.542,
            **kwargs,
        )
        state = _false_completion_state(with_valid_closure=False)
        state["advertisingClosure"]["durationSeconds"] = 3.0
        with patch("engine.builder2_media_finalization_resume._probe_duration", return_value=10.042):
            report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertAlmostEqual(report["calculatedExpectedFinalDurationSeconds"], 13.542, places=3)


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

        with patch("engine.builder2_closure_render._ffprobe_duration_seconds", side_effect=[10.042, 3.5, 13.542]):
            with patch("engine.builder2_closure_render._input_has_audio", return_value=False):
                with patch("engine.builder2_closure_render._default_font_path", return_value="/fonts/default.ttf"):
                    with patch("engine.builder2_closure_render._ffmpeg_bin", return_value="/usr/bin/ffmpeg"):
                        with patch("pathlib.Path.is_file", return_value=True):
                            with patch("pathlib.Path.read_bytes", return_value=b"video"):
                                with patch("pathlib.Path.write_text"):
                                    with patch("pathlib.Path.replace"):
                                        result = render_builder2_advertising_closure_endcard(
                                                "file:///tmp/in.mp4",
                                                product_name="Product",
                                                slogan="Slogan",
                                                duration_seconds=3.0,
                                                job_id=JOB_ID,
                                                output_path=Path(tempfile.gettempdir()) / "builder2_closure_cli_out.mp4",
                                                ffmpeg_runner=_fake_runner,
                                            )
        self.assertIsInstance(result, ClosureRenderResult)
        self.assertEqual(result.public_url, "")
        self.assertAlmostEqual(result.measured_duration_seconds, 13.542, places=3)


def _concat_failure_report(*, failure_reason: str = "builder2_closure_ffmpeg_failed") -> Dict[str, Any]:
    return {
        "jobId": JOB_ID,
        "ok": False,
        "preflight": False,
        "failureStage": "concatenation",
        "failureReason": failure_reason,
        "originalFailureClass": "Builder2ClosureRenderError",
        "originalFailureStage": "concatenation",
        "originalFailureCode": failure_reason,
        "closureRenderAttempted": True,
        "closureFfmpegExecutionAccepted": True,
        "safeFfmpegReturnCode": 1,
        "leaseReleaseAttempted": True,
        "leaseReleaseAccepted": True,
        "openAICalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
    }


class TestBuilder2RecoveryFailSafeReporting(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(
        os.environ,
        {"BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID},
        clear=False,
    )
    def test_concat_builder2_error_emits_json_done_and_exit_one(self, run_one: Any) -> None:
        run_one.return_value = _concat_failure_report()
        buffer = io.StringIO()

        def _write(data: str) -> int:
            buffer.write(data)
            return len(data)

        with patch("sys.stdout.write", side_effect=_write):
            with patch("sys.stdout.flush"):
                with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                    code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failureStage"], "concatenation")
        self.assertEqual(payload["originalFailureCode"], "builder2_closure_ffmpeg_failed")
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_RESUME_DONE", "\n".join(captured.output))

    @patch("engine.builder2_media_finalization_resume.render_builder2_advertising_closure_endcard")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease")
    @patch.dict(os.environ, {"BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID}, clear=False)
    def test_real_main_recovery_concat_error(
        self,
        release_lease: Any,
        acquire_lease: Any,
        _redis: Any,
        read_raw: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        pipeline: Any,
        closure_render: Any,
    ) -> None:
        read_raw.return_value = _false_completion_state(with_valid_closure=False)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")

        def _fail_pipeline(**kwargs: Any) -> None:
            report = kwargs["report"]
            report["failureStage"] = "concatenation"
            report["failureReason"] = "builder2_closure_ffmpeg_failed"
            report["originalFailureStage"] = "concatenation"
            report["originalFailureCode"] = "builder2_closure_ffmpeg_failed"
            report["originalFailureClass"] = "Builder2ClosureRenderError"
            report["closureRenderAttempted"] = True

        pipeline.side_effect = _fail_pipeline
        buffer = io.StringIO()

        def _write(data: str) -> int:
            buffer.write(data)
            return len(data)

        with patch("sys.stdout.write", side_effect=_write):
            with patch("sys.stdout.flush"):
                with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                    code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["failureStage"], "concatenation")
        release_lease.assert_called_once()
        self.assertTrue(payload.get("leaseReleaseAttempted"))
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_RESUME_DONE", "\n".join(captured.output))
        closure_render.assert_not_called()

    def test_path_values_are_json_safe(self) -> None:
        converted = json_safe_value(Path("/tmp/secret/out.mp4"))
        self.assertEqual(converted, "Path")
        payload = sanitize_media_finalization_report({"jobId": JOB_ID, "failureStage": Path("concatenation")})
        self.assertEqual(payload["failureStage"], "Path")
        json.dumps(payload, allow_nan=False)

    def test_exception_values_are_json_safe(self) -> None:
        converted = json_safe_value(RuntimeError("secret"))
        self.assertEqual(converted, "RuntimeError")
        payload = sanitize_media_finalization_report(
            {"jobId": JOB_ID, "failureReason": RuntimeError("builder2_closure_ffmpeg_failed")}
        )
        self.assertEqual(payload["failureReason"], "RuntimeError")
        json.dumps(payload, allow_nan=False)

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(os.environ, {"BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID}, clear=False)
    def test_forced_serialization_failure_emits_minimal_fallback(self, run_one: Any) -> None:
        run_one.return_value = _concat_failure_report()
        real_dumps = json.dumps
        attempts = {"count": 0}

        def _dumps_side_effect(*args: Any, **kwargs: Any) -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise TypeError("boom")
            return real_dumps(*args, **kwargs)

        buffer = io.StringIO()

        def _write(data: str) -> int:
            buffer.write(data)
            return len(data)

        with patch("engine.builder2_media_finalization_reporting.json.dumps", side_effect=_dumps_side_effect):
            with patch("sys.stdout.write", side_effect=_write):
                with patch("sys.stdout.flush"):
                    with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                        code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["failureReason"], "final_report_serialization_failed")
        self.assertTrue(payload["reportingFailureOccurred"])
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_RESUME_DONE", "\n".join(captured.output))

    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume.video_job_get")
    @patch("engine.builder2_media_finalization_resume.video_job_get_raw")
    @patch("engine.builder2_media_finalization_resume._read_raw")
    @patch("engine.builder2_media_finalization_resume.redis_configured", return_value=True)
    @patch("engine.builder2_media_finalization_resume.acquire_job_lease", return_value=True)
    @patch("engine.builder2_media_finalization_resume.release_job_lease", side_effect=RuntimeError("lease boom"))
    @patch("engine.builder2_media_finalization_resume.save_tournament_state")
    def test_lease_release_failure_preserves_original_failure_and_reports(
        self,
        save_state: Any,
        release_lease: Any,
        _acquire: Any,
        _redis: Any,
        read_raw: Any,
        job_get_raw: Any,
        job_get: Any,
        build_config: Any,
        pipeline: Any,
    ) -> None:
        read_raw.return_value = _false_completion_state(with_valid_closure=False)
        job_get_raw.return_value = _job_raw(video_url=HEADLINE_URL)
        job_get.return_value = {"publicBaseUrl": "https://ace.example.com"}
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com")

        def _fail_pipeline(**kwargs: Any) -> None:
            exc = Builder2ClosureRenderError(
                "builder2_closure_ffmpeg_failed",
                stage="concatenation",
                return_code=1,
            )
            preserve_original_failure(kwargs["report"], exc)
            kwargs["report"]["failureStage"] = exc.stage
            kwargs["report"]["failureReason"] = str(exc.args[0])

        pipeline.side_effect = _fail_pipeline
        report = run_one_media_finalization_resume(job_id=JOB_ID, acquire_lease=True)
        self.assertEqual(report["originalFailureStage"], "concatenation")
        self.assertTrue(report["leaseReleaseAttempted"])
        self.assertFalse(report["leaseReleaseAccepted"])
        save_state.assert_called_once()

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(os.environ, {"BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID}, clear=False)
    def test_system_exit_one_after_card_still_reports(self, run_one: Any) -> None:
        run_one.side_effect = SystemExit(1)
        buffer = io.StringIO()

        def _write(data: str) -> int:
            buffer.write(data)
            return len(data)

        with patch("sys.stdout.write", side_effect=_write):
            with patch("sys.stdout.flush"):
                with self.assertLogs("engine.builder2_media_finalization_resume", level="INFO") as captured:
                    code = main()
        self.assertEqual(code, 1)
        self.assertTrue(buffer.getvalue().strip().startswith("{"))
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_RESUME_DONE", "\n".join(captured.output))

    @patch("engine.builder2_media_finalization_resume.run_one_media_finalization_resume")
    @patch.dict(os.environ, {"BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID}, clear=False)
    def test_main_fallback_when_emit_raises(self, run_one: Any) -> None:
        run_one.return_value = _concat_failure_report()
        fallback = build_minimal_fallback_report(job_id=JOB_ID, preflight=False)
        with patch(
            "engine.builder2_media_finalization_resume.emit_media_finalization_resume_report",
            side_effect=[RuntimeError("emit boom"), fallback],
        ):
            with patch(
                "engine.builder2_media_finalization_resume.emit_fail_safe_media_finalization_report",
                return_value=fallback,
            ) as fallback_emit:
                buffer = io.StringIO()

                def _write(data: str) -> int:
                    buffer.write(data)
                    return len(data)

                with patch("sys.stdout.write", side_effect=_write):
                    with patch("sys.stdout.flush"):
                        code = main()
        self.assertEqual(code, 1)
        fallback_emit.assert_called_once()

    def test_emit_fail_safe_never_uses_os_exit(self) -> None:
        source = Path("engine/builder2_media_finalization_reporting.py").read_text(encoding="utf-8")
        self.assertNotIn("os._exit", source)

    @patch("engine.builder2_media_finalization_resume.run_finalization_preflight")
    @patch.dict(
        os.environ,
        {
            "BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID": JOB_ID,
            "BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT": "true",
        },
        clear=False,
    )
    def test_successful_preflight_behavior_unchanged(self, preflight_fn: Any) -> None:
        preflight_fn.return_value = {
            "jobId": JOB_ID,
            "ok": True,
            "preflight": True,
            "readyForFinalizationRecovery": True,
            "measuredFinalDurationSeconds": 13.534,
        }
        buffer = io.StringIO()

        def _write(data: str) -> int:
            buffer.write(data)
            return len(data)

        with patch("sys.stdout.write", side_effect=_write):
            with patch("sys.stdout.flush"):
                code = main()
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue().strip())
        self.assertTrue(payload["readyForFinalizationRecovery"])
        self.assertAlmostEqual(payload["measuredFinalDurationSeconds"], 13.534, places=3)


if __name__ == "__main__":
    unittest.main()
