"""
Builder2 media finalization parent process probe tests — mocks and synthetic children only.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import unittest
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from engine.builder2_media_finalization_process_probe import (
    ChildObservationState,
    _classify_diagnostic,
    main,
    run_media_finalization_process_probe,
    sanitize_process_probe_report,
)
from engine.builder2_media_finalization_resume import run_finalization_preflight
from tests.test_builder2_media_finalization_failure_inspect import (
    HEADLINE_URL,
    JOB_ID,
    _false_completion_state,
)


def _python_cmd(script: str) -> List[str]:
    return [sys.executable, "-u", "-c", textwrap.dedent(script)]


def _success_child_script() -> List[str]:
    return _python_cmd(
        """
        import json, sys, os
        print("INFO BUILDER2_MEDIA_FINALIZATION_RESUME_START jobId=test preflight=True", flush=True)
        print("INFO BUILDER2_CLOSURE_ENDCARD start jobId=test", flush=True)
        print("INFO BUILDER2_CLOSURE_CARD_RENDER_COMPLETED ffmpegReturnCode=0 outputCreated=true", flush=True)
        os.write(2, b"BUILDER2_CLOSURE_CONCAT_INVOKE stage=concatenation elapsedMs=1.0 cardOutputCreated=true\\n")
        print("INFO BUILDER2_CLOSURE_CONCAT_COMPLETED ffmpegReturnCode=0 outputCreated=true", flush=True)
        os.write(2, b"BUILDER2_CLOSURE_CONCAT_RETURNED stage=concatenation elapsedMs=2.0 concatReturnAccepted=true\\n")
        print("INFO BUILDER2_CLOSURE_OUTPUT_PROBED measuredFinalDurationSeconds=12.034", flush=True)
        print("INFO BUILDER2_CLOSURE_DURATION_VERIFIED durationAccepted=true", flush=True)
        print("INFO BUILDER2_CLOSURE_ENDCARD_DONE elapsedMs=3.0 durationAccepted=true", flush=True)
        os.write(2, b"BUILDER2_CLOSURE_RESULT_RETURNING stage=completion elapsedMs=3.0 durationAccepted=true\\n")
        print("INFO BUILDER2_MEDIA_FINALIZATION_RESUME_DONE jobId=test ok=True preflight=True", flush=True)
        print("INFO BUILDER2_MEDIA_FINALIZATION_REPORT_EMITTED jobId=test stdoutWriteAccepted=True", flush=True)
        payload = {
            "jobId": "test",
            "ok": True,
            "preflight": True,
            "readyForFinalizationRecovery": True,
            "redisMutations": 0,
            "publicationCalls": 0,
            "openAICalls": 0,
            "imageCalls": 0,
            "runwaySubmissionCalls": 0,
            "runwayPollingCalls": 0,
        }
        print(json.dumps(payload, indent=2), flush=True)
        os.write(2, b"BUILDER2_MEDIA_FINALIZATION_CHILD_ATEXIT\\n")
        """
    )


class TestProbeClassification(unittest.TestCase):
    def _obs(self, **milestones: bool) -> ChildObservationState:
        obs = ChildObservationState()
        for key, value in milestones.items():
            obs.milestones[key] = value
        return obs

    def test_success_classification(self) -> None:
        obs = self._obs(
            finalJsonObserved=True,
            finalDoneLogObserved=True,
            reportEmittedAcknowledgementObserved=True,
        )
        obs.child_reported_ok = True
        obs.child_reported_ready = True
        self.assertEqual(
            _classify_diagnostic(
                observation=obs,
                child_exited_normally=True,
                child_terminated_by_signal=False,
                child_timed_out=False,
                return_code=0,
            ),
            "child_completed_preflight_successfully",
        )

    def test_exit_zero_without_report(self) -> None:
        obs = self._obs(cardRenderCompletedObserved=True)
        self.assertEqual(
            _classify_diagnostic(
                observation=obs,
                child_exited_normally=True,
                child_terminated_by_signal=False,
                child_timed_out=False,
                return_code=0,
            ),
            "child_exited_zero_without_reporting",
        )

    def test_exit_nonzero_with_report(self) -> None:
        obs = self._obs(finalJsonObserved=True)
        self.assertEqual(
            _classify_diagnostic(
                observation=obs,
                child_exited_normally=False,
                child_terminated_by_signal=False,
                child_timed_out=False,
                return_code=1,
            ),
            "child_exited_nonzero_with_report",
        )

    def test_exit_nonzero_without_report(self) -> None:
        obs = ChildObservationState()
        self.assertEqual(
            _classify_diagnostic(
                observation=obs,
                child_exited_normally=False,
                child_terminated_by_signal=False,
                child_timed_out=False,
                return_code=1,
            ),
            "child_exited_nonzero_without_report",
        )

    def test_log_tail_loss_suspected(self) -> None:
        obs = self._obs(closureEndcardDoneObserved=True)
        self.assertEqual(
            _classify_diagnostic(
                observation=obs,
                child_exited_normally=True,
                child_terminated_by_signal=False,
                child_timed_out=False,
                return_code=0,
            ),
            "render_log_tail_loss_suspected",
        )


class TestProbeChildProcess(unittest.TestCase):
    def test_detects_successful_child_with_json_and_done(self) -> None:
        report = run_media_finalization_process_probe(
            JOB_ID,
            command=_success_child_script(),
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertTrue(report["childSpawnAccepted"])
        self.assertTrue(report["childExitedNormally"])
        self.assertEqual(report["diagnosticClassification"], "child_completed_preflight_successfully")
        self.assertTrue(report["ok"])
        self.assertTrue(report["childFinalJsonObserved"])
        self.assertTrue(report["childDoneObserved"])
        self.assertTrue(report["childReportAcknowledgementObserved"])
        self.assertTrue(report["childReportedReadyForRecovery"])
        self.assertEqual(report["childLastSafeMilestone"], "childAtexitObserved")

    def test_exit_zero_without_json(self) -> None:
        script = _python_cmd(
            """
            print("INFO BUILDER2_CLOSURE_CARD_RENDER_COMPLETED outputCreated=true", flush=True)
            """
        )
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=5,
            command=script,
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertTrue(report["childExitedNormally"])
        self.assertEqual(report["diagnosticClassification"], "child_exited_zero_without_reporting")
        self.assertFalse(report["ok"])
        self.assertFalse(report["childFinalJsonObserved"])

    def test_exit_one_with_failure_report(self) -> None:
        script = _python_cmd(
            """
            import json, sys
            print("INFO BUILDER2_MEDIA_FINALIZATION_RESUME_DONE jobId=test ok=False preflight=True", flush=True)
            print("INFO BUILDER2_MEDIA_FINALIZATION_REPORT_EMITTED jobId=test", flush=True)
            print(json.dumps({"jobId":"test","ok":False,"preflight":True,"failureStage":"concatenation"}), flush=True)
            sys.exit(1)
            """
        )
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=5,
            command=script,
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertEqual(report["childReturnCode"], 1)
        self.assertEqual(report["diagnosticClassification"], "child_exited_nonzero_with_report")
        self.assertFalse(report["ok"])

    def test_exit_one_without_report(self) -> None:
        script = _python_cmd("import sys; sys.exit(1)")
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=5,
            command=script,
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertEqual(report["diagnosticClassification"], "child_exited_nonzero_without_report")

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM unavailable")
    def test_detects_signal_termination(self) -> None:
        script = _python_cmd(
            """
            import os, signal, time
            print("INFO BUILDER2_CLOSURE_CARD_RENDER_COMPLETED outputCreated=true", flush=True)
            os.write(2, b"BUILDER2_MEDIA_FINALIZATION_CHILD_SIGNAL signal=SIGTERM\\n")
            signal.raise_signal(signal.SIGTERM)
            """
        )
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=5,
            command=script,
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertIn(
            report["diagnosticClassification"],
            {"child_terminated_by_signal", "child_exited_nonzero_without_report"},
        )
        self.assertFalse(report["ok"])

    def test_timeout_after_card_completion(self) -> None:
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=0.5,
            command=_python_cmd(
                """
                import os, time
                print("INFO BUILDER2_CLOSURE_CARD_RENDER_COMPLETED outputCreated=true", flush=True)
                os.write(2, b"BUILDER2_CLOSURE_CONCAT_INVOKE stage=concatenation elapsedMs=1.0 cardOutputCreated=true\\n")
                time.sleep(30)
                """
            ),
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertTrue(report["childTimedOut"])
        self.assertEqual(report["diagnosticClassification"], "child_timed_out_inside_or_after_concat")
        self.assertEqual(report["childLastSafeMilestone"], "concatStartObserved")

    def test_streams_stdout_and_stderr_with_prefixes(self) -> None:
        captured: List[str] = []
        script = _python_cmd(
            """
            import sys
            print("stdout-line", flush=True)
            print("stderr-line", file=sys.stderr, flush=True)
            """
        )
        run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=5,
            command=script,
            child_env=os.environ.copy(),
            emit=captured.append,
        )
        joined = "\n".join(captured)
        self.assertIn("BUILDER2_CHILD_STDOUT stdout-line", joined)
        self.assertIn("BUILDER2_CHILD_STDERR stderr-line", joined)

    def test_large_output_does_not_deadlock_parent(self) -> None:
        script = _python_cmd(
            """
            for i in range(4000):
                print(f"line-{i}", flush=True)
            import json
            print(json.dumps({"jobId":"test","ok":True,"preflight":True,"readyForFinalizationRecovery":True}), flush=True)
            print("INFO BUILDER2_MEDIA_FINALIZATION_RESUME_DONE ok=True", flush=True)
            print("INFO BUILDER2_MEDIA_FINALIZATION_REPORT_EMITTED", flush=True)
            """
        )
        started = time.monotonic()
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=30,
            command=script,
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        self.assertLess(time.monotonic() - started, 20)
        self.assertGreater(report["childOutputLineCount"], 3000)
        self.assertTrue(report["childStdoutEofObserved"])

    def test_parent_report_excludes_unsafe_content(self) -> None:
        script = _python_cmd(
            """
            print("SECRET HEADLINE https://ace.example.com/api/video-headline/token /tmp/path ffmpeg", flush=True)
            """
        )
        report = run_media_finalization_process_probe(
            JOB_ID,
            timeout_seconds=5,
            command=script,
            child_env=os.environ.copy(),
            emit=lambda _line: None,
        )
        blob = json.dumps(sanitize_process_probe_report(report))
        self.assertNotIn("https://", blob)
        self.assertNotIn("SECRET", blob)
        self.assertNotIn("/tmp/", blob)
        self.assertNotIn("ffmpeg", blob.lower())


class TestClosureLowLevelMarkers(unittest.TestCase):
    @patch("engine.builder2_closure_render._run_checked")
    @patch("engine.builder2_closure_render._ffprobe_duration_seconds", side_effect=[10.042, 12.034])
    @patch("engine.builder2_closure_render._input_has_audio", return_value=False)
    @patch("engine.builder2_closure_render._default_font_path", return_value="/fonts/default.ttf")
    @patch("engine.builder2_closure_render._ffmpeg_bin", return_value="/usr/bin/ffmpeg")
    def test_concat_invoke_before_runner_and_returned_after(
        self,
        _ffmpeg: Any,
        _font: Any,
        _audio: Any,
        _probe: Any,
        run_checked: Any,
    ) -> None:
        from engine.builder2_closure_render import render_builder2_advertising_closure_endcard

        marker_lines: List[bytes] = []
        original_write = os.write
        caller_output = Path(os.environ.get("TEMP", "/tmp")) / "closure_marker_caller_out.mp4"

        def _capture_write(fd: int, data: bytes) -> int:
            if fd == 2:
                marker_lines.append(data)
            return original_write(fd, data)

        def _run(cmd: Any, stage: str, category: str) -> int:
            Path(str(cmd[-1])).write_bytes(b"fake-video")
            return 0

        run_checked.side_effect = _run
        source = Path(os.environ.get("TEMP", "/tmp")) / "closure_source.mp4"
        source.write_bytes(b"fake")

        with patch("engine.builder2_closure_render.os.write", side_effect=_capture_write):
            result = render_builder2_advertising_closure_endcard(
                str(source),
                product_name="Product",
                slogan="Slogan",
                language="en",
                duration_seconds=3.0,
                output_path=caller_output,
            )

        joined = b"".join(marker_lines).decode("utf-8", errors="replace")
        invoke_pos = joined.find("BUILDER2_CLOSURE_CONCAT_INVOKE")
        returned_pos = joined.find("BUILDER2_CLOSURE_CONCAT_RETURNED")
        result_pos = joined.find("BUILDER2_CLOSURE_RESULT_RETURNING")
        self.assertGreater(invoke_pos, -1)
        self.assertGreater(returned_pos, invoke_pos)
        self.assertGreater(result_pos, returned_pos)
        self.assertAlmostEqual(result.measured_duration_seconds, 12.034, places=3)
        self.assertEqual(run_checked.call_count, 2)


class TestChildLifecycleDiagnostics(unittest.TestCase):
    def test_atexit_marker_on_normal_exit(self) -> None:
        script = _python_cmd(
            """
            from engine.builder2_media_finalization_child_diagnostics import register_child_lifecycle_diagnostics
            register_child_lifecycle_diagnostics()
            """
        )
        completed = subprocess.run(script, capture_output=True, text=True, timeout=10, check=False)
        self.assertIn("BUILDER2_MEDIA_FINALIZATION_CHILD_ATEXIT", completed.stderr)


class TestPreflightIsolationWithProbe(unittest.TestCase):
    @patch("engine.builder2_media_finalization_resume.build_media_resume_configuration")
    @patch("engine.builder2_media_finalization_resume._execute_finalization_render_pipeline")
    def test_preflight_still_zero_redis_and_publication(
        self,
        pipeline: Any,
        build_config: Any,
    ) -> None:
        def _ok(**kwargs: Any) -> None:
            kwargs["report"]["ok"] = True
            kwargs["report"]["readyForFinalizationRecovery"] = True

        pipeline.side_effect = _ok
        build_config.return_value = MagicMock(publicBaseUrl="https://ace.example.com", public_base_url=MagicMock(source="env"))
        state = _false_completion_state(with_valid_closure=False)
        report = run_finalization_preflight(job_id=JOB_ID, state=state, job_video_url=HEADLINE_URL)
        self.assertEqual(report["redisMutations"], 0)
        self.assertEqual(report["publicationCalls"], 0)
        self.assertEqual(report["openAICalls"], 0)
        self.assertEqual(report["imageCalls"], 0)
        self.assertEqual(report["runwaySubmissionCalls"], 0)
        self.assertEqual(report["runwayPollingCalls"], 0)


class TestProbeCli(unittest.TestCase):
    @patch.dict(os.environ, {"BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_JOB_ID": JOB_ID}, clear=False)
    @patch("engine.builder2_media_finalization_process_probe.run_media_finalization_process_probe")
    def test_cli_exits_nonzero_without_success(self, run_probe: Any) -> None:
        run_probe.return_value = {
            "jobId": JOB_ID,
            "ok": False,
            "probeCompleted": True,
            "diagnosticClassification": "child_exited_zero_without_reporting",
        }
        buffer = StringIO()
        with patch("sys.stdout", buffer):
            code = main()
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
