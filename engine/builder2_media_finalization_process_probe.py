"""
Builder2 media finalization preflight parent-process probe.

Run:
  BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_JOB_ID=<jobId> \\
    python -m engine.builder2_media_finalization_process_probe

Launches the real preflight CLI as a child, streams output safely, and emits a
primitive-only diagnostic JSON report. Performs no Redis mutations or publication.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = float(
    (os.environ.get("BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_TIMEOUT_SECONDS") or "600").strip() or "600"
)

_UNSAFE_PATTERNS = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"/(?:tmp|var|home|Users|private|data|storage)[^\s'\"]*", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\[^\s'\"]+"),
    re.compile(r"ffmpeg\b", re.IGNORECASE),
    re.compile(r"SECRET|SLOGAN|HEADLINE TEXT|Forgot Product", re.IGNORECASE),
    re.compile(r"ACE_VIDEO_HEADLINE_UPLOAD_SECRET|RUNWAY_API_KEY|OPENAI", re.IGNORECASE),
)

_MILESTONE_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("childStartObserved", "BUILDER2_MEDIA_FINALIZATION_RESUME_START"),
    ("closureStartObserved", "BUILDER2_CLOSURE_ENDCARD start"),
    ("cardRenderCompletedObserved", "BUILDER2_CLOSURE_CARD_RENDER_COMPLETED"),
    ("concatStartObserved", "BUILDER2_CLOSURE_CONCAT_INVOKE"),
    ("concatCompletedObserved", "BUILDER2_CLOSURE_CONCAT_RETURNED"),
    ("concatCompletedObserved", "BUILDER2_CLOSURE_CONCAT_COMPLETED"),
    ("closureOutputProbedObserved", "BUILDER2_CLOSURE_OUTPUT_PROBED"),
    ("closureDurationVerifiedObserved", "BUILDER2_CLOSURE_DURATION_VERIFIED"),
    ("closureEndcardDoneObserved", "BUILDER2_CLOSURE_ENDCARD_DONE"),
    ("closureResultReturningObserved", "BUILDER2_CLOSURE_RESULT_RETURNING"),
    ("finalDoneLogObserved", "BUILDER2_MEDIA_FINALIZATION_RESUME_DONE"),
    ("reportEmittedAcknowledgementObserved", "BUILDER2_MEDIA_FINALIZATION_REPORT_EMITTED"),
    ("childAtexitObserved", "BUILDER2_MEDIA_FINALIZATION_CHILD_ATEXIT"),
)

_SIGNAL_MARKER_PREFIX = "BUILDER2_MEDIA_FINALIZATION_CHILD_SIGNAL signal="

_REPORT_SAFE_KEYS: Tuple[str, ...] = (
    "jobId",
    "ok",
    "probeCompleted",
    "childSpawnAccepted",
    "childPidPresent",
    "childStdoutEofObserved",
    "childStderrEofObserved",
    "childReturnCode",
    "childExitedNormally",
    "childTerminatedBySignal",
    "childSignalNumber",
    "childTimedOut",
    "childRuntimeSeconds",
    "childOutputLineCount",
    "childLastSafeMilestone",
    "childFinalJsonObserved",
    "childDoneObserved",
    "childReportAcknowledgementObserved",
    "childReportingContractSatisfied",
    "diagnosticClassification",
    "childStartObserved",
    "closureStartObserved",
    "cardRenderCompletedObserved",
    "concatStartObserved",
    "concatCompletedObserved",
    "closureOutputProbedObserved",
    "closureDurationVerifiedObserved",
    "closureEndcardDoneObserved",
    "closureResultReturningObserved",
    "finalJsonObserved",
    "finalDoneLogObserved",
    "reportEmittedAcknowledgementObserved",
    "childReportedOk",
    "childReportedReadyForRecovery",
    "childAtexitObserved",
    "childSignalObserved",
    "openAICalls",
    "imageCalls",
    "runwaySubmissionCalls",
    "runwayPollingCalls",
    "publicationCalls",
    "redisMutations",
)


@dataclass
class ChildObservationState:
    milestones: Dict[str, bool] = field(default_factory=lambda: {key: False for key, _ in _MILESTONE_MARKERS})
    final_json_observed: bool = False
    child_reported_ok: Optional[bool] = None
    child_reported_ready: Optional[bool] = None
    parsed_final_json: Optional[Dict[str, Any]] = None
    child_signal_observed: Optional[str] = None
    output_line_count: int = 0
    stdout_buffer: List[str] = field(default_factory=list)

    def observe_line(self, line: str, *, is_stdout: bool = False) -> None:
        token = (line or "").strip()
        if not token:
            return
        self.output_line_count += 1
        if is_stdout:
            self.stdout_buffer.append(line)
            parsed = _extract_final_json("\n".join(self.stdout_buffer))
            if parsed is not None:
                self._apply_parsed_json(parsed)
        for key, marker in _MILESTONE_MARKERS:
            if marker in token:
                self.milestones[key] = True
        if token.startswith(_SIGNAL_MARKER_PREFIX):
            self.child_signal_observed = token[len(_SIGNAL_MARKER_PREFIX) :].strip() or None
        if not is_stdout and token.startswith("{") and '"jobId"' in token:
            parsed = _try_parse_json_object(token)
            if parsed is not None:
                self._apply_parsed_json(parsed)

    def _apply_parsed_json(self, parsed: Dict[str, Any]) -> None:
        self.parsed_final_json = parsed
        self.final_json_observed = True
        self.milestones["finalJsonObserved"] = True
        if "ok" in parsed:
            self.child_reported_ok = bool(parsed.get("ok"))
        if "readyForFinalizationRecovery" in parsed:
            self.child_reported_ready = bool(parsed.get("readyForFinalizationRecovery"))

    def last_safe_milestone(self) -> Optional[str]:
        order = [
            "childStartObserved",
            "closureStartObserved",
            "cardRenderCompletedObserved",
            "concatStartObserved",
            "concatCompletedObserved",
            "closureOutputProbedObserved",
            "closureDurationVerifiedObserved",
            "closureEndcardDoneObserved",
            "closureResultReturningObserved",
            "finalJsonObserved",
            "finalDoneLogObserved",
            "reportEmittedAcknowledgementObserved",
            "childAtexitObserved",
        ]
        last: Optional[str] = None
        for name in order:
            if self.milestones.get(name):
                last = name
        return last

    def reporting_contract_satisfied(self) -> bool:
        return bool(
            self.milestones.get("finalJsonObserved")
            and self.milestones.get("finalDoneLogObserved")
            and self.milestones.get("reportEmittedAcknowledgementObserved")
        )


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_final_json(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    last_obj: Optional[Dict[str, Any]] = None
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and ("jobId" in obj or "ok" in obj):
            last_obj = obj
        idx = max(end, start + 1)
    return last_obj


def _interpret_return_code(return_code: Optional[int]) -> Tuple[bool, bool, Optional[int]]:
    if return_code is None:
        return False, False, None
    if return_code < 0:
        return False, True, -return_code
    return return_code == 0, False, None


def _classify_diagnostic(
    *,
    observation: ChildObservationState,
    child_exited_normally: bool,
    child_terminated_by_signal: bool,
    child_timed_out: bool,
    return_code: Optional[int],
) -> str:
    if (
        child_exited_normally
        and observation.child_reported_ok is True
        and observation.child_reported_ready is True
        and observation.reporting_contract_satisfied()
    ):
        return "child_completed_preflight_successfully"

    if child_timed_out:
        if observation.milestones.get("cardRenderCompletedObserved") and not observation.milestones.get(
            "concatCompletedObserved"
        ):
            return "child_timed_out_inside_or_after_concat"
        return "child_timed_out_inside_or_after_concat"

    if child_terminated_by_signal or observation.child_signal_observed:
        return "child_terminated_by_signal"

    if child_exited_normally and not observation.milestones.get("finalJsonObserved"):
        if observation.milestones.get("closureEndcardDoneObserved") or observation.milestones.get(
            "closureResultReturningObserved"
        ):
            return "render_log_tail_loss_suspected"
        return "child_exited_zero_without_reporting"

    if return_code not in (None, 0):
        if observation.milestones.get("finalJsonObserved"):
            return "child_exited_nonzero_with_report"
        return "child_exited_nonzero_without_report"

    return "insufficient_evidence"


def _line_is_safe_for_report(line: str) -> bool:
    return not any(pattern.search(line) for pattern in _UNSAFE_PATTERNS)


def _forward_line(prefix: str, line: str, *, emit: Callable[[str], None]) -> None:
    safe = line.rstrip("\r\n")
    emit(f"{prefix} {safe}")


def _stream_reader(
    pipe: Optional[TextIO],
    *,
    prefix: str,
    observation: ChildObservationState,
    emit: Callable[[str], None],
    eof_flag: Dict[str, bool],
    is_stdout: bool,
) -> None:
    if pipe is None:
        eof_flag["value"] = True
        return
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            observation.observe_line(line, is_stdout=is_stdout)
            _forward_line(prefix, line, emit=emit)
    finally:
        eof_flag["value"] = True
        try:
            pipe.close()
        except Exception:
            pass


def _build_child_command() -> List[str]:
    return [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        "-m",
        "engine.builder2_media_finalization_resume",
    ]


def _build_child_env(job_id: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["BUILDER2_MEDIA_FINALIZATION_RESUME_PREFLIGHT"] = "true"
    env["BUILDER2_MEDIA_FINALIZATION_RESUME_JOB_ID"] = job_id
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_JOB_ID", None)
    return env


def run_media_finalization_process_probe(
    job_id: str,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    command: Optional[List[str]] = None,
    child_env: Optional[Dict[str, str]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "jobId": job_id,
        "ok": False,
        "probeCompleted": False,
        "childSpawnAccepted": False,
        "childPidPresent": False,
        "childStdoutEofObserved": False,
        "childStderrEofObserved": False,
        "childReturnCode": None,
        "childExitedNormally": False,
        "childTerminatedBySignal": False,
        "childSignalNumber": None,
        "childTimedOut": False,
        "childRuntimeSeconds": 0.0,
        "childOutputLineCount": 0,
        "childLastSafeMilestone": None,
        "childFinalJsonObserved": False,
        "childDoneObserved": False,
        "childReportAcknowledgementObserved": False,
        "childReportingContractSatisfied": False,
        "diagnosticClassification": "insufficient_evidence",
        "openAICalls": 0,
        "imageCalls": 0,
        "runwaySubmissionCalls": 0,
        "runwayPollingCalls": 0,
        "publicationCalls": 0,
        "redisMutations": 0,
    }
    for key, _ in _MILESTONE_MARKERS:
        report.setdefault(key, False)
    report.setdefault("childSignalObserved", None)

    writer = emit or (lambda text: print(text, flush=True))
    observation = ChildObservationState()
    started = time.monotonic()

    proc: Optional[subprocess.Popen[str]] = None
    stdout_eof = {"value": False}
    stderr_eof = {"value": False}
    try:
        proc = subprocess.Popen(
            command or _build_child_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env or _build_child_env(job_id),
            text=True,
            bufsize=1,
        )
        report["childSpawnAccepted"] = True
        report["childPidPresent"] = proc.pid is not None

        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(proc.stdout,),
            kwargs={
                "prefix": "BUILDER2_CHILD_STDOUT",
                "observation": observation,
                "emit": writer,
                "eof_flag": stdout_eof,
                "is_stdout": True,
            },
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(proc.stderr,),
            kwargs={
                "prefix": "BUILDER2_CHILD_STDERR",
                "observation": observation,
                "emit": writer,
                "eof_flag": stderr_eof,
                "is_stdout": False,
            },
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            return_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            report["childTimedOut"] = True
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return_code = proc.returncode

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        report["childStdoutEofObserved"] = bool(stdout_eof["value"])
        report["childStderrEofObserved"] = bool(stderr_eof["value"])
        report["childReturnCode"] = return_code
        report["childRuntimeSeconds"] = round(time.monotonic() - started, 3)

        exited_normally, terminated_by_signal, signal_number = _interpret_return_code(return_code)
        report["childExitedNormally"] = exited_normally
        report["childTerminatedBySignal"] = terminated_by_signal
        report["childSignalNumber"] = signal_number

        if observation.stdout_buffer:
            parsed = _extract_final_json("\n".join(observation.stdout_buffer))
            if parsed is not None:
                observation._apply_parsed_json(parsed)

        report["childOutputLineCount"] = observation.output_line_count
        report["childLastSafeMilestone"] = observation.last_safe_milestone()
        report["childFinalJsonObserved"] = observation.final_json_observed
        report["childDoneObserved"] = observation.milestones.get("finalDoneLogObserved", False)
        report["childReportAcknowledgementObserved"] = observation.milestones.get(
            "reportEmittedAcknowledgementObserved",
            False,
        )
        report["childReportingContractSatisfied"] = observation.reporting_contract_satisfied()
        report["childReportedOk"] = observation.child_reported_ok
        report["childReportedReadyForRecovery"] = observation.child_reported_ready
        report["childSignalObserved"] = observation.child_signal_observed
        for key, _ in _MILESTONE_MARKERS:
            report[key] = observation.milestones.get(key, False)

        report["diagnosticClassification"] = _classify_diagnostic(
            observation=observation,
            child_exited_normally=exited_normally,
            child_terminated_by_signal=terminated_by_signal,
            child_timed_out=bool(report["childTimedOut"]),
            return_code=return_code,
        )
        report["probeCompleted"] = True
        report["ok"] = report["diagnosticClassification"] == "child_completed_preflight_successfully"
    except Exception:
        logger.exception("BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_FAILED jobId=%s", job_id)
        report["probeCompleted"] = True
        report["diagnosticClassification"] = "insufficient_evidence"
    return report


def sanitize_process_probe_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key in _REPORT_SAFE_KEYS:
        if key in report:
            safe[key] = report[key]
    return safe


def print_process_probe_report(report: Dict[str, Any]) -> None:
    payload = sanitize_process_probe_report(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    job_id = _env("BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_JOB_ID")
    if not job_id:
        print_process_probe_report(
            {
                "jobId": "",
                "ok": False,
                "probeCompleted": True,
                "diagnosticClassification": "insufficient_evidence",
                "failureReason": "builder2_media_finalization_process_probe_job_id_missing",
            }
        )
        return 1

    logger.info("BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_START jobId=%s", job_id)
    report = run_media_finalization_process_probe(job_id)
    print_process_probe_report(report)
    logger.info(
        "BUILDER2_MEDIA_FINALIZATION_PROCESS_PROBE_DONE jobId=%s ok=%s classification=%s",
        job_id,
        report.get("ok"),
        report.get("diagnosticClassification"),
    )
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
