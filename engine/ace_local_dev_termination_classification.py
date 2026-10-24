"""
Read-only classification of how the previous LOCAL session terminated.

terminationType is inferred from evidence — never presented as certain fact.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TERMINATION_QUIT = "QUIT"
TERMINATION_POWER_LOSS = "POWER_LOSS"
TERMINATION_OS_CRASH = "OS_CRASH"
TERMINATION_APP_CRASH = "APP_CRASH"
TERMINATION_UNKNOWN_UNEXPECTED = "UNKNOWN_UNEXPECTED_TERMINATION"
TERMINATION_UNKNOWN = "UNKNOWN"
TERMINATION_NONE = "NONE"

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

COMPONENT_NAMES = ("backend", "worker", "frontend", "aceUndo", "memuraiWatch", "progressUi")

EXIT_CRASHED = "crashed"
EXIT_INTENTIONAL = "intentional_stop"
EXIT_NEVER_STARTED = "never_started"
EXIT_PID_REUSED = "pid_reused"
EXIT_REPLACED = "replaced_process"
EXIT_STILL_RUNNING = "still_running"
EXIT_UNKNOWN = "unknown"

TERMINATION_EVIDENCE_FILENAME = "termination-evidence.json"

ProcessInspector = Callable[[int], Dict[str, Any]]


def _parse_iso_utc(text: Optional[str]) -> Optional[datetime]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def default_process_inspector(pid: int) -> Dict[str, Any]:
    alive = _pid_alive(pid)
    return {
        "alive": alive,
        "processCreatedAtUtc": None,
        "imageName": None,
    }


def try_collect_windows_termination_evidence_readonly() -> Dict[str, Any]:
    """
    Best-effort Windows signals without Administrator.
    Real collection can be extended later; startup check remains read-only.
    """
    return {
        "eventLogAvailable": False,
        "eventLogMissing": True,
        "note": "windows_event_collection_deferred",
    }


def read_termination_evidence_file(local_dev_dir: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    path = local_dev_dir / TERMINATION_EVIDENCE_FILENAME
    if not path.is_file():
        return None, False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, True
    return (data if isinstance(data, dict) else None), False


def _evidence(code: str, detail: str, *, weight: str = "support") -> Dict[str, Any]:
    return {"code": code, "detail": detail[:480], "weight": weight}


@dataclass
class TerminationClassification:
    termination_type: str
    confidence: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    component_findings: List[Dict[str, Any]] = field(default_factory=list)


def _windows_signals(windows: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(windows, dict):
        return {
            "available": False,
            "unexpected_shutdown": False,
            "bugcheck_or_os_crash": False,
            "boot_time_utc": None,
            "os_continued_through_incident": None,
            "application_crash_events": [],
            "missing_event_log": False,
        }
    boot = _parse_iso_utc(str(windows.get("lastBootTimeUtc") or windows.get("bootTimeUtc") or ""))
    unexpected = bool(windows.get("unexpectedShutdown") or windows.get("eventId41"))
    bugcheck = bool(windows.get("bugcheckOrKernelCrash") or windows.get("eventId1001"))
    os_continued = windows.get("osContinuedThroughIncident")
    if os_continued is None and boot is not None:
        os_continued = True
    return {
        "available": bool(windows.get("eventLogAvailable", True)),
        "unexpected_shutdown": unexpected,
        "bugcheck_or_os_crash": bugcheck,
        "boot_time_utc": boot,
        "os_continued_through_incident": os_continued,
        "application_crash_events": list(windows.get("applicationCrashEvents") or []),
        "missing_event_log": bool(windows.get("eventLogMissing")),
    }


def _merge_component_evidence(
    registry: Optional[Dict[str, Any]],
    bundle: Optional[Dict[str, Any]],
    *,
    launcher_session: str,
    clean_shutdown: bool,
    inspector: ProcessInspector,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}
    if isinstance(bundle, dict):
        for row in bundle.get("components") or []:
            if isinstance(row, dict) and row.get("component"):
                by_name[str(row["component"])] = row

    components = {}
    if isinstance(registry, dict):
        raw = registry.get("components")
        if isinstance(raw, dict):
            components = raw

    names = set(COMPONENT_NAMES) | set(components.keys()) | set(by_name.keys())
    for name in sorted(names):
        reg_row = components.get(name) if isinstance(components.get(name), dict) else None
        inj = by_name.get(name)
        finding: Dict[str, Any] = {"component": name, "exitKind": EXIT_UNKNOWN, "notes": []}

        if inj and inj.get("exitKind"):
            finding["exitKind"] = str(inj.get("exitKind"))
            finding["notes"].append("termination_evidence_file")
            if inj.get("detail"):
                finding["notes"].append(str(inj.get("detail"))[:120])
            findings.append(finding)
            continue

        if not reg_row:
            finding["exitKind"] = EXIT_NEVER_STARTED
            finding["notes"].append("not_in_registry")
            findings.append(finding)
            continue

        pid = reg_row.get("pid") or reg_row.get("rootPid")
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            finding["exitKind"] = EXIT_NEVER_STARTED
            finding["notes"].append("invalid_pid")
            findings.append(finding)
            continue

        bound = str(reg_row.get("launcherSession") or "").strip()
        if launcher_session and bound and bound != launcher_session:
            finding["exitKind"] = EXIT_UNKNOWN
            finding["notes"].append("launcher_session_mismatch")
            findings.append(finding)
            continue

        insp = inspector(pid_int)
        alive = bool(insp.get("alive"))
        reg_started = _parse_iso_utc(str(reg_row.get("startedAt") or ""))
        proc_created = _parse_iso_utc(str(insp.get("processCreatedAtUtc") or ""))
        if alive and proc_created and reg_started and proc_created > reg_started:
            finding["exitKind"] = EXIT_PID_REUSED
            finding["notes"].append("pid_number_reused_new_process")
            findings.append(finding)
            continue

        if alive:
            if clean_shutdown:
                finding["exitKind"] = EXIT_STILL_RUNNING
                finding["notes"].append("alive_after_verified_quit")
            else:
                finding["exitKind"] = EXIT_STILL_RUNNING
                finding["notes"].append("process_still_running")
            findings.append(finding)
            continue

        started_by_stack = reg_row.get("startedByStack")
        if started_by_stack is False:
            finding["exitKind"] = EXIT_NEVER_STARTED
            finding["notes"].append("not_started_by_stack")
            findings.append(finding)
            continue

        if not clean_shutdown and started_by_stack is not False:
            finding["exitKind"] = EXIT_CRASHED
            finding["notes"].append("registered_pid_not_alive_no_clean_quit")
        else:
            finding["exitKind"] = EXIT_INTENTIONAL
            finding["notes"].append("pid_gone_expected")
        findings.append(finding)

    return findings


def classify_local_termination(
    *,
    launcher_session: str,
    session_started_at: Optional[str],
    clean_shutdown_verified: bool,
    registry: Optional[Dict[str, Any]],
    termination_evidence: Optional[Dict[str, Any]] = None,
    first_start: bool = False,
    inspector: Optional[ProcessInspector] = None,
) -> TerminationClassification:
    """
    Infer terminationType from read-only evidence. Does not mutate any state.
    """
    evidence: List[Dict[str, Any]] = []
    insp = inspector or default_process_inspector

    if first_start:
        return TerminationClassification(
            termination_type=TERMINATION_NONE,
            confidence=CONFIDENCE_HIGH,
            evidence=[_evidence("no_prior_session", "first_start")],
            component_findings=[],
        )

    if clean_shutdown_verified:
        evidence.append(_evidence("clean_shutdown_verified", f"launcherSession={launcher_session}", weight="primary"))
        return TerminationClassification(
            termination_type=TERMINATION_QUIT,
            confidence=CONFIDENCE_HIGH,
            evidence=evidence,
            component_findings=_merge_component_evidence(
                registry,
                termination_evidence,
                launcher_session=launcher_session,
                clean_shutdown=True,
                inspector=insp,
            ),
        )

    bundle = termination_evidence if isinstance(termination_evidence, dict) else {}
    windows = _windows_signals(bundle.get("windows") if isinstance(bundle.get("windows"), dict) else None)
    component_findings = _merge_component_evidence(
        registry,
        bundle,
        launcher_session=launcher_session,
        clean_shutdown=False,
        inspector=insp,
    )

    session_start = _parse_iso_utc(session_started_at)
    if session_start:
        evidence.append(_evidence("session_started_at", session_start.isoformat()))

    if windows["missing_event_log"]:
        evidence.append(_evidence("windows_event_log_missing", "no_admin_or_unavailable", weight="gap"))

    app_crash_components = [
        f
        for f in component_findings
        if f.get("exitKind") in (EXIT_CRASHED, EXIT_REPLACED)
        and str(f.get("component")) in ("backend", "worker", "frontend", "aceUndo", "memuraiWatch", "progressUi")
    ]
    crashed_named = [f for f in component_findings if f.get("exitKind") == EXIT_CRASHED]

    os_continued = windows.get("os_continued_through_incident")
    if os_continued is None and windows["boot_time_utc"] and session_start:
        os_continued = windows["boot_time_utc"] < session_start

    if os_continued is True:
        evidence.append(_evidence("os_continued", "boot_before_session_or_explicit_flag", weight="support"))

    power_signal = windows["unexpected_shutdown"]
    os_signal = windows["bugcheck_or_os_crash"]
    if power_signal:
        evidence.append(_evidence("windows_unexpected_shutdown", "inferred_power_loss_signal", weight="support"))
    if os_signal:
        evidence.append(_evidence("windows_bugcheck_or_kernel_crash", "inferred_os_crash_signal", weight="support"))

    for comp in app_crash_components:
        evidence.append(
            _evidence(
                "component_abnormal_exit",
                f"{comp.get('component')}:{comp.get('exitKind')}",
                weight="support",
            )
        )

    for evt in windows.get("application_crash_events") or []:
        if isinstance(evt, dict):
            evidence.append(
                _evidence(
                    "windows_application_crash_event",
                    f"{evt.get('source', 'app')}:{evt.get('eventId', '')}",
                    weight="support",
                )
            )

    if power_signal and os_signal:
        evidence.append(_evidence("conflicting_signals", "power_loss_and_os_crash", weight="conflict"))
        return TerminationClassification(
            termination_type=TERMINATION_UNKNOWN_UNEXPECTED,
            confidence=CONFIDENCE_LOW,
            evidence=evidence,
            component_findings=component_findings,
        )

    if crashed_named and os_continued is True and not power_signal and not os_signal:
        names = ",".join(str(c["component"]) for c in crashed_named)
        evidence.append(_evidence("app_crash_while_os_up", names, weight="primary"))
        return TerminationClassification(
            termination_type=TERMINATION_APP_CRASH,
            confidence=CONFIDENCE_HIGH if len(crashed_named) == 1 else CONFIDENCE_MEDIUM,
            evidence=evidence,
            component_findings=component_findings,
        )

    if crashed_named and os_continued is False:
        evidence.append(_evidence("component_down_with_os_reboot", "ambiguous_app_vs_os", weight="conflict"))
        return TerminationClassification(
            termination_type=TERMINATION_UNKNOWN_UNEXPECTED,
            confidence=CONFIDENCE_LOW,
            evidence=evidence,
            component_findings=component_findings,
        )

    if os_signal and not power_signal:
        return TerminationClassification(
            termination_type=TERMINATION_OS_CRASH,
            confidence=CONFIDENCE_MEDIUM if windows["available"] else CONFIDENCE_LOW,
            evidence=evidence,
            component_findings=component_findings,
        )

    if power_signal and not os_signal:
        return TerminationClassification(
            termination_type=TERMINATION_POWER_LOSS,
            confidence=CONFIDENCE_MEDIUM if windows["available"] else CONFIDENCE_LOW,
            evidence=evidence,
            component_findings=component_findings,
        )

    if crashed_named:
        return TerminationClassification(
            termination_type=TERMINATION_APP_CRASH,
            confidence=CONFIDENCE_LOW,
            evidence=evidence + [_evidence("limited_os_context", "app_exit_without_os_signals")],
            component_findings=component_findings,
        )

    if not windows["available"] and not bundle:
        evidence.append(_evidence("insufficient_evidence", "no_clean_shutdown_no_signals"))
        return TerminationClassification(
            termination_type=TERMINATION_UNKNOWN_UNEXPECTED,
            confidence=CONFIDENCE_LOW,
            evidence=evidence,
            component_findings=component_findings,
        )

    evidence.append(_evidence("insufficient_evidence", "unclassified_abnormal_end"))
    return TerminationClassification(
        termination_type=TERMINATION_UNKNOWN,
        confidence=CONFIDENCE_LOW,
        evidence=evidence,
        component_findings=component_findings,
    )


def termination_classification_to_dict(result: TerminationClassification) -> Dict[str, Any]:
    return {
        "terminationType": result.termination_type,
        "confidence": result.confidence,
        "evidence": list(result.evidence),
        "componentFindings": list(result.component_findings),
    }
