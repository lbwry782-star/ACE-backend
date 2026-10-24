"""Termination type classification tests (temp dirs; injected evidence only)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.ace_local_dev_cancel_progress import CancelExecuteProgressWriter, STEP_BACKEND_RESTORE
from engine.ace_local_dev_publish_execute_progress import PublishExecuteProgressWriter, STEP_GIT_PUSH_BACKEND
from engine.ace_local_dev_startup_recovery_check import (
    STARTUP_CRITICAL_RECOVERY,
    STARTUP_FIRST_START,
    STARTUP_NORMAL_START,
    STARTUP_RECOVERY_REQUIRED,
    STARTUP_RECOVERY_REVIEW,
    evaluate_previous_local_startup,
    local_dev_startup_gate_readonly_paths,
    read_recovery_ledger,
)
from engine.ace_local_dev_termination_classification import (
    EXIT_INTENTIONAL,
    EXIT_NEVER_STARTED,
    EXIT_PID_REUSED,
    TERMINATION_APP_CRASH,
    TERMINATION_NONE,
    TERMINATION_OS_CRASH,
    TERMINATION_POWER_LOSS,
    TERMINATION_QUIT,
    TERMINATION_UNKNOWN,
    TERMINATION_UNKNOWN_UNEXPECTED,
    classify_local_termination,
)


def _session_id() -> str:
    return f"ori-20260101-120000-{uuid.uuid4().hex[:8]}"


class TerminationTestLayout:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ace-termination-")
        self.ace_root = Path(self._tmp.name) / "ACE"
        self.local_dev = self.ace_root / "_backups" / "local-dev"
        self.sessions_root = self.ace_root / "_backups" / "sessions"
        self.local_dev.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._tmp.cleanup()

    def write_ledger(self, launcher: str, *, clean: bool = False, corrupt: bool = False) -> None:
        path = self.local_dev / "recovery-session-ledger.json"
        if corrupt:
            path.write_text("{bad", encoding="utf-8")
            return
        payload = {
            "schemaVersion": 1,
            "launcherSession": launcher,
            "sessionStartedAt": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "lastDirtyScanAt": None,
            "lastDirtyFingerprint": None,
            "lastVerifiedCheckpoint": None,
            "cleanShutdown": {
                "completed": clean,
                "completedAt": datetime.now(timezone.utc).isoformat() if clean else None,
                "launcherSession": launcher if clean else None,
            },
            "openIncident": None,
            "timingState": {},
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def write_managed(
        self,
        launcher: str,
        components: dict | None = None,
    ) -> None:
        payload = {
            "schemaVersion": 2,
            "launcherSession": launcher,
            "components": components
            or {"backend": None, "worker": None, "frontend": None, "aceUndo": None},
        }
        (self.local_dev / "managed-processes.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )

    def write_evidence_file(self, payload: dict) -> None:
        (self.local_dev / "termination-evidence.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


def _windows_power() -> dict:
    return {
        "windows": {
            "eventLogAvailable": True,
            "unexpectedShutdown": True,
            "eventId41": True,
            "lastBootTimeUtc": datetime.now(timezone.utc).isoformat(),
        }
    }


def _windows_os_crash() -> dict:
    return {
        "windows": {
            "eventLogAvailable": True,
            "bugcheckOrKernelCrash": True,
            "eventId1001": True,
            "lastBootTimeUtc": datetime.now(timezone.utc).isoformat(),
        }
    }


def _app_crash_evidence(component: str) -> dict:
    return {
        "windows": {
            "eventLogAvailable": True,
            "osContinuedThroughIncident": True,
            "lastBootTimeUtc": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        },
        "components": [{"component": component, "exitKind": "crashed", "detail": "simulated"}],
    }


class TestTerminationClassification(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = TerminationTestLayout()
        self.launcher = "20260101-120000"

    def tearDown(self) -> None:
        self.layout.close()

    def _eval(self, **kwargs):
        return evaluate_previous_local_startup(
            self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
            inspect_live_processes=False,
            **kwargs,
        )

    def test_quit_normal_start(self) -> None:
        self.layout.write_ledger(self.launcher, clean=True)
        self.layout.write_managed(self.launcher)
        d = self._eval()
        self.assertEqual(d.outcome, STARTUP_NORMAL_START)
        self.assertEqual(d.termination_type, TERMINATION_QUIT)
        self.assertEqual(d.confidence, "HIGH")

    def test_first_start_termination_none(self) -> None:
        d = self._eval()
        self.assertEqual(d.outcome, STARTUP_FIRST_START)
        self.assertEqual(d.termination_type, TERMINATION_NONE)

    def test_simulated_power_loss(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        self.layout.write_managed(self.launcher)
        d = self._eval(termination_evidence_override=_windows_power())
        self.assertEqual(d.termination_type, TERMINATION_POWER_LOSS)
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REQUIRED)

    def test_simulated_os_crash(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        d = self._eval(termination_evidence_override=_windows_os_crash())
        self.assertEqual(d.termination_type, TERMINATION_OS_CRASH)

    def test_app_crash_backend(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        self.layout.write_managed(
            self.launcher,
            {
                "backend": {"pid": 99999, "rootPid": 99999, "launcherSession": self.launcher, "startedByStack": True},
                "worker": None,
                "frontend": None,
            },
        )
        d = self._eval(termination_evidence_override=_app_crash_evidence("backend"))
        self.assertEqual(d.termination_type, TERMINATION_APP_CRASH)
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REQUIRED)

    def test_app_crash_worker(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        self.layout.write_managed(self.launcher)
        d = self._eval(termination_evidence_override=_app_crash_evidence("worker"))
        self.assertEqual(d.termination_type, TERMINATION_APP_CRASH)

    def _ev(self, **kwargs):
        return self._eval(**kwargs)

    def test_app_crash_ace_undo(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        d = self._eval(termination_evidence_override=_app_crash_evidence("aceUndo"))
        self.assertEqual(d.termination_type, TERMINATION_APP_CRASH)

    def test_frontend_intentional_stop(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        evidence = {
            "windows": {"osContinuedThroughIncident": True},
            "components": [{"component": "frontend", "exitKind": EXIT_INTENTIONAL, "detail": "user_closed_tab"}],
        }
        d = self._eval(termination_evidence_override=evidence)
        self.assertNotEqual(d.termination_type, TERMINATION_APP_CRASH)
        findings = {f["component"]: f["exitKind"] for f in d.component_findings}
        self.assertEqual(findings.get("frontend"), EXIT_INTENTIONAL)

    def test_pid_reuse_detection(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        self.layout.write_managed(
            self.launcher,
            {
                "backend": {
                    "pid": 4242,
                    "rootPid": 4242,
                    "launcherSession": self.launcher,
                    "startedByStack": True,
                    "startedAt": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                }
            },
        )

        def inspector(pid: int) -> dict:
            return {
                "alive": True,
                "processCreatedAtUtc": datetime.now(timezone.utc).isoformat(),
            }

        d = self._eval(process_inspector=inspector)
        kinds = {f["component"]: f["exitKind"] for f in d.component_findings}
        self.assertEqual(kinds.get("backend"), EXIT_PID_REUSED)

    def test_component_never_started(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        self.layout.write_managed(self.launcher, {"backend": None, "worker": None, "frontend": None})
        d = self._eval()
        never = [f for f in d.component_findings if f.get("exitKind") == EXIT_NEVER_STARTED]
        self.assertTrue(len(never) >= 1)

    def test_windows_up_during_app_crash(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        d = self._eval(termination_evidence_override=_app_crash_evidence("backend"))
        codes = {e["code"] for e in d.evidence}
        self.assertIn("app_crash_while_os_up", codes)

    def test_missing_windows_events(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        d = self._eval(termination_evidence_override={"windows": {"eventLogMissing": True}})
        self.assertIn(d.termination_type, (TERMINATION_UNKNOWN_UNEXPECTED, TERMINATION_UNKNOWN))

    def test_conflicting_power_and_os(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        evidence = {
            "windows": {
                "unexpectedShutdown": True,
                "bugcheckOrKernelCrash": True,
                "eventLogAvailable": True,
            }
        }
        d = self._eval(termination_evidence_override=evidence)
        self.assertEqual(d.termination_type, TERMINATION_UNKNOWN_UNEXPECTED)
        self.assertEqual(d.confidence, "LOW")

    def test_cancel_during_crash_critical(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        sid = _session_id()
        w = CancelExecuteProgressWriter(
            session_id=sid, operation_id=uuid.uuid4().hex, sessions_root=self.layout.sessions_root
        )
        w.begin_step(STEP_BACKEND_RESTORE)
        d = self._eval(termination_evidence_override=_app_crash_evidence("backend"))
        self.assertEqual(d.outcome, STARTUP_CRITICAL_RECOVERY)
        self.assertEqual(d.termination_type, TERMINATION_APP_CRASH)

    def test_publish_during_crash_critical(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        sid = _session_id()
        ex = PublishExecuteProgressWriter(
            session_id=sid, operation_id=uuid.uuid4().hex, sessions_root=self.layout.sessions_root
        )
        ex.begin_step(STEP_GIT_PUSH_BACKEND)
        d = self._eval(termination_evidence_override=_app_crash_evidence("worker"))
        self.assertEqual(d.outcome, STARTUP_CRITICAL_RECOVERY)

    def test_corrupt_ledger_review(self) -> None:
        self.layout.write_ledger(self.launcher, corrupt=True)
        d = self._eval()
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REVIEW)

    def test_evidence_files_not_mutated(self) -> None:
        self.layout.write_ledger(self.launcher, clean=False)
        self.layout.write_evidence_file(_windows_power())
        ld = self.layout.local_dev
        before = hashlib.sha256((ld / "termination-evidence.json").read_bytes()).hexdigest()
        ledger_before = (ld / "recovery-session-ledger.json").read_bytes()
        _ = self._eval()
        after = hashlib.sha256((ld / "termination-evidence.json").read_bytes()).hexdigest()
        ledger_after = (ld / "recovery-session-ledger.json").read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(ledger_before, ledger_after)

    def test_recovery_status_separate_from_termination(self) -> None:
        self.layout.write_ledger(self.launcher, clean=True)
        self.layout.write_managed(self.launcher)
        d = self._eval()
        self.assertEqual(d.recovery_status, STARTUP_NORMAL_START)
        self.assertEqual(d.termination_type, TERMINATION_QUIT)
        self.assertNotEqual(d.recovery_status, d.termination_type)

    def test_classify_unit_unknown_unexpected(self) -> None:
        r = classify_local_termination(
            launcher_session="s",
            session_started_at=datetime.now(timezone.utc).isoformat(),
            clean_shutdown_verified=False,
            registry=None,
            termination_evidence={"windows": {"unexpectedShutdown": True, "bugcheckOrKernelCrash": True}},
            first_start=False,
        )
        self.assertEqual(r.termination_type, TERMINATION_UNKNOWN_UNEXPECTED)


if __name__ == "__main__":
    unittest.main()
