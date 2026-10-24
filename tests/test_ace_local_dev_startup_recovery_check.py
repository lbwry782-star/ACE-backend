"""Isolated startup recovery check tests (temp dirs only; no live ACE)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.ace_local_dev_cancel_progress import (
    CancelExecuteProgressWriter,
    STEP_BACKEND_RESTORE,
)
from engine.ace_local_dev_publish_execute_progress import (
    PublishExecuteProgressWriter,
    STEP_GIT_PUSH_BACKEND,
    STEP_GITHUB_PAGES,
)
from engine.ace_local_dev_publish_external_effects_journal import (
    PublishExternalEffectsJournalWriter,
    STAGE_GITHUB_PAGES,
    STAGE_RUNNING,
    STAGE_UNKNOWN,
)
from engine.ace_local_dev_shutdown_gate import (
    GATE_SHUTDOWN_REQUESTED,
    _write_json_atomic,
    shutdown_gate_state_path,
)
from engine.ace_local_dev_startup_recovery_check import (
    STARTUP_CRITICAL_RECOVERY,
    STARTUP_FIRST_START,
    STARTUP_NORMAL_START,
    STARTUP_RECOVERY_REQUIRED,
    STARTUP_RECOVERY_REVIEW,
    assert_dev_operations_allowed,
    evaluate_previous_local_startup,
    local_dev_startup_gate_readonly_paths,
    read_recovery_ledger,
    recovery_ledger_path,
)


def _session_id() -> str:
    return f"ori-20260101-120000-{uuid.uuid4().hex[:8]}"


def _hash_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()


class StartupRecoveryTempLayout:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ace-startup-recovery-")
        self.ace_root = Path(self._tmp.name) / "ACE"
        self.sessions_root = self.ace_root / "_backups" / "sessions"
        self.local_dev = self.ace_root / "_backups" / "local-dev"
        self.local_dev.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._tmp.cleanup()

    def session_dir(self, session_id: str) -> Path:
        path = self.sessions_root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_ledger(
        self,
        launcher: str,
        *,
        clean: bool = False,
        open_incident: bool = False,
        corrupt: bool = False,
    ) -> None:
        path = recovery_ledger_path(self.ace_root)
        if corrupt:
            path.write_text("{broken", encoding="utf-8")
            return
        payload: dict = {
            "schemaVersion": 1,
            "launcherSession": launcher,
            "sessionStartedAt": datetime.now(timezone.utc).isoformat(),
            "lastDirtyScanAt": None,
            "lastDirtyFingerprint": None,
            "lastVerifiedCheckpoint": None,
            "cleanShutdown": {
                "completed": clean,
                "completedAt": datetime.now(timezone.utc).isoformat() if clean else None,
                "launcherSession": launcher if clean else None,
            },
            "openIncident": None,
            "timingState": {"lastLightCheckpointDueAt": None, "lastFullCheckpointDueAt": None},
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        if open_incident:
            payload["openIncident"] = {
                "kind": "test_incident",
                "openedAt": datetime.now(timezone.utc).isoformat(),
                "detail": "phase2 test",
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def write_managed(self, launcher: str, *, backend_pid: int = 0) -> None:
        comp = None
        if backend_pid > 0:
            comp = {
                "pid": backend_pid,
                "rootPid": backend_pid,
                "startedByStack": True,
            }
        payload = {
            "schemaVersion": 2,
            "launcherSession": launcher,
            "components": {"backend": comp, "worker": None, "frontend": None},
        }
        (self.local_dev / "managed-processes.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )

    def evaluate(self, **kwargs):
        return evaluate_previous_local_startup(
            self.ace_root,
            sessions_root=self.sessions_root,
            **kwargs,
        )


class TestAceLocalStartupRecoveryCheck(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = StartupRecoveryTempLayout()

    def tearDown(self) -> None:
        self.layout.close()

    def test_first_start(self) -> None:
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_FIRST_START)
        self.assertTrue(d.dev_operations_allowed)

    def test_normal_start_clean_shutdown(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=True)
        self.layout.write_managed("20260101-120000")
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_NORMAL_START)
        self.assertTrue(d.dev_operations_allowed)

    def test_clean_shutdown_not_verified(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        self.layout.write_managed("20260101-120000")
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REQUIRED)
        self.assertFalse(d.dev_operations_allowed)
        self.assertIn("clean_shutdown_not_verified", d.reasons)

    def test_simulated_power_loss(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        d = self.layout.evaluate(inspect_live_processes=False)
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REQUIRED)

    def test_ace_undo_only_live_backend_after_clean_shutdown_review(self) -> None:
        launcher = "20260101-120000"
        self.layout.write_ledger(launcher, clean=True)
        self.layout.write_managed(launcher, backend_pid=os.getpid())
        d = self.layout.evaluate(inspect_live_processes=True)
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REVIEW)
        self.assertIn("live_processes_after_verified_shutdown", d.reasons)

    def test_backend_crash_no_clean_shutdown(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        self.layout.write_managed("20260101-120000", backend_pid=999999)
        d = self.layout.evaluate(inspect_live_processes=True)
        self.assertIn(
            d.outcome,
            (STARTUP_RECOVERY_REQUIRED, STARTUP_RECOVERY_REVIEW),
        )
        self.assertFalse(d.dev_operations_allowed)

    def test_cancel_mid_backend_restore_critical(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        w.begin_step(STEP_BACKEND_RESTORE)
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_CRITICAL_RECOVERY)
        self.assertFalse(d.dev_operations_allowed)

    def test_publish_mid_git_push_critical(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        ex.begin_step(STEP_GIT_PUSH_BACKEND)
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_CRITICAL_RECOVERY)

    def test_publish_unknown_critical(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        ex.begin_step(STEP_GITHUB_PAGES)
        j = PublishExternalEffectsJournalWriter(
            session_id=sid,
            operation_id=op,
            sessions_root=self.layout.sessions_root,
        )
        j.set_stage(STAGE_GITHUB_PAGES, STAGE_UNKNOWN, error="deploy_result_unknown")
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_CRITICAL_RECOVERY)

    def test_corrupt_ledger_review(self) -> None:
        self.layout.write_ledger("20260101-120000", corrupt=True)
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REVIEW)
        self.assertFalse(d.dev_operations_allowed)

    def test_stale_progress_after_clean_shutdown_normal(self) -> None:
        launcher = "20260101-120000"
        self.layout.write_ledger(launcher, clean=True)
        self.layout.write_managed(launcher)
        sid = _session_id()
        op = uuid.uuid4().hex
        session_dir = self.layout.session_dir(sid)
        path = session_dir / f"cancel-execute-progress-{op}.json"
        path.write_text("{not-json", encoding="utf-8")
        old = datetime.now(timezone.utc) - timedelta(days=3)
        ts = old.timestamp()
        os.utime(path, (ts, ts))
        ledger, _ = read_recovery_ledger(self.layout.ace_root)
        assert ledger is not None
        cs = ledger["cleanShutdown"]
        cs["completedAt"] = datetime.now(timezone.utc).isoformat()
        recovery_ledger_path(self.layout.ace_root).write_text(
            json.dumps(ledger, indent=2) + "\n",
            encoding="utf-8",
        )
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_NORMAL_START)

    def test_launcher_session_mismatch_review(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=True)
        self.layout.write_managed("20260101-999999")
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REVIEW)

    def test_open_incident_required(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False, open_incident=True)
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REQUIRED)

    def test_dual_recovery_signals_incident_and_gate(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False, open_incident=True)
        _write_json_atomic(
            shutdown_gate_state_path(self.layout.ace_root),
            {
                "schemaVersion": 1,
                "gateStatus": GATE_SHUTDOWN_REQUESTED,
                "launcherSession": "20260101-120000",
                "requestedAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        )
        d = self.layout.evaluate()
        self.assertFalse(d.dev_operations_allowed)
        self.assertIn(d.outcome, (STARTUP_RECOVERY_REQUIRED, STARTUP_RECOVERY_REVIEW))

    def test_shutdown_started_not_completed_review(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        _write_json_atomic(
            shutdown_gate_state_path(self.layout.ace_root),
            {
                "schemaVersion": 1,
                "gateStatus": GATE_SHUTDOWN_REQUESTED,
                "launcherSession": "20260101-120000",
                "requestedAt": datetime.now(timezone.utc).isoformat(),
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        )
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_RECOVERY_REVIEW)

    def test_evaluate_does_not_mutate_state_files(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        w.begin_step(STEP_BACKEND_RESTORE)
        before = _hash_paths(local_dev_startup_gate_readonly_paths(self.layout.ace_root))
        session_files = list(self.layout.sessions_root.rglob("*"))
        session_hash_before = _hash_paths([p for p in session_files if p.is_file()])
        _ = self.layout.evaluate()
        after = _hash_paths(local_dev_startup_gate_readonly_paths(self.layout.ace_root))
        session_hash_after = _hash_paths([p for p in session_files if p.is_file()])
        self.assertEqual(before, after)
        self.assertEqual(session_hash_before, session_hash_after)

    def test_decision_before_new_ledger_init(self) -> None:
        self.layout.write_ledger("20260101-111111", clean=False)
        d = self.layout.evaluate()
        self.assertFalse(d.dev_operations_allowed)
        with self.assertRaises(RuntimeError):
            assert_dev_operations_allowed(d)
        ledger_before, _ = read_recovery_ledger(self.layout.ace_root)
        assert ledger_before is not None
        self.assertEqual(ledger_before.get("launcherSession"), "20260101-111111")

    def test_start_ace_local_not_wired(self) -> None:
        start_path = Path(__file__).resolve().parent.parent / "scripts" / "Start-AceLocal.ps1"
        text = start_path.read_text(encoding="utf-8")
        self.assertNotIn("AceLocalStartupRecoveryCheck", text)
        self.assertNotIn("startup_recovery_check", text)

    def test_publish_git_push_journal_running_critical(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        ex.begin_step(STEP_GIT_PUSH_BACKEND)
        j = PublishExternalEffectsJournalWriter(
            session_id=sid,
            operation_id=op,
            candidate_backend_sha="a" * 40,
            sessions_root=self.layout.sessions_root,
        )
        j.set_stage("git_push_backend", STAGE_RUNNING, evidence={"pushed": False})
        d = self.layout.evaluate()
        self.assertEqual(d.outcome, STARTUP_CRITICAL_RECOVERY)


if __name__ == "__main__":
    unittest.main()
