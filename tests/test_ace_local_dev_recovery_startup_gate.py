"""Recovery startup gate (phase 5.5) — check/report only."""
from __future__ import annotations

import json
import os
import unittest
import uuid

from engine.ace_local_dev_recovery_startup_coordinator import COORDINATOR_START_ALLOWED, COORDINATOR_START_BLOCKED
from engine.ace_local_dev_recovery_startup_gate import (
    evaluate_startup_recovery_gate,
    startup_decision_reports_dir,
)
from engine.ace_local_dev_startup_recovery_check import recovery_ledger_path
from tests.test_ace_local_dev_startup_recovery_check import StartupRecoveryTempLayout, _session_id


class TestRecoveryStartupGate(unittest.TestCase):
    def test_start_allowed_writes_report_no_ledger_change(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            before = None
            if recovery_ledger_path(layout.ace_root).is_file():
                before = recovery_ledger_path(layout.ace_root).read_bytes()
            payload = evaluate_startup_recovery_gate(layout.ace_root, acquire_coordinator_lock=False)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["coordinatorOutcome"], COORDINATOR_START_ALLOWED)
            self.assertTrue(payload["assignNewLauncherSession"])
            latest = startup_decision_reports_dir(layout.ace_root) / "recovery-startup-decision-latest.json"
            self.assertTrue(latest.is_file())
            report = json.loads(latest.read_text(encoding="utf-8"))
            self.assertTrue(report["checkOnly"])
            self.assertFalse(report["plannedFutureActions"]["autoRestoreEngine"]["executedInPhase55"])
            after = None
            if recovery_ledger_path(layout.ace_root).is_file():
                after = recovery_ledger_path(layout.ace_root).read_bytes()
            self.assertEqual(before, after)
        finally:
            layout.close()

    def test_recovery_needed_report_without_restore_engine(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260101-120000", clean=False)
            payload = evaluate_startup_recovery_gate(layout.ace_root, acquire_coordinator_lock=False)
            self.assertTrue(payload["ok"])
            self.assertNotEqual(payload["coordinatorOutcome"], COORDINATOR_START_ALLOWED)
            report = payload["report"]
            self.assertFalse(report["plannedFutureActions"]["autoRestoreEngine"]["executedInPhase55"])
            self.assertFalse(report["plannedFutureActions"]["autoRestoreEngine"]["wouldRun"] is None)
        finally:
            layout.close()

    def test_start_blocked_on_concurrent_lock(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260101-120000", clean=True)
            layout.write_managed("20260101-120000")
            from engine.ace_local_dev_recovery_startup_coordinator import coordinator_lock_path

            lock_path = coordinator_lock_path(layout.ace_root)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps({"pid": os.getpid(), "startedAt": "2026-01-01T00:00:00+00:00", "token": "x"}),
                encoding="utf-8",
            )
            second = evaluate_startup_recovery_gate(layout.ace_root, acquire_coordinator_lock=True)
            self.assertEqual(second["coordinatorOutcome"], COORDINATOR_START_BLOCKED)
            self.assertFalse(second["continueLocalStackStart"])
        finally:
            layout.close()

    def test_publish_unknown_preserves_evidence_in_report(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260101-120000", clean=False)
            sid = _session_id()
            op = uuid.uuid4().hex
            from engine.ace_local_dev_publish_execute_progress import PublishExecuteProgressWriter, STEP_GITHUB_PAGES
            from engine.ace_local_dev_publish_external_effects_journal import (
                PublishExternalEffectsJournalWriter,
                STAGE_GITHUB_PAGES,
                STAGE_UNKNOWN,
            )

            PublishExecuteProgressWriter(
                session_id=sid, operation_id=op, sessions_root=layout.sessions_root
            ).begin_step(STEP_GITHUB_PAGES)
            PublishExternalEffectsJournalWriter(
                session_id=sid,
                operation_id=op,
                sessions_root=layout.sessions_root,
            ).set_stage(STAGE_GITHUB_PAGES, STAGE_UNKNOWN, error="x")
            payload = evaluate_startup_recovery_gate(
                layout.ace_root,
                sessions_root=layout.sessions_root,
                acquire_coordinator_lock=False,
            )
            decision = payload["report"]["coordinatorDecision"]
            self.assertTrue(decision.get("isolatedOperations"))
        finally:
            layout.close()

    def test_start_script_wired_gate_not_restore_engine(self) -> None:
        start_path = __import__("pathlib").Path(__file__).resolve().parent.parent / "scripts" / "Start-AceLocal.ps1"
        text = start_path.read_text(encoding="utf-8")
        self.assertIn("AceLocalRecoveryStartupGateLib", text)
        self.assertIn("Invoke-AceLocalRecoveryStartupGateCheck", text)
        self.assertNotIn("run_conditional_auto_recovery", text)
        self.assertNotIn("auto_restore_engine", text)


if __name__ == "__main__":
    unittest.main()
