"""Pre-checkpoint write gate (phase 5.7)."""
from __future__ import annotations

import json
import os
import unittest

from engine.ace_local_dev_recovery_periodic_save_pre_write import (
    activation_snapshot_path,
    evaluate_coordinator_periodic_save_gate,
    evaluate_ledger_launcher_session_match,
    evaluate_pre_checkpoint_write_gate,
)
from engine.ace_local_dev_recovery_startup_coordinator import COORDINATOR_START_ALLOWED
from engine.ace_local_dev_recovery_startup_gate import (
    LATEST_REPORT_FILENAME,
    startup_decision_reports_dir,
)
from tests.test_ace_local_dev_startup_recovery_check import StartupRecoveryTempLayout


class TestPeriodicSavePreWrite(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE"] = "ok"

    def tearDown(self) -> None:
        os.environ.pop("ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE", None)

    def test_coordinator_snapshot_allows(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260201-120000", clean=True)
            snap = activation_snapshot_path(layout.ace_root)
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(
                json.dumps(
                    {
                        "launcherSession": "20260201-120000",
                        "coordinatorOutcome": COORDINATOR_START_ALLOWED,
                        "allowPeriodicSaveService": True,
                    }
                ),
                encoding="utf-8",
            )
            gate = evaluate_coordinator_periodic_save_gate(layout.ace_root, "20260201-120000")
            self.assertTrue(gate["allowed"])
        finally:
            layout.close()

    def test_ledger_session_mismatch_blocks(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260201-120000", clean=True)
            gate = evaluate_ledger_launcher_session_match(layout.ace_root, "20260201-999999")
            self.assertFalse(gate["allowed"])
        finally:
            layout.close()

    def test_reassess_picks_up_guard_file_block(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260201-120000", clean=True)
            snap = activation_snapshot_path(layout.ace_root)
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(
                json.dumps(
                    {
                        "launcherSession": "20260201-120000",
                        "coordinatorOutcome": COORDINATOR_START_ALLOWED,
                        "allowPeriodicSaveService": True,
                    }
                ),
                encoding="utf-8",
            )
            first = evaluate_pre_checkpoint_write_gate(layout.ace_root, "20260201-120000")
            self.assertTrue(first["allowed"])
            guard = snap.parent / "recovery-periodic-save-guard.json"
            guard.write_text(
                json.dumps({"blockCheckpoint": True, "reason": "critical_mid_write"}),
                encoding="utf-8",
            )
            second = evaluate_pre_checkpoint_write_gate(
                layout.ace_root,
                "20260201-120000",
                live_assessment_only=True,
                prior_assessment_at_iso=str(first["assessedAt"]),
            )
            self.assertFalse(second["allowed"])
        finally:
            layout.close()

    def test_background_loop_does_not_skip_live(self) -> None:
        loop_path = (
            __import__("pathlib").Path(__file__).resolve().parent.parent
            / "scripts"
            / "AceLocalRecoveryPeriodicSaveLib.ps1"
        )
        text = loop_path.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"SkipLiveAssessment:\$BackgroundMode")


if __name__ == "__main__":
    unittest.main()
