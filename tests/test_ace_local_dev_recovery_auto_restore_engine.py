"""Auto recovery engine tests — temp git workspaces only."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

from engine.ace_local_dev_cancel_progress import CancelExecuteProgressWriter, STEP_BACKEND_RESTORE
from engine.ace_local_dev_recovery_auto_restore_engine import (
    JOURNAL_STATUS_IN_PROGRESS,
    JOURNAL_STATUS_SKIPPED,
    identify_interrupted_restore,
    read_restore_journal,
    run_conditional_auto_recovery,
)
from engine.ace_local_dev_recovery_auto_restore_plan import build_auto_restore_plan
from engine.ace_local_dev_recovery_checkpoint_chain import build_verified_checkpoint_chain
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
    RecoveryStartupCoordinatorDecision,
    evaluate_recovery_startup_coordination,
)
from engine.ace_local_dev_startup_recovery_check import recovery_ledger_path
from tests.recovery_checkpoint_fixtures import write_full_checkpoint, write_light_checkpoint
from tests.test_ace_local_dev_recovery_startup_coordinator import CoordinatorTempLayout, _session_id


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "recovery52@ace.local"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Recovery52"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _decision(
    outcome: str,
    *,
    checkpoint_id: str = "",
    ace_assessment: dict | None = None,
) -> RecoveryStartupCoordinatorDecision:
    return RecoveryStartupCoordinatorDecision(
        coordinator_outcome=outcome,
        recovery_status=outcome,
        termination_type="POWER_LOSS",
        confidence="MEDIUM",
        evidence=[],
        previous_launcher_session="20260101-120000",
        last_verified_checkpoint={"checkpointId": checkpoint_id} if checkpoint_id else None,
        allow_local_processes=True,
        allow_dev_work=True,
        allow_new_launcher_session=True,
        allow_ledger_initialization=True,
        allow_periodic_save_service=True,
        checkpoint_assessment=ace_assessment
        or {"recommendedCheckpointId": checkpoint_id, "recommendedValidationOk": True},
        evidence_preservation={"observationEvidence": []},
    )


class TestAutoRestoreEngine(unittest.TestCase):
    def _layout(self) -> CoordinatorTempLayout:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        _init_git_repo(layout.ace_root / "ACE-Backend")
        _init_git_repo(layout.ace_root / "ACE-Frontend")
        return layout

    def test_start_allowed_skips_recovery(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            (be / "sample.txt").write_text("x\n", encoding="utf-8")
            before = (be / "sample.txt").read_text(encoding="utf-8")
            result = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_START_ALLOWED),
                backend_root=be,
                frontend_root=layout.ace_root / "ACE-Frontend",
            )
            self.assertTrue(result.skipped)
            self.assertEqual(result.skip_reason, "start_allowed_no_recovery")
            self.assertEqual((be / "sample.txt").read_text(encoding="utf-8"), before)
            self.assertIsNone(read_restore_journal(layout.ace_root))
        finally:
            layout.close()

    def test_full_only_missing_file_restored(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp-full-1"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "sample.txt", "ok\n")])
            layout.write_ledger("20260101-120000", clean=False)
            layout.write_ledger_checkpoint_ref("20260101-120000", cp, layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp)
            result = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertTrue(result.ran)
            self.assertIn("sample.txt", result.applied)
            self.assertEqual((be / "sample.txt").read_text(encoding="utf-8"), "ok\n")
        finally:
            layout.close()

    def test_full_light_chain_restore(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            full = "cp-full-2"
            light = "cp-light-2"
            write_full_checkpoint(layout.ace_root, full, [("BACKEND", "a.txt", "v1\n")])
            write_light_checkpoint(
                layout.ace_root,
                light,
                full,
                full,
                [("BACKEND", "a.txt", "v2\n")],
            )
            chain = build_verified_checkpoint_chain(layout.ace_root, light)
            self.assertTrue(chain.ok)
            self.assertEqual(len(chain.chain), 2)
            result = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=light),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertIn("a.txt", result.applied)
            self.assertEqual((be / "a.txt").read_text(encoding="utf-8"), "v2\n")
        finally:
            layout.close()

    def test_preserve_newer_workspace_no_overwrite(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp-conflict"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "a.txt", "old\n")])
            (be / "a.txt").write_text("user-new\n", encoding="utf-8")
            result = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual((be / "a.txt").read_text(encoding="utf-8"), "user-new\n")
            self.assertIn("a.txt", result.conflicts)
        finally:
            layout.close()

    def test_new_file_not_in_backup_untouched(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp-newfile"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "tracked.txt", "t\n")])
            (be / "brand-new.txt").write_text("only-local\n", encoding="utf-8")
            before = (be / "brand-new.txt").read_text(encoding="utf-8")
            run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual((be / "brand-new.txt").read_text(encoding="utf-8"), before)
        finally:
            layout.close()

    def test_broken_chain_fallback(self) -> None:
        layout = self._layout()
        try:
            good = "cp-good-chain"
            bad = "cp-bad-chain"
            write_full_checkpoint(layout.ace_root, good, [("BACKEND", "f.txt", "good\n")])
            bad_root = write_full_checkpoint(layout.ace_root, bad, [("BACKEND", "f.txt", "bad\n")])
            meta = json.loads((bad_root / "checkpoint-metadata.json").read_text(encoding="utf-8"))
            meta["baseCheckpointId"] = "missing-parent"
            (bad_root / "checkpoint-metadata.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
            plan = build_auto_restore_plan(
                ace_root=layout.ace_root,
                checkpoint_id=bad,
                backend_root=layout.ace_root / "ACE-Backend",
                frontend_root=layout.ace_root / "ACE-Frontend",
            )
            self.assertTrue(plan.checkpoint_valid)
            self.assertIn(good, plan.chain_ids)
        finally:
            layout.close()

    def test_crash_resume_completes_remaining(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp-resume"
            write_full_checkpoint(
                layout.ace_root,
                cp,
                [
                    ("BACKEND", "one.txt", "1\n"),
                    ("BACKEND", "two.txt", "2\n"),
                ],
            )
            (be / "one.txt").unlink(missing_ok=True)
            (be / "two.txt").unlink(missing_ok=True)
            first = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
                simulate_crash_after=1,
            )
            self.assertEqual(first.journal_status, JOURNAL_STATUS_IN_PROGRESS)
            info = identify_interrupted_restore(layout.ace_root)
            self.assertTrue(info["interrupted"])
            second = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
                resume_interrupted=True,
            )
            self.assertTrue(second.ran)
            self.assertEqual((be / "one.txt").read_text(encoding="utf-8"), "1\n")
            self.assertEqual((be / "two.txt").read_text(encoding="utf-8"), "2\n")
        finally:
            layout.close()

    def test_isolated_partial_only_restores_missing(self) -> None:
        layout = self._layout()
        try:
            be = layout.ace_root / "ACE-Backend"
            fe = layout.ace_root / "ACE-Frontend"
            cp = "cp-isolated"
            write_full_checkpoint(
                layout.ace_root,
                cp,
                [
                    ("BACKEND", "keep.txt", "old\n"),
                    ("BACKEND", "missing.txt", "m\n"),
                ],
            )
            (be / "keep.txt").write_text("user\n", encoding="utf-8")
            (be / "missing.txt").unlink(missing_ok=True)
            result = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_START_WITH_ISOLATED_OPERATIONS, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual((be / "keep.txt").read_text(encoding="utf-8"), "user\n")
            self.assertTrue((be / "missing.txt").is_file())
        finally:
            layout.close()

    def test_coordinator_plus_engine_power_loss_simulation(self) -> None:
        layout = self._layout()
        try:
            cp = "cp-power"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "pwr.txt", "pwr\n")])
            layout.write_ledger("20260101-120000", clean=False)
            layout.write_ledger_checkpoint_ref(
                "20260101-120000",
                cp,
                layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp,
            )
            coord = layout.evaluate_coordinator(inspect_live_processes=False)
            self.assertEqual(coord.coordinator_outcome, COORDINATOR_AUTO_RECOVERY_ALLOWED)
            be = layout.ace_root / "ACE-Backend"
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                coord,
                backend_root=be,
                frontend_root=layout.ace_root / "ACE-Frontend",
            )
            self.assertTrue(engine.ran)
            self.assertEqual((be / "pwr.txt").read_text(encoding="utf-8"), "pwr\n")
        finally:
            layout.close()

    def test_isolated_cancel_still_allows_missing_restore(self) -> None:
        layout = self._layout()
        try:
            layout.write_ledger("20260101-120000", clean=False)
            cp = "cp-cancel-iso"
            write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "safe.txt", "safe\n")])
            layout.write_ledger_checkpoint_ref(
                "20260101-120000",
                cp,
                layout.ace_root / "_backups/local-dev/recovery-checkpoints" / cp,
            )
                layout.ace_root,
                coord,
                backend_root=be,
                frontend_root=layout.ace_root / "ACE-Frontend",
            )
            self.assertTrue(engine.ran)
            self.assertEqual((be / "safe.txt").read_text(encoding="utf-8"), "safe\n")
        finally:
            layout.close()

    def test_start_ace_local_not_wired_engine(self) -> None:
        start_path = Path(__file__).resolve().parent.parent / "scripts" / "Start-AceLocal.ps1"
        text = start_path.read_text(encoding="utf-8")
        self.assertNotIn("auto_restore_engine", text)
        self.assertNotIn("run_conditional_auto_recovery", text)


if __name__ == "__main__":
    unittest.main()
