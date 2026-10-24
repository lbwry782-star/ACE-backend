"""Recovery startup coordinator tests (temp dirs only)."""
from __future__ import annotations

import hashlib
import json
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from engine.ace_local_dev_cancel_progress import (
    CancelExecuteProgressWriter,
    STEP_BACKEND_RESTORE,
)
from engine.ace_local_dev_publish_execute_progress import (
    PublishExecuteProgressWriter,
    STEP_GITHUB_PAGES,
)
from engine.ace_local_dev_publish_external_effects_journal import (
    PublishExternalEffectsJournalWriter,
    STAGE_GITHUB_PAGES,
    STAGE_UNKNOWN,
)
from engine.ace_local_dev_recovery_checkpoint_validate import (
    recovery_checkpoints_root,
    validate_recovery_checkpoint_root,
)
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_RECOVERY_UNSAFE,
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_BLOCKED,
    COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
    assert_coordinator_allows_periodic_save,
    build_periodic_save_activation_contract,
    evaluate_recovery_startup_coordination,
    release_startup_coordinator_lock,
    try_acquire_startup_coordinator_lock,
)
from engine.ace_local_dev_startup_recovery_check import (
    read_recovery_ledger,
    recovery_ledger_path,
)
from tests.test_ace_local_dev_startup_recovery_check import (
    StartupRecoveryTempLayout,
    _session_id,
)


def _write_minimal_verified_checkpoint(
    ace_root: Path,
    checkpoint_id: str,
    *,
    corrupt: bool = False,
) -> Path:
    root = recovery_checkpoints_root(ace_root) / checkpoint_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "BACKEND" / "git").mkdir(parents=True, exist_ok=True)
    (root / "FRONTEND" / "git").mkdir(parents=True, exist_ok=True)
    (root / "BACKEND" / "git" / "HEAD.txt").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / "FRONTEND" / "git" / "HEAD.txt").write_text("ref: refs/heads/main\n", encoding="utf-8")
    mirror = root / "BACKEND" / "mirror" / "sample.txt"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text("ok\n", encoding="utf-8")
    digest = hashlib.sha256(mirror.read_bytes()).hexdigest()
    if corrupt:
        digest = "0" * 64
    manifest = "Area,RelPath,ChangeType,BackupSha256\nBACKEND,sample.txt,modified,{0}\n".format(digest)
    (root / "MANIFEST.csv").write_text(manifest, encoding="utf-8")
    verified_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "checkpointId": checkpoint_id,
        "status": "VERIFIED",
        "verifiedAt": verified_at,
        "checkpointType": "full",
        "checkpointPath": str(root),
    }
    (root / "checkpoint-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return root


class CoordinatorTempLayout(StartupRecoveryTempLayout):
    def ensure_workspace_roots(self) -> None:
        (self.ace_root / "ACE-Backend").mkdir(parents=True, exist_ok=True)
        (self.ace_root / "ACE-Frontend").mkdir(parents=True, exist_ok=True)

    def write_ledger_checkpoint_ref(self, launcher: str, checkpoint_id: str, cp_root: Path) -> None:
        path = recovery_ledger_path(self.ace_root)
        ledger = json.loads(path.read_text(encoding="utf-8"))
        ledger["lastVerifiedCheckpoint"] = {
            "checkpointId": checkpoint_id,
            "checkpointPath": str(cp_root),
            "checkpointType": "full",
            "verifiedAt": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    def evaluate_coordinator(self, **kwargs):
        return evaluate_recovery_startup_coordination(
            self.ace_root,
            sessions_root=self.sessions_root,
            **kwargs,
        )


class TestRecoveryStartupCoordinator(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = CoordinatorTempLayout()

    def tearDown(self) -> None:
        release_startup_coordinator_lock(self.layout.ace_root)
        self.layout.close()

    def test_first_start_allowed(self) -> None:
        d = self.layout.evaluate_coordinator()
        self.assertEqual(d.coordinator_outcome, COORDINATOR_START_ALLOWED)
        self.assertEqual(d.gui_recovery_mode, "none")
        self.assertTrue(d.allow_periodic_save_service)

    def test_clean_quit_start_allowed(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=True)
        self.layout.write_managed("20260101-120000")
        d = self.layout.evaluate_coordinator(inspect_live_processes=False)
        self.assertEqual(d.coordinator_outcome, COORDINATOR_START_ALLOWED)
        self.assertEqual(d.termination_type, "QUIT")

    def test_power_loss_without_checkpoint_unsafe_but_local_may_start(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        d = self.layout.evaluate_coordinator(inspect_live_processes=False)
        self.assertEqual(d.coordinator_outcome, COORDINATOR_RECOVERY_UNSAFE)
        self.assertTrue(d.allow_local_processes)
        self.assertFalse(d.allow_periodic_save_service)

    def test_power_loss_auto_recovery_with_verified_checkpoint(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        self.layout.ensure_workspace_roots()
        cp_root = _write_minimal_verified_checkpoint(self.layout.ace_root, "cp-auto-1")
        self.layout.write_ledger_checkpoint_ref("20260101-120000", "cp-auto-1", cp_root)
        d = self.layout.evaluate_coordinator(inspect_live_processes=False)
        self.assertEqual(d.coordinator_outcome, COORDINATOR_AUTO_RECOVERY_ALLOWED)
        self.assertTrue(d.auto_restore_plan.get("safeAutoApply"))
        self.assertTrue(d.allow_local_processes)

    def test_app_crash_allows_isolated_local_start(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        self.layout.write_managed("20260101-120000", backend_pid=999999)
        d = self.layout.evaluate_coordinator(inspect_live_processes=True)
        self.assertIn(
            d.coordinator_outcome,
            (COORDINATOR_AUTO_RECOVERY_ALLOWED, COORDINATOR_START_WITH_ISOLATED_OPERATIONS, COORDINATOR_RECOVERY_UNSAFE),
        )
        self.assertTrue(d.allow_local_processes)

    def test_os_crash_via_evidence(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        d = self.layout.evaluate_coordinator(
            inspect_live_processes=False,
            termination_evidence_override={
                "windows": {
                    "eventLogAvailable": True,
                    "bugcheckOrKernelCrash": True,
                    "unexpectedShutdown": False,
                },
            },
        )
        self.assertEqual(d.termination_type, "OS_CRASH")
        self.assertEqual(d.coordinator_outcome, COORDINATOR_RECOVERY_UNSAFE)

    def test_power_loss_via_evidence(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        d = self.layout.evaluate_coordinator(
            inspect_live_processes=False,
            termination_evidence_override={
                "windows": {
                    "eventLogAvailable": True,
                    "unexpectedShutdown": True,
                    "bugcheckOrKernelCrash": False,
                },
            },
        )
        self.assertEqual(d.termination_type, "POWER_LOSS")

    def test_cancel_critical_isolated_local_still_allowed(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        w.begin_step(STEP_BACKEND_RESTORE)
        d = self.layout.evaluate_coordinator()
        self.assertEqual(d.coordinator_outcome, COORDINATOR_START_WITH_ISOLATED_OPERATIONS)
        self.assertTrue(d.allow_local_processes)
        self.assertIn("CANCEL", d.blocked_dev_operation_kinds)

    def test_publish_unknown_isolated_no_deploy(self) -> None:
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
        d = self.layout.evaluate_coordinator()
        self.assertEqual(d.coordinator_outcome, COORDINATOR_START_WITH_ISOLATED_OPERATIONS)
        self.assertTrue(d.unfinished_operations.get("publishUnknownExternal"))
        self.assertTrue(d.allow_local_processes)

    def test_new_incomplete_isolated(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        session_dir = self.layout.session_dir(sid)
        (session_dir / "session.json").write_text(
            json.dumps({"status": "IN_PROGRESS", "sessionId": sid}) + "\n",
            encoding="utf-8",
        )
        d = self.layout.evaluate_coordinator(inspect_live_processes=False)
        self.assertEqual(d.coordinator_outcome, COORDINATOR_START_WITH_ISOLATED_OPERATIONS)
        self.assertTrue(d.unfinished_operations.get("newIncomplete"))

    def test_corrupt_ledger_unsafe_preserve_evidence(self) -> None:
        self.layout.write_ledger("20260101-120000", corrupt=True)
        d = self.layout.evaluate_coordinator()
        self.assertEqual(d.coordinator_outcome, COORDINATOR_RECOVERY_UNSAFE)
        self.assertTrue(d.evidence_preservation.get("preserveLedgerUntilIncidentRecorded"))
        self.assertFalse(d.allow_ledger_initialization)

    def test_last_checkpoint_corrupt_prior_ok(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        good_id = "cp-good-001"
        bad_id = "cp-bad-002"
        _write_minimal_verified_checkpoint(self.layout.ace_root, good_id)
        bad_root = _write_minimal_verified_checkpoint(self.layout.ace_root, bad_id, corrupt=True)
        self.layout.write_ledger_checkpoint_ref("20260101-120000", bad_id, bad_root)
        d = self.layout.evaluate_coordinator(inspect_live_processes=False)
        self.assertFalse(validate_recovery_checkpoint_root(bad_root).ok)
        self.assertEqual(d.checkpoint_assessment.get("recommendedCheckpointId"), good_id)

    def test_session_mismatch_isolated_start(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=True)
        self.layout.write_managed("20260101-999999")
        d = self.layout.evaluate_coordinator()
        self.assertEqual(d.coordinator_outcome, COORDINATOR_START_WITH_ISOLATED_OPERATIONS)
        self.assertTrue(d.allow_local_processes)

    def test_periodic_save_blocked_when_isolated(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        w.begin_step(STEP_BACKEND_RESTORE)
        d = self.layout.evaluate_coordinator()
        with self.assertRaises(RuntimeError):
            assert_coordinator_allows_periodic_save(d, launcher_session="20260201-120000")

    def test_periodic_save_when_auto_recovery_clean(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        self.layout.ensure_workspace_roots()
        cp_root = _write_minimal_verified_checkpoint(self.layout.ace_root, "cp-ps-1")
        self.layout.write_ledger_checkpoint_ref("20260101-120000", "cp-ps-1", cp_root)
        d = self.layout.evaluate_coordinator(inspect_live_processes=False)
        self.assertEqual(d.coordinator_outcome, COORDINATOR_AUTO_RECOVERY_ALLOWED)
        assert_coordinator_allows_periodic_save(d, launcher_session="20260201-120000")

    def test_coordinator_does_not_mutate_ledger(self) -> None:
        self.layout.write_ledger("20260101-111111", clean=False)
        before, _ = read_recovery_ledger(self.layout.ace_root)
        assert before is not None
        _ = self.layout.evaluate_coordinator(inspect_live_processes=False)
        after, _ = read_recovery_ledger(self.layout.ace_root)
        assert after is not None
        self.assertEqual(before.get("launcherSession"), after.get("launcherSession"))

    def test_concurrent_startup_blocked(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=True)
        self.layout.write_managed("20260101-120000")
        lock = try_acquire_startup_coordinator_lock(self.layout.ace_root)
        self.assertTrue(lock.get("acquired"))
        try:
            d = self.layout.evaluate_coordinator(inspect_live_processes=False)
            self.assertEqual(d.coordinator_outcome, COORDINATOR_START_BLOCKED)
        finally:
            release_startup_coordinator_lock(self.layout.ace_root)

    def test_evidence_preservation_snapshot(self) -> None:
        self.layout.write_ledger("20260101-120000", clean=False)
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root
        )
        w.begin_step(STEP_BACKEND_RESTORE)
        d = self.layout.evaluate_coordinator()
        self.assertTrue(d.evidence_preservation.get("observationEvidence"))

    def test_start_ace_local_gate_only_no_restore(self) -> None:
        start_path = Path(__file__).resolve().parent.parent / "scripts" / "Start-AceLocal.ps1"
        text = start_path.read_text(encoding="utf-8")
        self.assertIn("AceLocalRecoveryStartupGate", text)
        self.assertNotIn("run_conditional_auto_recovery", text)


if __name__ == "__main__":
    unittest.main()
