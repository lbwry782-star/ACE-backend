"""Phase 5.3 — restore completeness verification (temp only)."""
from __future__ import annotations

import json
import subprocess
import unittest
import uuid
from pathlib import Path

from engine.ace_local_dev_cancel_progress import CancelExecuteProgressWriter, STEP_BACKEND_RESTORE
from engine.ace_local_dev_publish_execute_progress import PublishExecuteProgressWriter, STEP_GITHUB_PAGES
from engine.ace_local_dev_publish_external_effects_journal import (
    PublishExternalEffectsJournalWriter,
    STAGE_GITHUB_PAGES,
    STAGE_UNKNOWN,
)
from engine.ace_local_dev_recovery_asset_coverage import recovery_asset_coverage_table
from engine.ace_local_dev_recovery_auto_restore_engine import (
    read_restore_journal,
    run_conditional_auto_recovery,
)
from engine.ace_local_dev_recovery_restore_completeness import (
    collect_protected_state_fingerprints,
    read_git_snapshot,
    verify_restore_completeness,
)
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
    evaluate_recovery_startup_coordination,
)
from tests.recovery_checkpoint_fixtures import write_full_checkpoint
from tests.test_ace_local_dev_recovery_auto_restore_engine import _decision, _init_git_repo
from tests.test_ace_local_dev_recovery_startup_coordinator import CoordinatorTempLayout, _session_id


class TestRecoveryPhase53Completeness(unittest.TestCase):
    def test_asset_coverage_table_complete(self) -> None:
        assets = recovery_asset_coverage_table()
        ids = {a["asset"] for a in assets}
        for required in (
            "backend_worktree_files",
            "env_dpapi_blobs",
            "production_registry",
            "open_active_cycle",
            "new_cancel_publish_state",
        ):
            self.assertIn(required, ids)

    def test_start_allowed_no_journal_no_file_touch(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        _init_git_repo(layout.ace_root / "ACE-Backend")
        _init_git_repo(layout.ace_root / "ACE-Frontend")
        layout.write_ledger("20260101-120000", clean=True)
        layout.write_managed("20260101-120000")
        try:
            be = layout.ace_root / "ACE-Backend"
            (be / "local.txt").write_bytes(b"stay\n")
            before = (be / "local.txt").read_bytes()
            coord = layout.evaluate_coordinator(inspect_live_processes=False)
            self.assertEqual(coord.coordinator_outcome, COORDINATOR_START_ALLOWED)
            self.assertEqual(coord.gui_recovery_mode, "none")
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                coord,
                backend_root=be,
                frontend_root=layout.ace_root / "ACE-Frontend",
            )
            self.assertTrue(engine.skipped)
            self.assertIsNone(read_restore_journal(layout.ace_root))
            self.assertEqual((be / "local.txt").read_bytes(), before)
        finally:
            layout.close()

    def test_git_head_unchanged_after_restore(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        be = layout.ace_root / "ACE-Backend"
        fe = layout.ace_root / "ACE-Frontend"
        _init_git_repo(be)
        _init_git_repo(fe)
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"], cwd=be, check=True)
        git_before = {"backend": read_git_snapshot(be), "frontend": read_git_snapshot(fe)}
        cp = "cp-git-safe"
        write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "tracked.txt", "t\n")])
        try:
            run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            report = verify_restore_completeness(
                ace_root=layout.ace_root,
                backend_root=be,
                frontend_root=fe,
                git_before=git_before,
                protected_before={},
            )
            self.assertTrue(report.git_backend_unchanged)
            self.assertTrue(report.git_frontend_unchanged)
        finally:
            layout.close()

    def test_env_restore_no_plaintext_in_engine_output(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        be = layout.ace_root / "ACE-Backend"
        fe = layout.ace_root / "ACE-Frontend"
        cp = "cp-env"
        write_full_checkpoint(
            layout.ace_root,
            cp,
            [("BACKEND", "code.txt", "c\n")],
            env_blobs={"backend.env.local": b"\x01\x02" * 16},
        )

        def _fake_decrypt(*, dpapi_path, target_path):
            target_path.write_bytes(b"KEY=value\n")
            return True, "env_restored"

        try:
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
                decrypt_and_write=_fake_decrypt,
            )
            self.assertIn("backend.env.local", engine.env_restored)
            blob = json.dumps(engine.to_dict())
            self.assertNotIn("KEY=value", blob)
        finally:
            layout.close()

    def test_env_modified_not_auto_restored(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        be = layout.ace_root / "ACE-Backend"
        fe = layout.ace_root / "ACE-Frontend"
        cp = "cp-env-block"
        write_full_checkpoint(
            layout.ace_root,
            cp,
            [],
            env_blobs={"backend.env.local": b"blob"},
        )
        (be / ".env.local").write_bytes(b"user-changed\n")
        try:
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual(engine.env_restored, [])
            self.assertEqual((be / ".env.local").read_bytes(), b"user-changed\n")
        finally:
            layout.close()

    def test_registry_and_sessions_unchanged(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        be = layout.ace_root / "ACE-Backend"
        fe = layout.ace_root / "ACE-Frontend"
        _init_git_repo(be)
        _init_git_repo(fe)
        reg_dir = layout.ace_root / "_registry"
        reg_dir.mkdir(parents=True)
        (reg_dir / "production-registry.json").write_text('{"x":1}\n', encoding="utf-8")
        (reg_dir / "publish-candidate-registry.json").write_text('{"y":2}\n', encoding="utf-8")
        sid = _session_id()
        session_dir = layout.session_dir(sid)
        (session_dir / "session.json").write_text(
            json.dumps({"status": "OPEN", "sessionId": sid}) + "\n",
            encoding="utf-8",
        )
        protected_before = collect_protected_state_fingerprints(layout.ace_root)
        cp = "cp-protected"
        write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "a.txt", "a\n")])
        try:
            engine = run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertTrue(engine.completeness.get("protectedUnchanged"))
            self.assertEqual(
                (session_dir / "session.json").read_text(encoding="utf-8"),
                json.dumps({"status": "OPEN", "sessionId": sid}) + "\n",
            )
        finally:
            layout.close()

    def test_publish_unknown_registry_unchanged_isolated(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        be = layout.ace_root / "ACE-Backend"
        fe = layout.ace_root / "ACE-Frontend"
        layout.write_ledger("20260101-120000", clean=False)
        reg_dir = layout.ace_root / "_registry"
        reg_dir.mkdir(parents=True)
        reg_path = reg_dir / "publish-candidate-registry.json"
        reg_path.write_text('{"status":"ACTIVE"}\n', encoding="utf-8")
        before_reg = reg_path.read_text(encoding="utf-8")
        sid = _session_id()
        op = uuid.uuid4().hex
        PublishExecuteProgressWriter(
            session_id=sid, operation_id=op, sessions_root=layout.sessions_root
        ).begin_step(STEP_GITHUB_PAGES)
        PublishExternalEffectsJournalWriter(
            session_id=sid,
            operation_id=op,
            sessions_root=layout.sessions_root,
        ).set_stage(STAGE_GITHUB_PAGES, STAGE_UNKNOWN, error="unknown")
        coord = evaluate_recovery_startup_coordination(
            layout.ace_root,
            sessions_root=layout.sessions_root,
        )
        self.assertEqual(coord.coordinator_outcome, COORDINATOR_START_WITH_ISOLATED_OPERATIONS)
        try:
            run_conditional_auto_recovery(
                layout.ace_root,
                coord,
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual(reg_path.read_text(encoding="utf-8"), before_reg)
        finally:
            layout.close()

    def test_uncommitted_changes_preserved(self) -> None:
        layout = CoordinatorTempLayout()
        layout.ensure_workspace_roots()
        be = layout.ace_root / "ACE-Backend"
        fe = layout.ace_root / "ACE-Frontend"
        _init_git_repo(be)
        cp = "cp-uncommitted"
        write_full_checkpoint(layout.ace_root, cp, [("BACKEND", "old.txt", "old\n")])
        (be / "old.txt").write_bytes(b"my-edit\n")
        (be / "new-only.txt").write_bytes(b"new\n")
        try:
            run_conditional_auto_recovery(
                layout.ace_root,
                _decision(COORDINATOR_AUTO_RECOVERY_ALLOWED, checkpoint_id=cp),
                backend_root=be,
                frontend_root=fe,
            )
            self.assertEqual((be / "old.txt").read_bytes(), b"my-edit\n")
            self.assertEqual((be / "new-only.txt").read_bytes(), b"new\n")
        finally:
            layout.close()

    def test_start_ace_local_not_wired_phase53(self) -> None:
        start = Path(__file__).resolve().parent.parent / "scripts" / "Start-AceLocal.ps1"
        text = start.read_text(encoding="utf-8")
        self.assertNotIn("restore_completeness", text)
        self.assertNotIn("phase53", text)


if __name__ == "__main__":
    unittest.main()
