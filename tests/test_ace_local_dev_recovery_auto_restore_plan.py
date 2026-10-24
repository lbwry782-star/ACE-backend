"""Auto-restore plan tests (temp sandboxes only)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.ace_local_dev_recovery_auto_restore_plan import (
    ACTION_PRESERVE_BOTH,
    ACTION_RESTORE_FROM_CHECKPOINT,
    ACTION_SKIP_UNCHANGED,
    apply_auto_restore_plan_in_test_sandbox,
    build_auto_restore_plan,
)
from tests.recovery_checkpoint_fixtures import write_full_checkpoint


class TestAutoRestorePlan(unittest.TestCase):
    def test_unchanged_files_skip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ace-auto-restore-") as tmp:
            ace = Path(tmp) / "ACE"
            backend = ace / "ACE-Backend"
            frontend = ace / "ACE-Frontend"
            backend.mkdir(parents=True)
            frontend.mkdir(parents=True)
            cp_id = "cp1"
            write_full_checkpoint(ace, cp_id, [("BACKEND", "sample.txt", "ok\n")])
            (backend / "sample.txt").write_bytes(b"ok\n")
            plan = build_auto_restore_plan(
                ace_root=ace,
                checkpoint_id=cp_id,
                backend_root=backend,
                frontend_root=frontend,
            )
            self.assertTrue(plan.safe_auto_apply)
            self.assertEqual(plan.entries[0].action, ACTION_SKIP_UNCHANGED)

    def test_newer_workspace_preserved_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ace-auto-restore-") as tmp:
            ace = Path(tmp) / "ACE"
            backend = ace / "ACE-Backend"
            frontend = ace / "ACE-Frontend"
            backend.mkdir(parents=True)
            frontend.mkdir(parents=True)
            cp_id = "cp1"
            write_full_checkpoint(ace, cp_id, [("BACKEND", "sample.txt", "ok\n")])
            (backend / "sample.txt").write_text("user-edited\n", encoding="utf-8")
            plan = build_auto_restore_plan(
                ace_root=ace,
                checkpoint_id=cp_id,
                backend_root=backend,
                frontend_root=frontend,
            )
            self.assertFalse(plan.safe_auto_apply)
            self.assertEqual(plan.entries[0].action, ACTION_PRESERVE_BOTH)
            apply = apply_auto_restore_plan_in_test_sandbox(
                plan,
                backend_root=backend,
                frontend_root=frontend,
                safety_backup_parent=Path(tmp) / "safety",
            )
            self.assertIn("sample.txt", apply["preserved"])
            self.assertEqual((backend / "sample.txt").read_text(encoding="utf-8"), "user-edited\n")

    def test_missing_file_restored_in_sandbox(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ace-auto-restore-") as tmp:
            ace = Path(tmp) / "ACE"
            backend = ace / "ACE-Backend"
            frontend = ace / "ACE-Frontend"
            backend.mkdir(parents=True)
            frontend.mkdir(parents=True)
            cp_id = "cp1"
            write_full_checkpoint(ace, cp_id, [("BACKEND", "sample.txt", "ok\n")])
            sample = backend / "sample.txt"
            self.assertFalse(sample.is_file())
            plan = build_auto_restore_plan(
                ace_root=ace,
                checkpoint_id=cp_id,
                backend_root=backend,
                frontend_root=frontend,
            )
            self.assertTrue(plan.safe_auto_apply)
            self.assertEqual(plan.entries[0].action, ACTION_RESTORE_FROM_CHECKPOINT)
            apply = apply_auto_restore_plan_in_test_sandbox(
                plan,
                backend_root=backend,
                frontend_root=frontend,
                safety_backup_parent=Path(tmp) / "safety",
            )
            self.assertIn("sample.txt", apply["applied"])
            self.assertEqual((sample).read_text(encoding="utf-8"), "ok\n")


if __name__ == "__main__":
    unittest.main()
