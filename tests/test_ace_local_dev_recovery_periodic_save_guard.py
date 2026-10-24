"""Tests for live recovery periodic save guard."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from engine.ace_local_dev_execute_guard import (
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_recovery_periodic_save_guard import assess_recovery_periodic_save_checkpoint


class RecoveryPeriodicSaveGuardTests(unittest.TestCase):
    def tearDown(self) -> None:
        end_local_dev_execute()
        os.environ.pop("ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE", None)

    def setUp(self) -> None:
        os.environ["ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE"] = "ok"

    def test_execute_guard_blocks(self) -> None:
        try_begin_local_dev_execute()
        with tempfile.TemporaryDirectory(prefix="ace-cp41-guard-") as tmp:
            ace = Path(tmp)
            (ace / "_backups" / "local-dev").mkdir(parents=True)
            result = assess_recovery_periodic_save_checkpoint(ace_root=ace)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["reason"], "execute_guard_busy")

    def test_active_cancel_blocks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ace-cp41-guard-") as tmp:
            ace = Path(tmp)
            sessions = ace / "_backups" / "sessions"
            sessions.mkdir(parents=True)
            (sessions / ".ori-cancel-dev.lock").write_text('{"status":"IN_PROGRESS"}\n', encoding="utf-8")
            prev = os.environ.get("ACE_ORI_ACE_ROOT")
            os.environ["ACE_ORI_ACE_ROOT"] = str(ace)
            try:
                result = assess_recovery_periodic_save_checkpoint(ace_root=ace)
            finally:
                if prev is None:
                    os.environ.pop("ACE_ORI_ACE_ROOT", None)
                else:
                    os.environ["ACE_ORI_ACE_ROOT"] = prev
        self.assertFalse(result["allowed"])


if __name__ == "__main__":
    unittest.main()
