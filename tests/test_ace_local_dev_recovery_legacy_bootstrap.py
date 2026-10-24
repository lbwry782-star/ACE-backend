"""Legacy install recovery bootstrap assessment (temp dirs only)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from engine.ace_local_dev_recovery_legacy_bootstrap import (
    CLASSIFICATION_LEGACY,
    RESTORE_JOURNAL_FILENAME,
    assess_legacy_install_recovery_bootstrap,
    bootstrap_lock_path,
)
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP,
    evaluate_recovery_startup_coordination,
)
from engine.ace_local_dev_shutdown_gate import _write_json_atomic
from engine.ace_local_dev_startup_recovery_check import (
    STARTUP_FIRST_START,
    STARTUP_LEGACY_INSTALL_RECOVERY_BOOTSTRAP,
    evaluate_previous_local_startup,
)


class LegacyBootstrapLayout:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ace-legacy-bootstrap-")
        self.ace_root = Path(self._tmp.name) / "ACE"
        self.local_dev = self.ace_root / "_backups" / "local-dev"
        self.sessions = self.ace_root / "_backups" / "sessions"
        self.registry = self.ace_root / "_registry"
        self.local_dev.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.registry.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._tmp.cleanup()

    def write_managed(self, launcher: str) -> None:
        payload = {
            "schemaVersion": 2,
            "launcherSession": launcher,
            "components": None,
            "updatedAt": "2026-01-01T00:00:00+00:00",
        }
        _write_json_atomic(self.local_dev / "managed-processes.json", payload)

    def write_open_active(self) -> None:
        _write_json_atomic(
            self.sessions / "active-dev-cycle.json",
            {"cycleId": f"ori-{uuid.uuid4().hex[:8]}", "phase": "OPEN"},
        )
        _write_json_atomic(
            self.registry / "production-registry.json",
            {"activePublishCandidateId": "test-active"},
        )


class TestLegacyBootstrapAssessment(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = LegacyBootstrapLayout()

    def tearDown(self) -> None:
        self.layout.close()

    def test_fresh_install_not_legacy(self) -> None:
        d = evaluate_previous_local_startup(
            self.layout.ace_root,
            sessions_root=self.layout.sessions,
            inspect_live_processes=False,
        )
        self.assertEqual(d.outcome, STARTUP_FIRST_START)

    def test_legacy_allowed(self) -> None:
        launcher = "20261024-040249"
        self.layout.write_managed(launcher)
        self.layout.write_open_active()
        assess = assess_legacy_install_recovery_bootstrap(
            self.layout.ace_root,
            sessions_root=self.layout.sessions,
        )
        self.assertEqual(assess.classification, CLASSIFICATION_LEGACY)
        self.assertTrue(assess.bootstrap_allowed)
        d = evaluate_previous_local_startup(
            self.layout.ace_root,
            sessions_root=self.layout.sessions,
            inspect_live_processes=False,
        )
        self.assertEqual(d.outcome, STARTUP_LEGACY_INSTALL_RECOVERY_BOOTSTRAP)
        coord = evaluate_recovery_startup_coordination(
            self.layout.ace_root,
            sessions_root=self.layout.sessions,
            inspect_live_processes=False,
            acquire_coordinator_lock=False,
        )
        self.assertEqual(coord.coordinator_outcome, COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP)
        self.assertTrue(coord.allow_ledger_initialization)
        self.assertFalse(coord.allow_periodic_save_service)

    def test_restore_journal_blocks(self) -> None:
        self.layout.write_managed("20261024-040249")
        (self.layout.local_dev / RESTORE_JOURNAL_FILENAME).write_text(
            json.dumps({"status": "IN_PROGRESS"}) + "\n",
            encoding="utf-8",
        )
        assess = assess_legacy_install_recovery_bootstrap(
            self.layout.ace_root,
            sessions_root=self.layout.sessions,
        )
        self.assertFalse(assess.bootstrap_allowed)
        self.assertIn("recovery_restore_journal_present", assess.block_reasons)

    def test_checkpoint_dir_blocks(self) -> None:
        self.layout.write_managed("20261024-040249")
        cp_root = self.layout.local_dev / "recovery-checkpoints" / "cp-test"
        cp_root.mkdir(parents=True)
        assess = assess_legacy_install_recovery_bootstrap(
            self.layout.ace_root,
            sessions_root=self.layout.sessions,
        )
        self.assertFalse(assess.bootstrap_allowed)

    def test_bootstrap_lock_blocks(self) -> None:
        self.layout.write_managed("20261024-040249")
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            bootstrap_lock_path(self.layout.ace_root).write_text(
                json.dumps({"pid": proc.pid}) + "\n",
                encoding="utf-8",
            )
            assess = assess_legacy_install_recovery_bootstrap(
                self.layout.ace_root,
                sessions_root=self.layout.sessions,
            )
            self.assertFalse(assess.bootstrap_allowed)
        finally:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
