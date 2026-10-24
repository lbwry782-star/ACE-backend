"""Recovery periodic save background activation (phase 5.6)."""
from __future__ import annotations

import unittest

from engine.ace_local_dev_recovery_periodic_save_service import (
    evaluate_periodic_save_background_activation,
)
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
)


class TestPeriodicSaveBackgroundActivation(unittest.TestCase):
    def test_allowed_start_allowed(self) -> None:
        payload = evaluate_periodic_save_background_activation(
            coordinator_outcome=COORDINATOR_START_ALLOWED,
            allow_periodic_save_service=True,
            launcher_session="20260201-120000",
        )
        self.assertTrue(payload["allowed"])
        self.assertTrue(payload["mustRunHidden"])

    def test_blocked_recovery_unresolved(self) -> None:
        payload = evaluate_periodic_save_background_activation(
            coordinator_outcome=COORDINATOR_START_ALLOWED,
            allow_periodic_save_service=True,
            launcher_session="20260201-120000",
            recovery_unresolved=True,
        )
        self.assertFalse(payload["allowed"])
        self.assertIn("recovery_unresolved", payload["reasons"])

    def test_blocked_isolated(self) -> None:
        payload = evaluate_periodic_save_background_activation(
            coordinator_outcome=COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
            allow_periodic_save_service=False,
            launcher_session="20260201-120000",
            isolated_operations=["publish_unknown"],
        )
        self.assertFalse(payload["allowed"])

    def test_auto_recovery_allowed_with_flag(self) -> None:
        payload = evaluate_periodic_save_background_activation(
            coordinator_outcome=COORDINATOR_AUTO_RECOVERY_ALLOWED,
            allow_periodic_save_service=True,
            launcher_session="sess",
        )
        self.assertTrue(payload["allowed"])

    def test_start_stop_wired_phase59(self) -> None:
        root = __import__("pathlib").Path(__file__).resolve().parent.parent / "scripts"
        start_text = (root / "Start-AceLocal.ps1").read_text(encoding="utf-8")
        stop_text = (root / "Stop-AceLocal.ps1").read_text(encoding="utf-8")
        self.assertIn("Start-AceLocalRecoveryPeriodicSaveFromLocalStackStart", start_text)
        self.assertIn("Stop-AceLocalRecoveryPeriodicSaveForLocalQuit", stop_text)


if __name__ == "__main__":
    unittest.main()
