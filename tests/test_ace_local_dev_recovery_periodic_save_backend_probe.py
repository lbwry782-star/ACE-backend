"""Backend health probe for periodic save (phase 5.8)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from engine.ace_local_dev_recovery_periodic_save_backend_probe import (
    HEALTH_PATH,
    probe_local_backend_health,
    validate_probe_url,
)
from engine.ace_local_dev_recovery_periodic_save_guard import assess_recovery_periodic_save_checkpoint
from engine.ace_local_dev_recovery_periodic_save_pre_write import evaluate_pre_checkpoint_write_gate
from tests.test_ace_local_dev_startup_recovery_check import StartupRecoveryTempLayout


class TestBackendProbe(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE", None)
        os.environ.pop("ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_URL", None)

    def test_validate_blocks_non_localhost(self) -> None:
        ok, reason = validate_probe_url("http://203.0.113.1/health")
        self.assertFalse(ok)
        self.assertEqual(reason, "probe_url_non_localhost_blocked")

    def test_fixture_ok_allows_assess(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            os.environ["ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE"] = "ok"
            result = assess_recovery_periodic_save_checkpoint(ace_root=layout.ace_root)
            self.assertTrue(result["allowed"])
            self.assertIn("backendProbe", result)
        finally:
            layout.close()

    def test_fixture_unreachable_blocks_checkpoint(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            layout.write_ledger("20260201-120000", clean=True)
            os.environ["ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE"] = "unreachable"
            gate = evaluate_pre_checkpoint_write_gate(layout.ace_root, "20260201-120000")
            self.assertFalse(gate["allowed"])
            self.assertTrue(any("backend_probe" in r for r in gate["reasons"]))
        finally:
            layout.close()

    def test_http_500_fixture(self) -> None:
        os.environ["ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_FIXTURE"] = "http500"
        payload = probe_local_backend_health()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["reason"], "backend_probe_http_500")

    def test_live_http_probe_ok(self) -> None:
        layout = StartupRecoveryTempLayout()
        try:
            body = b"ok"

            class FakeResp:
                status = 200

                def read(self, n: int = -1) -> bytes:
                    return body

                def __enter__(self):
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

            with mock.patch(
                "engine.ace_local_dev_recovery_periodic_save_backend_probe.urllib.request.urlopen",
                return_value=FakeResp(),
            ):
                payload = probe_local_backend_health(ace_root=layout.ace_root)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["endpoint"].endswith(HEALTH_PATH))
        finally:
            layout.close()

    def test_url_override_non_localhost_blocked(self) -> None:
        os.environ["ACE_RECOVERY_PERIODIC_SAVE_BACKEND_PROBE_URL"] = "http://example.com/health"
        layout = StartupRecoveryTempLayout()
        try:
            payload = probe_local_backend_health(ace_root=layout.ace_root)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["reason"], "probe_url_non_localhost_blocked")
        finally:
            layout.close()


if __name__ == "__main__":
    unittest.main()
