"""
LOCAL shutdown gate on API entry points — TEMP ace root only.

Run: python -m unittest tests.test_ace_local_dev_shutdown_route_gate -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from engine.ace_local_dev_config import (
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)
from engine.ace_local_dev_shutdown_gate import (
    GATE_SAFE_TO_STOP,
    GATE_SHUTDOWN_REQUESTED,
    request_local_shutdown,
)
from engine.ace_local_dev_shutdown_route_gate import (
    ERROR_SHUTDOWN_GATE_DENIED,
    ERROR_SHUTDOWN_GATE_STATE_UNREADABLE,
    local_dev_shutdown_blocks_new_operation,
)
from tests.test_ace_local_dev_cycle import _session_fixture


class ShutdownRouteGateTempLayout:
    def __init__(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-shutdown-route-gate-")
        self.ace_root = Path(self._tmp) / "ACE"
        self.local_dev = self.ace_root / "_backups" / "local-dev"
        self.sessions_root = self.ace_root / "_backups" / "sessions"
        self.local_dev.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.launcher_session = "20260101-120000"
        (self.local_dev / "managed-processes.json").write_text(
            json.dumps({"launcherSession": self.launcher_session}) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class LocalDevShutdownRouteGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        self.layout = ShutdownRouteGateTempLayout()
        os.environ["ACE_ORI_ACE_ROOT"] = str(self.layout.ace_root)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.layout.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_ORI_ACE_ROOT", None)
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        self.layout.close()

    def _request_shutdown(self) -> None:
        request_local_shutdown(
            self.layout.ace_root,
            launcher_session=self.layout.launcher_session,
        )

    @patch("engine.ace_local_dev_start_cycle.run_new_session_baseline_json")
    @patch("engine.ace_local_dev_start_cycle.run_baseline_validate_json")
    def test_normal_new_allowed(self, mock_validate, mock_baseline) -> None:
        mock_validate.return_value = {"ok": True, "mode": "VALIDATE"}
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.layout.sessions_root, sid)
        mock_baseline.return_value = {
            "ok": True,
            "sessionId": sid,
            "status": "READY",
            "sessionRoot": str(self.layout.sessions_root / sid),
        }
        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get("ok"))

    @patch("engine.ace_local_dev_cancel_execute.run_post_cancel_auto_start")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_normal_cancel_execute_allowed(
        self, mock_preview, mock_execute, mock_auto
    ) -> None:
        from engine.ace_local_dev_cycle_store import write_active_dev_cycle_pointer
        from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint

        sid = "ori-20261024-120000-bbccddee"
        _session_fixture(self.layout.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.layout.sessions_root)
        preview_raw = {
            "ok": True,
            "plan": {
                "sessionId": sid,
                "canExecute": True,
                "blockers": [],
                "registryInfraDrift": [],
                "cancelCycle": {"status": "OPEN"},
                "infraTargets": {
                    "backendCommit": "7" * 40,
                    "frontendCommit": "b" * 40,
                },
                "commitsToCancel": {"backend": [], "frontend": []},
                "git": {"backend": {}, "frontend": {}},
            },
        }
        mock_preview.return_value = preview_raw
        mock_execute.return_value = {"ok": True, "result": {"status": "SUCCESS"}}
        mock_auto.return_value = {"ok": False, "phase": "PENDING", "cancelledSessionId": sid}
        fp = compute_preview_fingerprint(preview_raw)
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        self.assertNotIn(resp.status_code, (503,))
        self.assertNotEqual((resp.get_json() or {}).get("error"), ERROR_SHUTDOWN_GATE_DENIED)
        body = resp.get_json() or {}
        self.assertNotEqual(body.get("error"), ERROR_SHUTDOWN_GATE_DENIED)

    @patch("engine.ace_local_dev_publish_uri_candidate_dry_run.execute_publish_uri_candidate_dry_run")
    def test_normal_publish_dry_run_allowed(self, mock_dry) -> None:
        sid = "ori-20261024-120000-ccddff00"
        _session_fixture(self.layout.sessions_root, sid)
        from engine.ace_local_dev_cycle_store import write_active_dev_cycle_pointer

        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.layout.sessions_root)
        mock_dry.return_value = {"ok": True, "dryRun": True}
        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/dry-run",
            json={
                "candidateBackendSha": "a" * 40,
                "publishUriPreviewFingerprint": "fp",
            },
        )
        self.assertNotEqual(resp.get_json().get("error"), ERROR_SHUTDOWN_GATE_DENIED)

    def test_shutdown_blocks_new_start(self) -> None:
        self._request_shutdown()
        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
        )
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertFalse(body.get("ok"))
        self.assertEqual(body.get("error"), ERROR_SHUTDOWN_GATE_DENIED)
        self.assertTrue(body.get("shutdownGateBlocked"))

    @patch("engine.ace_local_dev_publish_uri_candidate_dry_run.execute_publish_uri_candidate_dry_run")
    def test_shutdown_blocks_publish_dry_run(self, mock_dry) -> None:
        sid = "ori-20261024-120000-ddff0011"
        _session_fixture(self.layout.sessions_root, sid)
        from engine.ace_local_dev_cycle_store import write_active_dev_cycle_pointer

        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.layout.sessions_root)
        self._request_shutdown()
        resp = self.client.post(
            "/api/local-dev/publish-uri/candidate/dry-run",
            json={
                "candidateBackendSha": "a" * 40,
                "publishUriPreviewFingerprint": "fp",
            },
        )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json().get("error"), ERROR_SHUTDOWN_GATE_DENIED)
        mock_dry.assert_not_called()

    @patch("engine.ace_local_dev_cancel_execute.run_post_cancel_auto_start")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_shutdown_cancel_execute_not_blocked_by_gate(
        self, mock_preview, mock_execute, mock_auto
    ) -> None:
        from engine.ace_local_dev_cycle_store import write_active_dev_cycle_pointer
        from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint

        sid = "ori-20261024-120000-eeff0011"
        _session_fixture(self.layout.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.layout.sessions_root)
        preview_raw = {
            "ok": True,
            "plan": {
                "sessionId": sid,
                "canExecute": True,
                "blockers": [],
                "registryInfraDrift": [],
                "cancelCycle": {"status": "OPEN"},
                "infraTargets": {
                    "backendCommit": "7" * 40,
                    "frontendCommit": "b" * 40,
                },
                "commitsToCancel": {"backend": [], "frontend": []},
                "git": {"backend": {}, "frontend": {}},
            },
        }
        mock_preview.return_value = preview_raw
        mock_execute.return_value = {"ok": True, "result": {"status": "SUCCESS"}}
        mock_auto.return_value = {"ok": False, "phase": "PENDING", "cancelledSessionId": sid}

        self._request_shutdown()
        fp = compute_preview_fingerprint(preview_raw)
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        body = resp.get_json() or {}
        self.assertNotEqual(body.get("error"), ERROR_SHUTDOWN_GATE_DENIED)
        self.assertFalse(body.get("shutdownGateBlocked"))

    def test_session_mismatch_blocks(self) -> None:
        request_local_shutdown(
            self.layout.ace_root,
            launcher_session="other-session-999",
        )
        code = local_dev_shutdown_blocks_new_operation(
            ace_root=self.layout.ace_root,
            expected_launcher_session=self.layout.launcher_session,
        )
        self.assertEqual(code, "ace_shutdown_gate_session_mismatch")

    def test_corrupt_gate_file_blocks(self) -> None:
        gate_path = self.layout.local_dev / "shutdown-gate-state.json"
        gate_path.write_text("{not-json", encoding="utf-8")
        code = local_dev_shutdown_blocks_new_operation(ace_root=self.layout.ace_root)
        self.assertEqual(code, ERROR_SHUTDOWN_GATE_STATE_UNREADABLE)

    def test_safe_to_stop_blocks_new_start(self) -> None:
        request_local_shutdown(
            self.layout.ace_root,
            launcher_session=self.layout.launcher_session,
        )
        from engine.ace_local_dev_shutdown_gate import refresh_local_shutdown_gate

        refresh_local_shutdown_gate(
            self.layout.ace_root,
            launcher_session=self.layout.launcher_session,
            sessions_root=self.layout.sessions_root,
        )
        state_path = self.layout.local_dev / "shutdown-gate-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state.get("gateStatus"), GATE_SAFE_TO_STOP)
        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
        )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(
            resp.get_json().get("error"),
            "ace_shutdown_gate_shutting_down",
        )

    def test_builder_preview_unaffected(self) -> None:
        self._request_shutdown()
        resp = self.client.post(
            "/api/preview",
            json={"productDescription": "test"},
        )
        body = resp.get_json()
        if isinstance(body, dict):
            self.assertNotEqual(body.get("error"), ERROR_SHUTDOWN_GATE_DENIED)
            self.assertFalse(body.get("shutdownGateBlocked"))
        self.assertNotIn(ERROR_SHUTDOWN_GATE_DENIED, resp.get_data(as_text=True))

    def test_dev_cycle_status_unaffected(self) -> None:
        self._request_shutdown()
        resp = self.client.get("/api/local-dev/dev-cycle/status")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("shutdownGateBlocked", resp.get_json() or {})


if __name__ == "__main__":
    unittest.main()
