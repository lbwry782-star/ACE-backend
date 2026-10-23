"""
Start cycle + cancel execute local API — TEMP / mocked tests only.

Run: python -m unittest tests.test_ace_local_dev_start_cancel_api -v
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
from engine.ace_local_dev_cycle_store import (
    ACTIVE_CYCLE_FILENAME,
    resolve_active_dev_cycle,
    write_active_dev_cycle_pointer,
)
from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint
from tests.test_ace_local_dev_cycle import _session_fixture


def _sample_preview_raw(session_id: str, *, can_execute: bool = True, blockers=None) -> dict:
    return {
        "ok": True,
        "plan": {
            "sessionId": session_id,
            "canExecute": can_execute,
            "blockers": blockers or [],
            "registryInfraDrift": [],
            "cancelCycle": {"status": "OPEN"},
            "infraTargets": {
                "backendCommit": "7f380051a27f294cc8ed0d5a42dbaffc71879ca2",
                "frontendCommit": "b1aa0ad2907a083dd8b9e77b2b4621171d2cbbe5",
            },
            "commitsToCancel": {"backend": [], "frontend": []},
            "git": {"backend": {}, "frontend": {}},
        },
    }


class LocalDevStartCancelApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-start-cancel-api-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)
        os.environ.pop("ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE", None)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    @patch("engine.ace_local_dev_start_cycle.run_new_session_baseline_json")
    @patch("engine.ace_local_dev_start_cycle.run_baseline_validate_json")
    def test_start_cycle_success(self, mock_validate, mock_baseline) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        mock_validate.return_value = {"ok": True, "mode": "VALIDATE"}
        sid = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        mock_baseline.return_value = {
            "ok": True,
            "sessionId": sid,
            "status": "READY",
            "sessionRoot": str(self.sessions_root / sid),
        }

        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("sessionId"), sid)
        self.assertTrue((self.sessions_root / ACTIVE_CYCLE_FILENAME).is_file())

    def test_start_blocked_when_open_exists(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json().get("error"), "ace_dev_cycle_open_exists")

    def test_start_requires_confirm(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.post("/api/local-dev/dev-cycle/start", json={"confirmStart": "NOPE"})
        self.assertEqual(resp.status_code, 400)

    @patch("engine.ace_local_dev_cancel_execute.is_real_ace_cancel_execute_guarded", return_value=True)
    def test_cancel_execute_real_layout_blocked_without_env(self, _mock_guard) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": "a" * 64},
        )
        self.assertEqual(resp.status_code, 403, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertEqual(body.get("error"), "ace_cancel_execute_not_allowed")

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_cancel_execute_does_not_allow_real_without_env(self, mock_preview, mock_exec) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        mock_exec.return_value = {"ok": True, "result": {"status": "SUCCESS"}}

        self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        mock_exec.assert_called_once()
        self.assertFalse(mock_exec.call_args.kwargs.get("allow_real_execute"))

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_cancel_execute_success(self, mock_preview, mock_exec) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        mock_exec.return_value = {"ok": True, "result": {"status": "SUCCESS"}}

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get("ok"))
        mock_exec.assert_called_once()

    def test_cancel_without_confirm(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "NO", "previewFingerprint": "a" * 64},
        )
        self.assertEqual(resp.status_code, 400)

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_cancel_stale_fingerprint(self, mock_preview) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        mock_preview.return_value = _sample_preview_raw(sid)

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": "0" * 64},
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json().get("error"), "ace_cancel_preview_fingerprint_stale")

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_cancel_with_blockers(self, mock_preview) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid, can_execute=False, blockers=["REGISTRY_STALE"])
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json().get("error"), "ace_cancel_execute_blocked")

    def test_cancel_after_cancelled(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid, cycle_status="CANCELLED")
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": "f" * 64},
        )
        self.assertEqual(resp.status_code, 409)

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_parallel_execute_second_rejected(self, mock_preview, mock_exec) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)

        from engine.ace_local_dev_execute_guard import try_begin_local_dev_execute

        try_begin_local_dev_execute()
        try:
            resp = self.client.post(
                "/api/local-dev/cancel-development/execute",
                json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
            )
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.get_json().get("error"), "ace_local_dev_execute_in_progress")
        finally:
            from engine.ace_local_dev_execute_guard import end_local_dev_execute

            end_local_dev_execute()

    def test_rejects_session_id_in_body(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={
                "confirmExecute": "CANCEL",
                "previewFingerprint": "a" * 64,
                "sessionId": "evil",
            },
        )
        self.assertEqual(resp.status_code, 400)

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_cancel_execute_failure_no_pointer_refresh(self, mock_preview, mock_exec) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        mock_exec.return_value = {"ok": False, "error": "PLAN_BLOCKED:SIMULATED", "recovery": None}

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp},
        )
        self.assertEqual(resp.status_code, 503)
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "OPEN")

    def test_non_loopback_blocked(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.post(
            "/api/local-dev/dev-cycle/start",
            json={"confirmStart": "START"},
            environ_overrides={"REMOTE_ADDR": "203.0.113.50"},
        )
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
