"""
Active dev cycle pointer + status API — isolated TEMP tests.

Run: python -m unittest tests.test_ace_local_dev_cycle -v
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


def _session_fixture(
    sessions_root: Path,
    session_id: str,
    *,
    ready: bool = True,
    cycle_status: str = "OPEN",
    include_session_json: bool = True,
) -> Path:
    folder = sessions_root / session_id
    folder.mkdir(parents=True, exist_ok=True)
    if include_session_json:
        (folder / "session.json").write_text(
            json.dumps({"status": "READY" if ready else "PENDING"}),
            encoding="utf-8",
        )
    if cycle_status:
        (folder / "cancel-dev-cycle.state.json").write_text(
            json.dumps({"status": cycle_status}),
            encoding="utf-8",
        )
    return folder


class ActiveDevCycleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-cycle-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_open_cycle_via_pointer(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "OPEN")
        self.assertEqual(res.session_id, sid)
        self.assertTrue(res.baseline_ready)
        self.assertEqual(res.source, "pointer")
        self.assertTrue(res.pointer_valid)

    def test_env_fallback_when_no_pointer(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        os.environ["ACE_ORI_DEV_SESSION_ID"] = sid
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.source, "env_fallback")
        self.assertEqual(res.session_id, sid)

    def test_cancelled_cycle(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid, cycle_status="CANCELLED")
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "CANCELLED")

    def test_missing_pointer_none(self) -> None:
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "NONE")

    def test_corrupt_pointer(self) -> None:
        path = self.sessions_root / ACTIVE_CYCLE_FILENAME
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        path.write_text("{not-json", encoding="utf-8")
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "INVALID")
        self.assertIn("ACTIVE_CYCLE_POINTER_INVALID", res.blockers)

    def test_missing_session_json(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid, include_session_json=False)
        (self.sessions_root / ACTIVE_CYCLE_FILENAME).write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "sessionId": sid,
                    "cycleStatus": "OPEN",
                    "baselineReady": True,
                }
            ),
            encoding="utf-8",
        )
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "INVALID")

    def test_multiple_open_cycles(self) -> None:
        a = "ori-20261023-010342-2d952e88"
        b = "ori-20261024-020000-aabbccdd"
        _session_fixture(self.sessions_root, a)
        _session_fixture(self.sessions_root, b)
        (self.sessions_root / ACTIVE_CYCLE_FILENAME).write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "sessionId": a,
                    "cycleStatus": "OPEN",
                    "baselineReady": True,
                }
            ),
            encoding="utf-8",
        )
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "INVALID")
        self.assertIn("MULTIPLE_OPEN_CYCLES", res.blockers)

    def test_simulated_restart_reads_pointer_without_env(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        os.environ["ACE_ORI_DEV_SESSION_ID"] = sid
        first = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)
        second = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(first.session_id, second.session_id)
        self.assertEqual(second.source, "pointer")

    def test_pointer_wins_over_env(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        other = "ori-20261024-020000-aabbccdd"
        _session_fixture(self.sessions_root, sid)
        _session_fixture(self.sessions_root, other, cycle_status="CANCELLED")
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        os.environ["ACE_ORI_DEV_SESSION_ID"] = other
        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.session_id, sid)
        self.assertEqual(res.source, "pointer")


class ActiveDevCycleStatusApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-cycle-api-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_status_requires_loopback(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        resp = self.client.get(
            "/api/local-dev/dev-cycle/status",
            environ_overrides={"REMOTE_ADDR": "203.0.113.50"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_status_open_pointer(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        resp = self.client.get("/api/local-dev/dev-cycle/status")
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body["activeCycle"]["status"], "OPEN")
        self.assertFalse(body.get("canStartNewCycle"))
        text = json.dumps(body)
        self.assertNotIn("ACE_ORI", text)

    @patch("engine.ace_local_dev_routes.run_cancel_dev_preview_powershell")
    def test_preview_with_pointer_no_env(self, mock_run) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        mock_run.return_value = {"ok": True, "plan": {"sessionId": sid, "cancelCycle": {"status": "OPEN"}, "blockers": [], "canExecute": True, "commitsToCancel": {"backend": [], "frontend": []}, "git": {"backend": {}, "frontend": {}}}}
        resp = self.client.get("/api/local-dev/cancel-development/preview")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get("ok"))
        mock_run.assert_called_once_with(sid)

    def test_preview_invalid_pointer_blocks(self) -> None:
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")
        (self.sessions_root / ACTIVE_CYCLE_FILENAME).write_text("{}", encoding="utf-8")
        resp = self.client.get("/api/local-dev/cancel-development/preview")
        body = resp.get_json()
        self.assertFalse(body.get("ok"))
        self.assertEqual(body.get("error"), "ace_active_dev_cycle_invalid")


if __name__ == "__main__":
    unittest.main()
