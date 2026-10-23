"""Post-cancel auto Start — isolated store tests (no real Ori layout)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.ace_local_dev_cycle_store import (
    ACTIVE_CYCLE_FILENAME,
    resolve_active_dev_cycle,
    write_active_dev_cycle_pointer,
)
from engine.ace_local_dev_post_cancel_auto_start import (
    PHASE_COMPLETED,
    PHASE_FAILED,
    PHASE_IN_PROGRESS,
    PHASE_PENDING,
    POST_CANCEL_AUTO_START_FILENAME,
    ensure_post_cancel_auto_start_record,
    run_post_cancel_auto_start,
    try_resume_post_cancel_auto_start,
)
from tests.test_ace_local_dev_cycle import _session_fixture


class PostCancelAutoStartTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-post-cancel-auto-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)

    def tearDown(self) -> None:
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _mark_cancelled(self, sid: str) -> None:
        folder = self.sessions_root / sid
        (folder / "cancel-dev-cycle.state.json").write_text(
            json.dumps({"schemaVersion": 1, "status": "CANCELLED"}),
            encoding="utf-8",
        )
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)

    def _fake_start(self, new_sid: str):
        def _inner(*, sessions_root=None):
            root = sessions_root or self.sessions_root
            _session_fixture(root, new_sid, cycle_status="OPEN")
            write_active_dev_cycle_pointer(session_id=new_sid, sessions_root=root)
            return {"ok": True, "sessionId": new_sid, "mode": "START_CYCLE"}

        return _inner

    @patch("engine.ace_local_dev_post_cancel_auto_start.run_start_dev_cycle_core")
    def test_success_opens_new_cycle(self, mock_start) -> None:
        old = "ori-20261023-010342-2d952e88"
        new = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        mock_start.side_effect = self._fake_start(new)

        result = run_post_cancel_auto_start(cancelled_session_id=old, sessions_root=self.sessions_root)
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("newSessionId"), new)

        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "OPEN")
        self.assertEqual(res.session_id, new)
        old_state = json.loads(
            (self.sessions_root / old / "cancel-dev-cycle.state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(old_state.get("status"), "CANCELLED")

    @patch("engine.ace_local_dev_post_cancel_auto_start.run_start_dev_cycle_core")
    def test_idempotent_second_call(self, mock_start) -> None:
        old = "ori-20261023-010342-2d952e88"
        new = "ori-20261024-120000-aabbccdd"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        mock_start.side_effect = self._fake_start(new)

        run_post_cancel_auto_start(cancelled_session_id=old, sessions_root=self.sessions_root)
        run_post_cancel_auto_start(cancelled_session_id=old, sessions_root=self.sessions_root)
        mock_start.assert_called_once()

    @patch("engine.ace_local_dev_post_cancel_auto_start.run_start_dev_cycle_core")
    def test_start_failure_leaves_recovery_state(self, mock_start) -> None:
        old = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        mock_start.side_effect = RuntimeError("ace_dev_cycle_baseline_failed")

        result = run_post_cancel_auto_start(cancelled_session_id=old, sessions_root=self.sessions_root)
        self.assertFalse(result.get("ok"))
        self.assertEqual(result.get("phase"), PHASE_FAILED)

        res = resolve_active_dev_cycle(sessions_root=self.sessions_root)
        self.assertEqual(res.cycle_status, "CANCELLED_AWAITING_NEW_CYCLE")
        record = json.loads(
            (self.sessions_root / old / POST_CANCEL_AUTO_START_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(record.get("phase"), PHASE_FAILED)

    def test_try_resume_idle_without_record(self) -> None:
        old = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        self.assertIsNone(try_resume_post_cancel_auto_start(sessions_root=self.sessions_root))
        self.assertEqual(
            resolve_active_dev_cycle(sessions_root=self.sessions_root).cycle_status,
            "CANCELLED",
        )

    @patch("engine.ace_local_dev_post_cancel_auto_start.run_start_dev_cycle_core")
    def test_try_resume_pending_record(self, mock_start) -> None:
        old = "ori-20261023-010342-2d952e88"
        new = "ori-20261024-120000-ccddeeff"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        (self.sessions_root / old / POST_CANCEL_AUTO_START_FILENAME).write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "cancelledSessionId": old,
                    "phase": PHASE_PENDING,
                    "newSessionId": None,
                    "lastError": None,
                }
            ),
            encoding="utf-8",
        )
        mock_start.side_effect = self._fake_start(new)

        resumed = try_resume_post_cancel_auto_start(sessions_root=self.sessions_root)
        self.assertTrue(resumed and resumed.get("ok"))
        mock_start.assert_called_once()

    @patch("engine.ace_local_dev_post_cancel_auto_start.run_start_dev_cycle_core")
    def test_try_resume_in_progress_record(self, mock_start) -> None:
        old = "ori-20261023-010342-2d952e88"
        new = "ori-20261024-120000-ddeeff00"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        (self.sessions_root / old / "cancel-dev-cycle.state.json").write_text(
            json.dumps({"schemaVersion": 1, "status": "CANCELLED_AWAITING_NEW_CYCLE"}),
            encoding="utf-8",
        )
        (self.sessions_root / old / POST_CANCEL_AUTO_START_FILENAME).write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "cancelledSessionId": old,
                    "phase": PHASE_IN_PROGRESS,
                    "newSessionId": None,
                    "lastError": None,
                }
            ),
            encoding="utf-8",
        )
        mock_start.side_effect = self._fake_start(new)

        resumed = try_resume_post_cancel_auto_start(sessions_root=self.sessions_root)
        self.assertTrue(resumed and resumed.get("ok"))
        mock_start.assert_called_once()

    def test_try_resume_completed_record_no_op(self) -> None:
        old = "ori-20261023-010342-2d952e88"
        new = "ori-20261024-120000-eeff0011"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        _session_fixture(self.sessions_root, new, cycle_status="OPEN")
        write_active_dev_cycle_pointer(session_id=new, sessions_root=self.sessions_root)
        (self.sessions_root / old / POST_CANCEL_AUTO_START_FILENAME).write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "cancelledSessionId": old,
                    "phase": PHASE_COMPLETED,
                    "newSessionId": new,
                    "lastError": None,
                }
            ),
            encoding="utf-8",
        )
        self.assertIsNone(try_resume_post_cancel_auto_start(sessions_root=self.sessions_root))

    def test_try_resume_open_pointer_no_op(self) -> None:
        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        self.assertIsNone(try_resume_post_cancel_auto_start(sessions_root=self.sessions_root))

    def test_ensure_record_idempotent(self) -> None:
        old = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        root = self.sessions_root / old
        first = ensure_post_cancel_auto_start_record(root, old)
        second = ensure_post_cancel_auto_start_record(root, old)
        self.assertEqual(first.get("phase"), PHASE_PENDING)
        self.assertEqual(second.get("phase"), PHASE_PENDING)
        self.assertEqual(
            len(list(root.glob(POST_CANCEL_AUTO_START_FILENAME))),
            1,
        )

    @patch("engine.ace_local_dev_post_cancel_auto_start.run_start_dev_cycle_core")
    def test_resume_after_failed_start(self, mock_start) -> None:
        old = "ori-20261023-010342-2d952e88"
        new = "ori-20261024-120000-bbccdd00"
        _session_fixture(self.sessions_root, old, cycle_status="CANCELLED")
        self._mark_cancelled(old)
        (self.sessions_root / old / POST_CANCEL_AUTO_START_FILENAME).write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "cancelledSessionId": old,
                    "phase": PHASE_FAILED,
                    "newSessionId": None,
                    "lastError": "simulated",
                }
            ),
            encoding="utf-8",
        )
        (self.sessions_root / old / "cancel-dev-cycle.state.json").write_text(
            json.dumps({"status": "CANCELLED_AWAITING_NEW_CYCLE"}),
            encoding="utf-8",
        )
        mock_start.side_effect = self._fake_start(new)

        resumed = try_resume_post_cancel_auto_start(sessions_root=self.sessions_root)
        self.assertTrue(resumed and resumed.get("ok"))
        self.assertEqual(resolve_active_dev_cycle(sessions_root=self.sessions_root).session_id, new)

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_execute_failure_does_not_auto_start(self, mock_preview, mock_exec) -> None:
        from engine.ace_local_dev_cancel_execute import execute_cancel_development
        from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint

        sid = "ori-20261023-010342-2d952e88"
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = {
            "ok": True,
            "plan": {
                "sessionId": sid,
                "canExecute": True,
                "blockers": [],
                "registryInfraDrift": [],
                "cancelCycle": {"status": "OPEN"},
                "infraTargets": {"backendCommit": "a" * 40, "frontendCommit": "b" * 40},
            },
        }
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        mock_exec.return_value = {"ok": False, "error": "PLAN_BLOCKED:test"}

        with patch("engine.ace_local_dev_cancel_execute.run_post_cancel_auto_start") as mock_auto:
            payload = execute_cancel_development(confirm_execute="CANCEL", preview_fingerprint=fp)
            self.assertFalse(payload.get("ok"))
            mock_auto.assert_not_called()


if __name__ == "__main__":
    unittest.main()
