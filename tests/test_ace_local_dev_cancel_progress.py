"""
Cancel Execute durable progress — unit and integration tests.

Run: python -m unittest tests.test_ace_local_dev_cancel_progress -v
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import subprocess

import app as app_module
from engine.ace_local_dev_powershell import run_cancel_execute_json
from engine.ace_local_dev_cancel_progress import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    STEP_BACKUP_CREATION,
    STEP_POST_CANCEL_AUTO_START,
    STEP_PRE_CHECKS,
    CancelExecuteProgressWriter,
    assert_operation_id_available,
    find_progress_file,
    load_public_progress,
    new_operation_id,
    public_progress_payload,
    resolve_operation_id,
    sanitize_progress_error,
)
from engine.ace_local_dev_config import (
    confirm_local_dev_loopback_bind,
    enable_local_dev_security_bypass_for_tests,
    reset_local_dev_config_for_tests,
)
from engine.ace_local_dev_cycle_store import write_active_dev_cycle_pointer
from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint
from tests.test_ace_local_dev_cycle import _session_fixture
from tests.test_ace_local_dev_start_cancel_api import _sample_preview_raw


class CancelProgressWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-cancel-progress-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        self.session_id = "ori-20261023-010342-2d952e88"
        (self.sessions_root / self.session_id).mkdir(parents=True)

    def tearDown(self) -> None:
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_atomic_write_and_step_lifecycle(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.begin_step(STEP_PRE_CHECKS)
        writer.complete_step(STEP_PRE_CHECKS)
        writer.begin_step(STEP_BACKUP_CREATION)
        raw = json.loads(writer.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["operationId"], op)
        self.assertEqual(raw["cancelStatus"], STATUS_IN_PROGRESS)
        self.assertEqual(raw["currentStep"]["id"], STEP_BACKUP_CREATION)
        self.assertEqual(len(raw["completedSteps"]), 1)

    def test_find_by_operation_id(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        found = find_progress_file(op, sessions_root=self.sessions_root)
        self.assertEqual(found, writer.path)

    def test_operation_id_reuse_rejected(self) -> None:
        op = new_operation_id()
        CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        with self.assertRaises(ValueError) as ctx:
            assert_operation_id_available(op, session_id=self.session_id, sessions_root=self.sessions_root)
        self.assertEqual(str(ctx.exception), "ace_cancel_progress_operation_id_already_used")

    def test_client_operation_id_accepted(self) -> None:
        op = uuid.uuid4().hex
        resolved = resolve_operation_id(op)
        self.assertEqual(resolved, op)

    def test_cancel_completed_auto_start_failed(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.set_cancel_status(STATUS_COMPLETED)
        writer.begin_post_cancel_auto_start()
        writer.apply_post_cancel_auto_start_result(
            {"ok": False, "phase": "FAILED", "error": "ace_post_cancel_auto_start_failed", "message": "simulated"}
        )
        raw = json.loads(writer.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["cancelStatus"], STATUS_COMPLETED)
        self.assertEqual(raw["operationStatus"], STATUS_FAILED)
        self.assertEqual(raw["postCancelAutoStart"]["phase"], STATUS_FAILED)

    def test_full_success_sets_operation_completed(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.set_cancel_status(STATUS_COMPLETED)
        writer.begin_post_cancel_auto_start()
        writer.apply_post_cancel_auto_start_result(
            {"ok": True, "phase": "COMPLETED", "newSessionId": "ori-20261024-120000-aabbccdd"}
        )
        raw = json.loads(writer.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["operationStatus"], STATUS_COMPLETED)
        self.assertEqual(raw["postCancelAutoStart"]["phase"], STATUS_COMPLETED)

    def test_sanitize_error_strips_paths(self) -> None:
        msg = sanitize_progress_error(r"FAILED D:\secret\_backups\cancel-dev-backups\x")
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertNotIn("D:\\", msg)
        self.assertIn("[path]", msg)

    def test_powershell_progress_step_hook_updates_file(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.complete_step(STEP_PRE_CHECKS)
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        lib = scripts / "OriDevCancelDevLib.ps1"
        ps_cmd = (
            f". '{lib}' ; "
            f"$env:ACE_CANCEL_EXECUTE_PROGRESS_PATH = '{writer.path}' ; "
            f"$env:ACE_CANCEL_EXECUTE_OPERATION_ID = '{op}' ; "
            "Start-OriCancelExecuteProgressStep -StepId 'backup_creation' ; "
            "Complete-OriCancelExecuteProgressStep -StepId 'backup_creation'"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, (completed.stderr or completed.stdout)[:500])
        raw = json.loads(writer.path.read_text(encoding="utf-8"))
        step_ids = {s["id"] for s in raw.get("completedSteps") or []}
        self.assertIn(STEP_PRE_CHECKS, step_ids)
        self.assertIn(STEP_BACKUP_CREATION, step_ids)

    def test_public_payload_includes_cancel_and_auto_start(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.set_cancel_status(STATUS_COMPLETED)
        payload = public_progress_payload(json.loads(writer.path.read_text(encoding="utf-8")))
        self.assertIn("cancelStatus", payload)
        self.assertIn("postCancelAutoStart", payload)


class CancelProgressApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        reset_local_dev_config_for_tests()
        self._tmp = tempfile.mkdtemp(prefix="ace-cancel-progress-api-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        self.session_id = "ori-20261023-010342-2d952e88"
        (self.sessions_root / self.session_id).mkdir(parents=True)
        enable_local_dev_security_bypass_for_tests()
        confirm_local_dev_loopback_bind("127.0.0.1")

    def tearDown(self) -> None:
        reset_local_dev_config_for_tests()
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_progress_get_requires_operation_id(self) -> None:
        resp = self.client.get("/api/local-dev/cancel-development/progress")
        self.assertEqual(resp.status_code, 400)

    def test_progress_get_returns_public_payload(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.begin_step(STEP_PRE_CHECKS)
        writer.complete_step(STEP_PRE_CHECKS)

        resp = self.client.get(
            f"/api/local-dev/cancel-development/progress?operationId={op}&sessionId={self.session_id}"
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body.get("ok"))
        self.assertEqual(body.get("operationId"), op)
        self.assertEqual(body.get("sessionId"), self.session_id)
        self.assertNotIn("backupRoot", body)

    def test_progress_not_found(self) -> None:
        op = new_operation_id()
        resp = self.client.get(f"/api/local-dev/cancel-development/progress?operationId={op}")
        self.assertEqual(resp.status_code, 404)

    def test_parallel_progress_reads(self) -> None:
        op = new_operation_id()
        writer = CancelExecuteProgressWriter(
            session_id=self.session_id,
            operation_id=op,
            sessions_root=self.sessions_root,
        )
        writer.begin_step(STEP_PRE_CHECKS)

        def _get() -> int:
            resp = self.client.get(
                f"/api/local-dev/cancel-development/progress?operationId={op}&sessionId={self.session_id}"
            )
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            codes = list(pool.map(lambda _: _get(), range(16)))
        self.assertTrue(all(c == 200 for c in codes))

    @patch("engine.ace_local_dev_cancel_execute.run_post_cancel_auto_start")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_execute_accepts_client_operation_id(
        self, mock_preview, mock_exec, mock_auto_start
    ) -> None:
        sid = self.session_id
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        mock_exec.return_value = {"ok": True, "result": {"status": "SUCCESS"}}
        mock_auto_start.return_value = {
            "ok": True,
            "phase": "COMPLETED",
            "newSessionId": "ori-20261024-120000-aabbccdd",
        }
        client_op = uuid.uuid4().hex

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp, "operationId": client_op},
        )
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        self.assertEqual(resp.get_json().get("operationId"), client_op)
        self.assertTrue(find_progress_file(client_op, sessions_root=self.sessions_root))

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_duplicate_operation_id_rejected(self, mock_preview) -> None:
        sid = self.session_id
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        dup = uuid.uuid4().hex
        CancelExecuteProgressWriter(session_id=sid, operation_id=dup, sessions_root=self.sessions_root)

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp, "operationId": dup},
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json().get("error"), "ace_cancel_progress_operation_id_already_used")

    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_stale_fingerprint_returns_operation_id(self, mock_preview) -> None:
        sid = self.session_id
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        mock_preview.return_value = _sample_preview_raw(sid)
        client_op = uuid.uuid4().hex

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={
                "confirmExecute": "CANCEL",
                "previewFingerprint": "0" * 64,
                "operationId": client_op,
            },
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json().get("operationId"), client_op)
        prog = load_public_progress(client_op, sessions_root=self.sessions_root)
        self.assertIsNotNone(prog)
        assert prog is not None
        self.assertEqual(prog.get("operationStatus"), STATUS_FAILED)

    @patch("engine.ace_local_dev_cancel_execute.run_post_cancel_auto_start")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_execute_json")
    @patch("engine.ace_local_dev_cancel_execute.run_cancel_dev_preview_powershell")
    def test_auto_start_failure_progress_distinction(
        self, mock_preview, mock_exec, mock_auto_start
    ) -> None:
        sid = self.session_id
        _session_fixture(self.sessions_root, sid)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=self.sessions_root)
        raw = _sample_preview_raw(sid)
        mock_preview.return_value = raw
        fp = compute_preview_fingerprint(raw)
        client_op = uuid.uuid4().hex
        mock_exec.return_value = {"ok": True, "result": {"status": "SUCCESS"}}
        mock_auto_start.return_value = {
            "ok": False,
            "phase": "FAILED",
            "error": "ace_post_cancel_auto_start_failed",
            "message": "simulated",
        }

        resp = self.client.post(
            "/api/local-dev/cancel-development/execute",
            json={"confirmExecute": "CANCEL", "previewFingerprint": fp, "operationId": client_op},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get("ok"))
        prog = load_public_progress(client_op, sessions_root=self.sessions_root)
        assert prog is not None
        self.assertEqual(prog.get("cancelStatus"), STATUS_COMPLETED)
        self.assertEqual(prog.get("operationStatus"), STATUS_FAILED)
        steps = [s["id"] for s in prog.get("completedSteps") or []]
        self.assertIn(STEP_POST_CANCEL_AUTO_START, steps)

    @patch("engine.ace_local_dev_powershell.subprocess.run")
    def test_run_cancel_execute_passes_progress_args(self, mock_run) -> None:
        mock_run.return_value = type(
            "R",
            (),
            {"returncode": 0, "stdout": '{"ok":true,"result":{"status":"SUCCESS"}}', "stderr": ""},
        )()
        op = "a" * 32
        path = str(self.sessions_root / self.session_id / f"cancel-execute-progress-{op}.json")
        run_cancel_execute_json(
            self.session_id,
            progress_path=path,
            operation_id=op,
        )
        cmd = mock_run.call_args[0][0]
        self.assertIn("-ProgressPath", cmd)
        self.assertIn(path, cmd)
        self.assertIn("-ProgressOperationId", cmd)
        self.assertIn(op, cmd)

    def test_non_loopback_progress_blocked(self) -> None:
        op = new_operation_id()
        resp = self.client.get(
            f"/api/local-dev/cancel-development/progress?operationId={op}",
            environ_overrides={"REMOTE_ADDR": "192.168.1.1"},
        )
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
