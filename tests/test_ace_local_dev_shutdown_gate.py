"""Isolated shutdown gate tests (temp dirs only; no live ACE repos or stack stop)."""
from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.ace_local_dev_cancel_progress import (
    CancelExecuteProgressWriter,
    STEP_BACKEND_RESTORE,
    STEP_BACKUP_CREATION,
    STEP_CLOSE_CANCEL_CYCLE,
    STEP_FRONTEND_RESTORE,
    STEP_PRE_CHECKS,
)
from engine.ace_local_dev_execute_guard import (
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_publish_execute_progress import (
    PublishExecuteProgressWriter,
    STEP_GIT_PUSH_BACKEND,
    STEP_GITHUB_PAGES,
    STEP_PRODUCTION_VERIFY,
    STEP_REGISTRY_UPDATE,
)
from engine.ace_local_dev_publish_external_effects_journal import (
    PublishExternalEffectsJournalWriter,
    STAGE_GITHUB_PAGES,
    STAGE_RUNNING,
    STAGE_SUCCEEDED,
    STAGE_UNKNOWN,
    load_publish_external_effects_journal,
)
from engine.ace_local_dev_publish_prepare_progress import PublishPrepareProgressWriter
from engine.ace_local_dev_shutdown_gate import (
    GATE_RUNNING,
    GATE_SAFE_TO_STOP,
    GATE_SHUTDOWN_BLOCKED,
    GATE_SHUTDOWN_REQUESTED,
    GATE_WAITING_FOR_OPERATION,
    local_dev_operation_start_allowed,
    observe_active_local_dev_operations,
    read_shutdown_gate_state,
    refresh_local_shutdown_gate,
    request_local_shutdown,
)


def _session_id() -> str:
    return f"ori-20260101-120000-{uuid.uuid4().hex[:8]}"


class ShutdownGateTempLayout:
    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ace-shutdown-gate-")
        self.ace_root = Path(self._tmp.name) / "ACE"
        self.sessions_root = self.ace_root / "_backups" / "sessions"
        self.local_dev = self.ace_root / "_backups" / "local-dev"
        self.local_dev.mkdir(parents=True, exist_ok=True)
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        (self.local_dev / "managed-processes.json").write_text(
            json.dumps({"launcherSession": "20260101-120000"}) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self._tmp.cleanup()

    def session_dir(self, session_id: str) -> Path:
        path = self.sessions_root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path


class TestAceLocalShutdownGate(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = ShutdownGateTempLayout()
        end_local_dev_execute()

    def tearDown(self) -> None:
        end_local_dev_execute()
        self.layout.close()

    def _refresh(self, **kwargs):
        return refresh_local_shutdown_gate(
            self.layout.ace_root,
            launcher_session="20260101-120000",
            sessions_root=self.layout.sessions_root,
            **kwargs,
        )

    def test_shutdown_request_idle_becomes_safe(self) -> None:
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_SAFE_TO_STOP)

    def test_shutdown_during_new_waits(self) -> None:
        sid = _session_id()
        session_dir = self.layout.session_dir(sid)
        (session_dir / "session.json").write_text(
            json.dumps({"status": "IN_PROGRESS", "sessionId": sid}) + "\n",
            encoding="utf-8",
        )
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_cancel_before_git_change_waits_not_safe(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        writer = CancelExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        writer.begin_step(STEP_PRE_CHECKS)
        writer.complete_step(STEP_PRE_CHECKS)
        writer.begin_step(STEP_BACKUP_CREATION)
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)
        obs = state["evaluation"]["scan"]["observations"]
        self.assertTrue(any(o.get("kind") == "CANCEL" for o in obs))

    def test_cancel_mid_backend_restore_waits_critical(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        w.begin_step(STEP_BACKEND_RESTORE)
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)
        cancel_obs = [o for o in state["evaluation"]["scan"]["observations"] if o.get("kind") == "CANCEL"]
        self.assertTrue(cancel_obs and cancel_obs[0].get("cancelCritical"))

    def test_cancel_mid_frontend_restore_critical(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        w.begin_step(STEP_FRONTEND_RESTORE)
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_cancel_after_success_allows_safe(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        w = CancelExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        w.begin_step(STEP_CLOSE_CANCEL_CYCLE)
        w.complete_step(STEP_CLOSE_CANCEL_CYCLE)
        w.finish(operation_status="COMPLETED", cancel_status="COMPLETED")
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_SAFE_TO_STOP)

    def test_publish_prepare_active_waits(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        p = PublishPrepareProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        p.begin_step("fresh_preview")
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_publish_before_git_push_waits(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        ex.begin_step(STEP_GIT_PUSH_BACKEND)
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_publish_git_push_running_waits(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        ex.begin_step(STEP_GIT_PUSH_BACKEND)
        j = PublishExternalEffectsJournalWriter(
            session_id=sid,
            operation_id=op,
            candidate_backend_sha="a" * 40,
            sessions_root=self.layout.sessions_root,
        )
        j.set_stage("git_push_backend", STAGE_RUNNING, evidence={"pushed": False})
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_render_timeout_blocks_unknown(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        ex.begin_step("render_web_deploy")
        old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        journal_path = self.layout.session_dir(sid) / f"publish-external-effects-journal-{op}.json"
        payload = {
            "schemaVersion": 1,
            "operationId": op,
            "sessionId": sid,
            "candidateBackendSha": "b" * 40,
            "stages": {
                "render_web": {
                    "status": STAGE_RUNNING,
                    "startedAt": old,
                    "updatedAt": old,
                    "evidence": {},
                }
            },
        }
        journal_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh(stage_timeout_seconds=30.0)
        self.assertEqual(state["gateStatus"], GATE_SHUTDOWN_BLOCKED)

    def test_github_pages_unknown_blocks(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        ex.begin_step(STEP_GITHUB_PAGES)
        j = PublishExternalEffectsJournalWriter(
            session_id=sid,
            operation_id=op,
            sessions_root=self.layout.sessions_root,
        )
        j.set_stage(STAGE_GITHUB_PAGES, STAGE_UNKNOWN, error="deploy_result_unknown")
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_SHUTDOWN_BLOCKED)

    def test_registry_update_active_waits(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        ex = PublishExecuteProgressWriter(session_id=sid, operation_id=op, sessions_root=self.layout.sessions_root)
        ex.begin_step(STEP_REGISTRY_UPDATE)
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)

    def test_backend_unreachable_blocks(self) -> None:
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh(backend_reachable=False)
        self.assertEqual(state["gateStatus"], GATE_SHUTDOWN_BLOCKED)

    def test_corrupt_progress_blocks(self) -> None:
        sid = _session_id()
        session_dir = self.layout.session_dir(sid)
        bad = session_dir / f"cancel-execute-progress-{uuid.uuid4().hex}.json"
        bad.write_text("{not-json", encoding="utf-8")
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_SHUTDOWN_BLOCKED)

    def test_concurrent_shutdown_requests_block(self) -> None:
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        second = request_local_shutdown(self.layout.ace_root, launcher_session="20260101-999999")
        self.assertEqual(second["gateStatus"], GATE_SHUTDOWN_BLOCKED)

    def test_journal_atomic_survives_read(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        j = PublishExternalEffectsJournalWriter(
            session_id=sid,
            operation_id=op,
            sessions_root=self.layout.sessions_root,
        )
        j.set_stage("git_push_backend", STAGE_SUCCEEDED, evidence={"commitSha": "c" * 40})
        loaded = load_publish_external_effects_journal(op, session_id=sid, sessions_root=self.layout.sessions_root)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["stages"]["git_push_backend"]["status"], STAGE_SUCCEEDED)

    def test_journal_tmp_without_final_leaves_prior_or_missing(self) -> None:
        sid = _session_id()
        op = uuid.uuid4().hex
        session_dir = self.layout.session_dir(sid)
        final_path = session_dir / f"publish-external-effects-journal-{op}.json"
        final_path.write_text(json.dumps({"schemaVersion": 1, "stages": {}}) + "\n", encoding="utf-8")
        tmp = final_path.with_name(f"{final_path.name}.tmp.{uuid.uuid4().hex}")
        tmp.write_text("{", encoding="utf-8")
        loaded = load_publish_external_effects_journal(op, session_id=sid, sessions_root=self.layout.sessions_root)
        self.assertIsNotNone(loaded)

    def test_execute_guard_active_waits(self) -> None:
        try_begin_local_dev_execute()
        try:
            request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
            state = self._refresh()
            self.assertEqual(state["gateStatus"], GATE_WAITING_FOR_OPERATION)
        finally:
            end_local_dev_execute()

    def test_operation_start_blocked_after_shutdown_request(self) -> None:
        request_local_shutdown(self.layout.ace_root, launcher_session="20260101-120000")
        allowed, code = local_dev_operation_start_allowed(
            self.layout.ace_root,
            expected_launcher_session="20260101-120000",
        )
        self.assertFalse(allowed)
        self.assertEqual(code, "ace_shutdown_gate_operations_frozen")

    def test_routes_not_wired(self) -> None:
        routes_path = Path(__file__).resolve().parent.parent / "engine" / "ace_local_dev_routes.py"
        text = routes_path.read_text(encoding="utf-8")
        self.assertNotIn("shutdown_gate", text)
        self.assertNotIn("request_local_shutdown", text)

    def test_idle_gate_running(self) -> None:
        state = self._refresh()
        self.assertEqual(state["gateStatus"], GATE_RUNNING)

    def test_observe_does_not_mutate_repos(self) -> None:
        scan = observe_active_local_dev_operations(
            ace_root=self.layout.ace_root,
            sessions_root=self.layout.sessions_root,
        )
        self.assertIn("observations", scan)
        gate = read_shutdown_gate_state(self.layout.ace_root)
        self.assertTrue(gate is None or gate.get("gateStatus") != GATE_SAFE_TO_STOP)


if __name__ == "__main__":
    unittest.main()
