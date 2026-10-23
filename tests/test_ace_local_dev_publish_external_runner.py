"""
External PUBLISH simulation runner — integration tests (no network).

Run: python -m unittest tests.test_ace_local_dev_publish_external_runner -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.ace_local_dev_publish_candidate_prechecks import PublishPrecheckResult
from engine.ace_local_dev_publish_execute_progress import load_public_publish_execute_progress
from engine.ace_local_dev_publish_external_runner import (
    PUBLISH_RUNNER_MODE_SIMULATION,
    PublishExternalRunnerError,
    _RunnerSessionLock,
    assert_no_forbidden_credentials_in_env,
    assert_simulation_only_mode,
    run_external_publish_simulation,
)
from engine.ace_local_dev_publish_render_sim_transport import PublishRenderSimState
from engine.ace_local_dev_publish_runner_cli import main as runner_cli_main
from engine.ace_local_dev_publish_uri_candidate_execute import (
    PublishExecuteRejected,
    execute_publish_uri_candidate,
)
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_candidate_preview_fingerprint
from tests.test_ace_local_dev_publish_uri_candidate import _raw
from tests.test_ace_local_dev_publish_uri_candidate_execute import CAND, OP, SID
from tests.test_ace_local_dev_publish_uri_candidate_prepare import _ready_plan

REGISTRY_REAL = Path(__file__).resolve().parent.parent.parent / "_registry" / "production-registry.json"


def _precheck_bundle() -> PublishPrecheckResult:
    raw = _raw(_ready_plan(SID))
    plan = raw["plan"]
    return PublishPrecheckResult(
        session_id=SID,
        candidate_backend_sha=CAND,
        raw_preview=raw,
        preview_public={"ok": True},
        plan=plan,
        readiness={**plan["readiness"], "hasFrontendProductChanges": False},
        fingerprint=compute_publish_uri_candidate_preview_fingerprint(raw),
        execution_plan={
            "verifiedSources": {"candidateBackendSha": CAND},
            "allowedProductFiles": {"backend": plan["proposedChanges"]["backend"]["files"], "frontend": []},
            "simulationOnly": False,
        },
    )


class ExternalRunnerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-ext-runner-")
        self.sessions_root = Path(self._tmp) / "sessions"
        self.sessions_root.mkdir(parents=True)
        self.session_dir = self.sessions_root / SID
        self.session_dir.mkdir()
        (self.session_dir / "cancel-dev-cycle.state.json").write_text(
            json.dumps({"status": "OPEN"}),
            encoding="utf-8",
        )
        (self.sessions_root / "active-dev-cycle.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "sessionId": SID,
                    "cycleStatus": "OPEN",
                    "baselineReady": True,
                }
            ),
            encoding="utf-8",
        )
        self.registry_path = Path(self._tmp) / "publish-candidate-registry.json"
        self.registry_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "sessionId": SID,
                    "candidateBackendSha": CAND,
                    "productionBackendBaselineSha": "fadbddd4abb672fd2f4c436f2afddeeff5ea2f67",
                    "status": "ACTIVE",
                }
            ),
            encoding="utf-8",
        )
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ["ACE_PUBLISH_CANDIDATE_REGISTRY_PATH"] = str(self.registry_path)
        self._registry_mtime = REGISTRY_REAL.stat().st_mtime if REGISTRY_REAL.is_file() else 0
        self._env_backup = {
            k: os.environ.get(k)
            for k in ("ACE_RENDER_API_KEY", "RENDER_API_KEY", "GITHUB_TOKEN")
        }
        for k in self._env_backup:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_PUBLISH_CANDIDATE_REGISTRY_PATH", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run_sim(self, operation_id: str = OP, *, patch_candidate_validate: bool = True, **kwargs):
        bundle = _precheck_bundle()
        patches = [
            patch(
                "engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks",
                return_value=bundle,
            ),
            patch(
                "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
                return_value=(SID, CAND, {}),
            ),
        ]
        if patch_candidate_validate:
            patches.append(
                patch(
                    "engine.ace_local_dev_publish_external_runner.validate_active_publish_candidate",
                    return_value=CAND,
                )
            )
        for p in patches:
            p.start()
        try:
            return run_external_publish_simulation(
                session_id=SID,
                operation_id=operation_id,
                publish_uri_preview_fingerprint=bundle.fingerprint,
                sessions_root=self.sessions_root,
                **kwargs,
            )
        finally:
            for p in patches:
                p.stop()


class ExternalRunnerTests(ExternalRunnerTestBase):
    def test_simulation_success(self) -> None:
        out = self._run_sim()
        self.assertTrue(out.get("ok"))
        prog = load_public_publish_execute_progress(OP, session_id_hint=SID, sessions_root=self.sessions_root)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "COMPLETED")
        journal = self.session_dir / f"publish-external-runner-journal-{OP}.json"
        self.assertTrue(journal.is_file())

    def test_invalid_operation_id(self) -> None:
        with self.assertRaises(PublishExternalRunnerError) as ctx:
            run_external_publish_simulation(
                session_id=SID,
                operation_id="bad",
                publish_uri_preview_fingerprint="0" * 64,
                sessions_root=self.sessions_root,
            )
        self.assertEqual(ctx.exception.code, "ace_publish_execute_operation_id_invalid")

    def test_session_mismatch_active_cycle(self) -> None:
        (self.sessions_root / "active-dev-cycle.json").write_text(
            json.dumps({"sessionId": "ori-20261024-999999-99999999", "cycleStatus": "OPEN"}),
            encoding="utf-8",
        )
        with self.assertRaises(PublishExternalRunnerError) as ctx:
            self._run_sim()
        self.assertEqual(ctx.exception.code, "ace_publish_external_runner_session_not_active")

    def test_candidate_not_active(self) -> None:
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        data["status"] = "STALE"
        self.registry_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(PublishExternalRunnerError) as ctx:
            self._run_sim(patch_candidate_validate=False)
        self.assertEqual(ctx.exception.code, "ace_publish_candidate_registry_stale")

    def test_fingerprint_stale_via_execute(self) -> None:
        bundle = _precheck_bundle()
        with patch(
            "engine.ace_local_dev_publish_external_runner.validate_active_publish_candidate",
            return_value=CAND,
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks",
            side_effect=__import__(
                "engine.ace_local_dev_publish_candidate_prechecks",
                fromlist=["PublishPrecheckRejected"],
            ).PublishPrecheckRejected("ace_publish_uri_candidate_fingerprint_stale"),
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
            return_value=(SID, CAND, {}),
        ):
            with self.assertRaises(PublishExternalRunnerError) as ctx:
                run_external_publish_simulation(
                    session_id=SID,
                    operation_id="d" * 32,
                    publish_uri_preview_fingerprint=bundle.fingerprint,
                    sessions_root=self.sessions_root,
                )
        self.assertEqual(ctx.exception.code, "ace_publish_uri_candidate_fingerprint_stale")

    def test_duplicate_operation_id(self) -> None:
        self._run_sim(operation_id="e" * 32)
        with self.assertRaises(PublishExternalRunnerError) as ctx:
            self._run_sim(operation_id="e" * 32)
        self.assertEqual(ctx.exception.code, "ace_publish_execute_operation_id_already_used")

    def test_parallel_runner_blocked(self) -> None:
        lock = _RunnerSessionLock(self.session_dir, "a" * 32)
        lock.acquire()
        try:
            with self.assertRaises(PublishExternalRunnerError) as ctx:
                self._run_sim(operation_id="b" * 32)
            self.assertEqual(ctx.exception.code, "ace_publish_external_runner_session_busy")
        finally:
            lock.release()

    def test_interrupted_lock_released(self) -> None:
        state = PublishRenderSimState()
        state.fail_web = True
        with self.assertRaises(PublishExternalRunnerError):
            self._run_sim(operation_id="c" * 32, sim_state=state)
        self.assertFalse((self.session_dir / "publish-external-runner.lock").is_file())

    def test_render_sim_failure_partial(self) -> None:
        state = PublishRenderSimState()
        state.fail_web = True
        with self.assertRaises(PublishExternalRunnerError):
            self._run_sim(operation_id="1" * 32, sim_state=state)
        prog = load_public_publish_execute_progress("1" * 32, session_id_hint=SID, sessions_root=self.sessions_root)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "PARTIAL")

    def test_precheck_failure_not_partial(self) -> None:
        bundle = _precheck_bundle()
        with patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks",
            side_effect=__import__(
                "engine.ace_local_dev_publish_candidate_prechecks",
                fromlist=["PublishPrecheckRejected"],
            ).PublishPrecheckRejected("ace_dev_cycle_not_open"),
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
            return_value=(SID, CAND, {}),
        ):
            with self.assertRaises(PublishExecuteRejected):
                execute_publish_uri_candidate(
                    confirm_execute="PUBLISH",
                    publish_uri_preview_fingerprint=bundle.fingerprint,
                    operation_id="2" * 32,
                )
        prog = load_public_publish_execute_progress("2" * 32, session_id_hint=SID, sessions_root=self.sessions_root)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "FAILED")

    def test_cli_rejects_real_mode(self) -> None:
        rc = runner_cli_main(
            [
                "--session-id",
                SID,
                "--operation-id",
                OP,
                "--publish-uri-preview-fingerprint",
                "0" * 64,
                "--mode",
                "REAL",
            ]
        )
        self.assertEqual(rc, 1)

    def test_forbidden_credentials_env(self) -> None:
        os.environ["ACE_RENDER_API_KEY"] = "must-not-use"
        with self.assertRaises(PublishExternalRunnerError) as ctx:
            assert_no_forbidden_credentials_in_env()
        self.assertEqual(ctx.exception.code, "ace_publish_external_runner_credentials_forbidden")

    def test_registry_and_git_unchanged(self) -> None:
        self._run_sim(operation_id="3" * 32)
        if REGISTRY_REAL.is_file():
            self.assertEqual(REGISTRY_REAL.stat().st_mtime, self._registry_mtime)

    def test_progress_persisted_after_success(self) -> None:
        self._run_sim(operation_id="4" * 32)
        path = self.sessions_root / SID / f"publish-execute-progress-{'4' * 32}.json"
        self.assertTrue(path.is_file())
        reloaded = load_public_publish_execute_progress("4" * 32, session_id_hint=SID, sessions_root=self.sessions_root)
        assert reloaded is not None
        self.assertEqual(reloaded["operationStatus"], "COMPLETED")

    def test_flask_local_execute_still_blocked(self) -> None:
        from engine.ace_local_dev_publish_execute_adapters import BlockedPublishExecuteAdapters

        bundle = _precheck_bundle()
        with patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted",
            return_value=False,
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
            return_value=(SID, CAND, {}),
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks",
            return_value=bundle,
        ):
            with self.assertRaises(PublishExecuteRejected) as ctx:
                execute_publish_uri_candidate(
                    confirm_execute="PUBLISH",
                    publish_uri_preview_fingerprint=bundle.fingerprint,
                    operation_id="5" * 32,
                    adapters=BlockedPublishExecuteAdapters(),
                )
        self.assertEqual(ctx.exception.code, "ace_publish_execute_not_allowed")

    def test_assert_simulation_mode_only(self) -> None:
        with self.assertRaises(PublishExternalRunnerError):
            assert_simulation_only_mode("REAL")


if __name__ == "__main__":
    unittest.main()
