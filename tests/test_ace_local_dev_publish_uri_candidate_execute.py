"""
Publish Uri Candidate Execute — isolated mocks only (no git/deploy/registry I/O).

Run: python -m unittest tests.test_ace_local_dev_publish_uri_candidate_execute -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional
from unittest.mock import patch

from engine.ace_local_dev_publish_candidate_prechecks import PublishPrecheckResult
from engine.ace_local_dev_publish_execute_adapters import (
    PublishExecuteContext,
    StageResult,
)
from engine.ace_local_dev_publish_execute_progress import (
    STEP_FINGERPRINT,
    STEP_GIT_PUSH_BACKEND,
    STEP_GITHUB_PAGES,
    STEP_PRODUCTION_VERIFY,
    STEP_REAL_EXECUTE_GATE,
    STEP_REGISTRY_UPDATE,
    STEP_RENDER_WEB,
    STEP_RENDER_WORKER,
    load_public_publish_execute_progress,
)
from engine.ace_local_dev_publish_uri_candidate_execute import (
    PublishExecuteRejected,
    execute_publish_uri_candidate,
)
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_candidate_preview_fingerprint
from tests.test_ace_local_dev_publish_uri_candidate import CAND, PROD, _candidate_plan, _raw
from tests.test_ace_local_dev_publish_uri_candidate_prepare import _ready_plan

SID = "ori-20261024-120000-aabbccdd"
OP = "f" * 32


@dataclass
class MockPublishExecuteAdapters:
    fail_at: Optional[str] = None
    calls: Dict[str, int] = field(default_factory=dict)
    inject_local_path: bool = False

    def _tick(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def _maybe_fail(self, stage: str) -> Optional[StageResult]:
        if self.fail_at == stage:
            return StageResult(ok=False, code=f"ace_publish_execute_{stage}_failed", message=stage)
        return None

    def prepare_isolated_release(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("prepare_isolated_release")
        if self.inject_local_path:
            plan = dict(ctx.precheck.execution_plan)
            files = dict(plan.get("allowedProductFiles") or {})
            files["backend"] = list(files.get("backend") or []) + ["scripts/OriPublishUriLib.ps1"]
            plan["allowedProductFiles"] = files
            ctx.precheck.execution_plan = plan
            return StageResult(ok=False, code="ace_publish_execute_local_path_leak", message="leak")
        fail = self._maybe_fail("isolated_release")
        if fail:
            return fail
        return StageResult(ok=True, code="ready")

    def git_push_backend(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("git_push_backend")
        fail = self._maybe_fail("git_push")
        return fail or StageResult(ok=True, code="git_ok")

    def wait_render_web(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("wait_render_web")
        fail = self._maybe_fail("render_web")
        return fail or StageResult(ok=True, code="render_web_ok", details={"commitSha": CAND})

    def wait_render_worker(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("wait_render_worker")
        fail = self._maybe_fail("render_worker")
        if fail:
            return fail
        web_sha = ctx.precheck.candidate_backend_sha
        return StageResult(ok=True, code="render_worker_ok", details={"commitSha": web_sha})

    def deploy_github_pages(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("deploy_github_pages")
        fail = self._maybe_fail("github_pages")
        return fail or StageResult(ok=True, code="pages_ok")

    def verify_production(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("verify_production")
        fail = self._maybe_fail("production_verify")
        return fail or StageResult(ok=True, code="verify_ok")

    def update_production_registry(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("update_production_registry")
        fail = self._maybe_fail("registry_update")
        return fail or StageResult(ok=True, code="registry_ok")


def _precheck_bundle(session_id: str = SID) -> PublishPrecheckResult:
    raw = _raw(_ready_plan(session_id))
    plan = raw["plan"]
    return PublishPrecheckResult(
        session_id=session_id,
        candidate_backend_sha=CAND,
        raw_preview=raw,
        preview_public={"ok": True},
        plan=plan,
        readiness=plan["readiness"],
        fingerprint=compute_publish_uri_candidate_preview_fingerprint(raw),
        execution_plan={
            "verifiedSources": {"candidateBackendSha": CAND},
            "allowedProductFiles": {"backend": plan["proposedChanges"]["backend"]["files"], "frontend": []},
            "simulationOnly": False,
        },
    )


class ExecuteTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-publish-exec-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ["ACE_PUBLISH_CANDIDATE_REGISTRY_PATH"] = str(
            Path(self._tmp) / "publish-candidate-registry.json"
        )
        os.environ["ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE"] = "1"
        os.environ.pop("ACE_ORI_DEV_SESSION_ID", None)

    def tearDown(self) -> None:
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_PUBLISH_CANDIDATE_REGISTRY_PATH", None)
        os.environ.pop("ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE", None)
        shutil.rmtree(self._tmp, ignore_errors=True)
        real = Path(__file__).resolve().parent.parent.parent / "_backups" / "sessions" / SID
        if real.is_dir():
            shutil.rmtree(real, ignore_errors=True)


class PublishExecuteTests(ExecuteTestBase):
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted",
        return_value=True,
    )
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate")
    def test_mock_full_success(self, mock_resolve, mock_pre, _gate) -> None:
        bundle = _precheck_bundle()
        mock_resolve.return_value = (SID, CAND, {})
        mock_pre.return_value = bundle
        fp = bundle.fingerprint
        adapters = MockPublishExecuteAdapters()
        out = execute_publish_uri_candidate(
            confirm_execute="PUBLISH",
            publish_uri_preview_fingerprint=fp,
            operation_id=OP,
            adapters=adapters,
        )
        self.assertTrue(out["ok"])
        prog = load_public_publish_execute_progress(OP, session_id_hint=SID)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "COMPLETED")
        self.assertGreaterEqual(len(prog["completedSteps"]), 10)
        self.assertTrue(all(s.get("uiMessageHe") for s in prog["completedSteps"]))

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate")
    def test_precheck_fingerprint_stale(self, mock_resolve, mock_pre) -> None:
        from engine.ace_local_dev_publish_candidate_prechecks import PublishPrecheckRejected

        mock_resolve.return_value = (SID, CAND, {})
        mock_pre.side_effect = PublishPrecheckRejected("ace_publish_uri_candidate_fingerprint_stale")
        with self.assertRaises(PublishExecuteRejected) as ctx:
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint="0" * 64,
                operation_id="a" * 32,
                adapters=MockPublishExecuteAdapters(),
            )
        self.assertEqual(ctx.exception.code, "ace_publish_uri_candidate_fingerprint_stale")

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate")
    def test_duplicate_operation_id(self, mock_resolve, mock_pre, _gate) -> None:
        bundle = _precheck_bundle()
        mock_resolve.return_value = (SID, CAND, {})
        mock_pre.return_value = bundle
        fp = bundle.fingerprint
        execute_publish_uri_candidate(
            confirm_execute="PUBLISH",
            publish_uri_preview_fingerprint=fp,
            operation_id=OP,
            adapters=MockPublishExecuteAdapters(),
        )
        with self.assertRaises(PublishExecuteRejected) as ctx:
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id=OP,
                adapters=MockPublishExecuteAdapters(),
            )
        self.assertEqual(ctx.exception.code, "ace_publish_execute_operation_id_already_used")

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_git_push_failure_partial(self, *_mocks) -> None:
        _mocks[1].return_value = _precheck_bundle()
        fp = _mocks[1].return_value.fingerprint
        with self.assertRaises(PublishExecuteRejected):
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="b" * 32,
                adapters=MockPublishExecuteAdapters(fail_at="git_push"),
            )
        prog = load_public_publish_execute_progress("b" * 32, session_id_hint=SID)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "PARTIAL")

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_render_web_failure(self, *_mocks) -> None:
        _mocks[1].return_value = _precheck_bundle()
        fp = _mocks[1].return_value.fingerprint
        with self.assertRaises(PublishExecuteRejected):
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="c" * 32,
                adapters=MockPublishExecuteAdapters(fail_at="render_web"),
            )

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_render_worker_failure(self, *_mocks) -> None:
        _mocks[1].return_value = _precheck_bundle()
        fp = _mocks[1].return_value.fingerprint
        with self.assertRaises(PublishExecuteRejected):
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="d" * 32,
                adapters=MockPublishExecuteAdapters(fail_at="render_worker"),
            )

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_github_pages_failure(self, *_mocks) -> None:
        bundle = _precheck_bundle()
        bundle.readiness["hasFrontendProductChanges"] = True
        _mocks[1].return_value = bundle
        fp = bundle.fingerprint
        with self.assertRaises(PublishExecuteRejected):
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="e" * 32,
                adapters=MockPublishExecuteAdapters(fail_at="github_pages"),
            )

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_production_verify_failure(self, *_mocks) -> None:
        _mocks[1].return_value = _precheck_bundle()
        fp = _mocks[1].return_value.fingerprint
        with self.assertRaises(PublishExecuteRejected):
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="9" * 32,
                adapters=MockPublishExecuteAdapters(fail_at="production_verify"),
            )

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_registry_update_failure(self, *_mocks) -> None:
        _mocks[1].return_value = _precheck_bundle()
        fp = _mocks[1].return_value.fingerprint
        with self.assertRaises(PublishExecuteRejected):
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="8" * 32,
                adapters=MockPublishExecuteAdapters(fail_at="registry_update"),
            )

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate")
    def test_real_execute_gate_blocks_local(self, mock_resolve, mock_pre) -> None:
        bundle = _precheck_bundle()
        mock_resolve.return_value = (SID, CAND, {})
        mock_pre.return_value = bundle
        fp = bundle.fingerprint
        os.environ.pop("ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE", None)
        with self.assertRaises(PublishExecuteRejected) as ctx:
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="7" * 32,
                adapters=MockPublishExecuteAdapters(),
            )
        self.assertEqual(ctx.exception.code, "ace_publish_execute_not_allowed")

    @patch("engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted", return_value=True)
    @patch("engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks")
    @patch(
        "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
        return_value=(SID, CAND, {}),
    )
    def test_local_path_leak_blocked(self, *_mocks) -> None:
        _mocks[1].return_value = _precheck_bundle()
        fp = _mocks[1].return_value.fingerprint
        with self.assertRaises(PublishExecuteRejected) as ctx:
            execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=fp,
                operation_id="6" * 32,
                adapters=MockPublishExecuteAdapters(inject_local_path=True),
            )
        self.assertIn("local_path_leak", ctx.exception.code)


if __name__ == "__main__":
    unittest.main()
