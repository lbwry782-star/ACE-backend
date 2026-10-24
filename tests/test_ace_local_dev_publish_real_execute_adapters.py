"""
RealPublishExecuteAdapters — isolated port mocks only (no GitHub/Render/production registry).

Run: python -m unittest tests.test_ace_local_dev_publish_real_execute_adapters -v
"""
from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from engine.ace_local_dev_publish_candidate_prechecks import PublishPrecheckResult
from engine.ace_local_dev_publish_execute_adapters import PublishExecuteContext
from engine.ace_local_dev_publish_real_execute_adapters import RealPublishExecuteAdapters
from engine.ace_local_dev_publish_real_execute_ports import (
    REAL_PUBLISH_STAGE_NOTES_KEY,
    GitBackendPushOutcome,
    GitHubPagesDeployOutcome,
    ProductionRegistryUpdateOutcome,
    ProductionVerifyOutcome,
    RealPublishExecutePorts,
    RenderDeployOutcome,
    is_production_registry_path,
    missing_prerequisite_stages,
    registry_update_allowed,
)
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_candidate_preview_fingerprint
from tests.test_ace_local_dev_publish_uri_candidate import CAND, _raw
from tests.test_ace_local_dev_publish_uri_candidate_prepare import _ready_plan

SID = "ori-20261024-120000-aabbccdd"


def _bundle(*, frontend: bool = False) -> PublishPrecheckResult:
    raw = _raw(_ready_plan(SID))
    plan = raw["plan"]
    readiness = dict(plan["readiness"])
    if frontend:
        readiness["hasFrontendProductChanges"] = True
    return PublishPrecheckResult(
        session_id=SID,
        candidate_backend_sha=CAND,
        raw_preview=raw,
        preview_public={"ok": True},
        plan=plan,
        readiness=readiness,
        fingerprint=compute_publish_uri_candidate_preview_fingerprint(raw),
        execution_plan={
            "verifiedSources": {
                "candidateBackendSha": CAND,
                "proposedFrontendHead": "a" * 40 if frontend else "",
            },
            "allowedProductFiles": {"backend": [], "frontend": []},
        },
    )


@dataclass
class MockPorts:
    git_calls: int = 0
    web_calls: int = 0
    worker_calls: int = 0
    pages_calls: int = 0
    verify_calls: int = 0
    registry_calls: int = 0
    git_fail: bool = False
    web_fail: bool = False
    worker_fail: bool = False
    pages_fail: bool = False
    verify_fail: bool = False
    registry_fail: bool = False
    git_already_remote: bool = False
    registry_written: List[str] = field(default_factory=list)

    def build(self) -> RealPublishExecutePorts:
        owner = self

        class _Git:
            def push_backend_to_main(self, *, candidate_sha: str, execution_plan: Dict[str, Any]) -> GitBackendPushOutcome:
                owner.git_calls += 1
                if owner.git_fail:
                    return GitBackendPushOutcome(ok=False, code="ace_test_git_push_failed", message="fail")
                if owner.git_already_remote:
                    return GitBackendPushOutcome(
                        ok=True,
                        code="ace_test_git_already_remote",
                        commit_sha=candidate_sha,
                        already_on_remote=True,
                        idempotent=True,
                    )
                return GitBackendPushOutcome(
                    ok=True,
                    code="ace_test_git_pushed",
                    commit_sha=candidate_sha,
                    pushed=True,
                )

        class _Web:
            def wait_for_web_live(self, *, candidate_sha: str) -> RenderDeployOutcome:
                owner.web_calls += 1
                if owner.web_fail:
                    return RenderDeployOutcome(ok=False, code="ace_test_render_web_failed", message="web fail")
                return RenderDeployOutcome(
                    ok=True,
                    deploy_id="dep-web-test",
                    commit_sha=candidate_sha,
                    already_live=False,
                )

        class _Worker:
            def wait_for_worker_live(self, *, candidate_sha: str) -> RenderDeployOutcome:
                owner.worker_calls += 1
                if owner.worker_fail:
                    return RenderDeployOutcome(ok=False, code="ace_test_render_worker_failed", message="worker fail")
                return RenderDeployOutcome(
                    ok=True,
                    deploy_id="dep-worker-test",
                    commit_sha=candidate_sha,
                    observed_only=True,
                )

        class _Pages:
            def deploy_frontend(self, *, frontend_commit_sha: str, execution_plan: Dict[str, Any]) -> GitHubPagesDeployOutcome:
                owner.pages_calls += 1
                if owner.pages_fail:
                    return GitHubPagesDeployOutcome(ok=False, code="ace_test_pages_failed", message="pages fail")
                return GitHubPagesDeployOutcome(
                    ok=True,
                    workflow_run_id="wf-test-1",
                    frontend_commit_sha=frontend_commit_sha or ("b" * 40),
                )

        class _Verify:
            def verify_production(
                self,
                *,
                candidate_backend_sha: str,
                frontend_deploy_required: bool,
                stage_snapshot: Dict[str, Any],
            ) -> ProductionVerifyOutcome:
                owner.verify_calls += 1
                if owner.verify_fail:
                    return ProductionVerifyOutcome(ok=False, code="ace_test_verify_failed", message="verify fail")
                return ProductionVerifyOutcome(
                    ok=True,
                    backend_commit_sha=candidate_backend_sha,
                    frontend_verified=frontend_deploy_required,
                )

        class _Registry:
            def update_production_registry(self, *, ctx, stage_snapshot, verify_outcome) -> ProductionRegistryUpdateOutcome:
                owner.registry_calls += 1
                if owner.registry_fail:
                    return ProductionRegistryUpdateOutcome(ok=False, code="ace_test_registry_failed", message="registry fail")
                path = str(Path(tempfile.gettempdir()) / "ace-publish-real-test" / "registry.json")
                owner.registry_written.append(path)
                return ProductionRegistryUpdateOutcome(ok=True, written_path=path)

        return RealPublishExecutePorts(
            git_backend_push=_Git(),
            render_web=_Web(),
            render_worker=_Worker(),
            github_pages=_Pages(),
            production_verify=_Verify(),
            registry_update=_Registry(),
        )


class RealPublishExecuteAdapterTests(unittest.TestCase):
    def test_full_backend_success_via_ports(self) -> None:
        mocks = MockPorts()
        adapters = RealPublishExecuteAdapters(mocks.build())
        pre = _bundle()
        ctx = PublishExecuteContext(precheck=pre, operation_id="op1")
        self.assertTrue(adapters.prepare_isolated_release(ctx).ok)
        self.assertTrue(adapters.git_push_backend(ctx).ok)
        self.assertTrue(adapters.wait_render_web(ctx).ok)
        self.assertTrue(adapters.wait_render_worker(ctx).ok)
        self.assertTrue(adapters.verify_production(ctx).ok)
        self.assertTrue(adapters.update_production_registry(ctx).ok)
        self.assertEqual(mocks.git_calls, 1)
        self.assertEqual(mocks.registry_calls, 1)
        stages = ctx.notes[REAL_PUBLISH_STAGE_NOTES_KEY]
        self.assertTrue(stages["git_push_backend"]["ok"])
        self.assertTrue(stages["update_production_registry"]["ok"])

    def test_git_push_idempotent_no_double_push_requirement(self) -> None:
        mocks = MockPorts(git_already_remote=True)
        adapters = RealPublishExecuteAdapters(mocks.build())
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="op2")
        adapters.prepare_isolated_release(ctx)
        res = adapters.git_push_backend(ctx)
        self.assertTrue(res.ok)
        self.assertTrue(res.details.get("idempotent"))
        self.assertEqual(mocks.git_calls, 1)

    def test_render_web_blocked_when_git_failed(self) -> None:
        mocks = MockPorts(git_fail=True)
        adapters = RealPublishExecuteAdapters(mocks.build())
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="op3")
        adapters.prepare_isolated_release(ctx)
        self.assertFalse(adapters.git_push_backend(ctx).ok)
        web = adapters.wait_render_web(ctx)
        self.assertFalse(web.ok)
        self.assertEqual(web.code, "ace_publish_real_render_web_prerequisite")
        self.assertEqual(mocks.web_calls, 0)

    def test_git_ok_render_web_fail_partial(self) -> None:
        mocks = MockPorts(web_fail=True)
        adapters = RealPublishExecuteAdapters(mocks.build())
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="op4")
        adapters.prepare_isolated_release(ctx)
        self.assertTrue(adapters.git_push_backend(ctx).ok)
        self.assertFalse(adapters.wait_render_web(ctx).ok)
        reg = adapters.update_production_registry(ctx)
        self.assertFalse(reg.ok)
        self.assertEqual(reg.code, "ace_publish_real_registry_prerequisites_missing")
        self.assertEqual(mocks.registry_calls, 0)

    def test_worker_fail_blocks_registry(self) -> None:
        mocks = MockPorts(worker_fail=True)
        adapters = RealPublishExecuteAdapters(mocks.build())
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="op5")
        adapters.prepare_isolated_release(ctx)
        adapters.git_push_backend(ctx)
        adapters.wait_render_web(ctx)
        self.assertFalse(adapters.wait_render_worker(ctx).ok)
        self.assertFalse(adapters.update_production_registry(ctx).ok)
        self.assertEqual(mocks.registry_calls, 0)

    def test_frontend_path_requires_pages_before_verify(self) -> None:
        mocks = MockPorts()
        adapters = RealPublishExecuteAdapters(mocks.build())
        pre = _bundle(frontend=True)
        ctx = PublishExecuteContext(precheck=pre, operation_id="op6")
        adapters.prepare_isolated_release(ctx)
        adapters.git_push_backend(ctx)
        adapters.wait_render_web(ctx)
        adapters.wait_render_worker(ctx)
        verify = adapters.verify_production(ctx)
        self.assertFalse(verify.ok)
        self.assertEqual(verify.code, "ace_publish_real_verify_frontend_missing")
        adapters.deploy_github_pages(ctx)
        self.assertTrue(adapters.verify_production(ctx).ok)

    def test_verify_fail_blocks_registry(self) -> None:
        mocks = MockPorts(verify_fail=True)
        adapters = RealPublishExecuteAdapters(mocks.build())
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="op7")
        adapters.prepare_isolated_release(ctx)
        adapters.git_push_backend(ctx)
        adapters.wait_render_web(ctx)
        adapters.wait_render_worker(ctx)
        self.assertFalse(adapters.verify_production(ctx).ok)
        self.assertFalse(adapters.update_production_registry(ctx).ok)
        self.assertEqual(mocks.registry_calls, 0)

    def test_registry_helper_missing_stages(self) -> None:
        ctx = PublishExecuteContext(precheck=_bundle(), operation_id="op8")
        ctx.notes[REAL_PUBLISH_STAGE_NOTES_KEY] = {"git_push_backend": {"ok": True}}
        missing = missing_prerequisite_stages(ctx)
        self.assertIn("wait_render_web", missing)
        blocked = registry_update_allowed(ctx)
        assert blocked is not None
        self.assertFalse(blocked.ok)

    def test_production_registry_path_guard(self) -> None:
        safe = Path(tempfile.gettempdir()) / "ace-publish-real-test" / "production-registry.json"
        self.assertFalse(is_production_registry_path(safe))
        unsafe = Path("D:/ace/_registry/production-registry.json")
        self.assertTrue(is_production_registry_path(unsafe))

    def test_routes_do_not_import_real_adapters(self) -> None:
        routes = importlib.import_module("engine.ace_local_dev_routes")
        source_path = Path(routes.__file__).resolve()
        text = source_path.read_text(encoding="utf-8")
        self.assertNotIn("RealPublishExecuteAdapters", text)
        self.assertNotIn("ace_local_dev_publish_real_execute_adapters", text)

    def test_execute_module_default_still_blocked_adapter(self) -> None:
        execute_mod = importlib.import_module("engine.ace_local_dev_publish_uri_candidate_execute")
        text = Path(execute_mod.__file__).resolve().read_text(encoding="utf-8")
        self.assertIn("BlockedPublishExecuteAdapters()", text)
        self.assertNotIn("RealPublishExecuteAdapters", text)

    def test_sim_adapters_module_separate_from_real(self) -> None:
        sim_text = Path(
            importlib.import_module("engine.ace_local_dev_publish_render_sim_execute_adapters").__file__
        ).resolve().read_text(encoding="utf-8")
        self.assertNotIn("RealPublishExecuteAdapters", sim_text)
        self.assertIn("SimulatedRenderPublishExecuteAdapters", sim_text)


if __name__ == "__main__":
    unittest.main()
