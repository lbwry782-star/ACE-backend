"""
Publish Execute + Render API simulation — integration tests (mock transport only).

Run: python -m unittest tests.test_ace_local_dev_publish_render_sim_execute -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from urllib.parse import urlparse

from engine.ace_local_dev_publish_candidate_prechecks import PublishPrecheckResult
from engine.ace_local_dev_publish_execute_adapters import StageResult
from engine.ace_local_dev_publish_execute_progress import (
    STEP_GIT_PUSH_BACKEND,
    STEP_PRODUCTION_VERIFY,
    STEP_REGISTRY_UPDATE,
    STEP_RENDER_WEB,
    STEP_RENDER_WORKER,
    load_public_publish_execute_progress,
)
from engine.ace_local_dev_publish_render_sim_execute_adapters import (
    build_simulated_render_publish_execute_adapters,
)
from engine.ace_local_dev_publish_uri_candidate_execute import (
    PublishExecuteRejected,
    execute_publish_uri_candidate,
)
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_candidate_preview_fingerprint
from engine.ace_local_dev_render_api_transport import MockRenderApiTransport, RenderApiResponse
from engine.ace_local_dev_render_api_types import (
    DEPLOY_STATUS_BUILD_FAILED,
    DEPLOY_STATUS_BUILD_IN_PROGRESS,
    DEPLOY_STATUS_LIVE,
    DEPLOY_STATUS_QUEUED,
    TRIGGER_API,
    TRIGGER_NEW_COMMIT,
)
from engine.ace_local_dev_render_publish_deploy import (
    ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
    ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
)
from tests.test_ace_local_dev_publish_uri_candidate import _raw
from tests.test_ace_local_dev_publish_uri_candidate_execute import CAND, OP, SID
from tests.test_ace_local_dev_publish_uri_candidate_prepare import _ready_plan

WEB = ACE_RENDER_DEFAULT_WEB_SERVICE_ID
WORKER = ACE_RENDER_DEFAULT_WORKER_SERVICE_ID
REGISTRY_REAL = Path(__file__).resolve().parent.parent.parent / "_registry" / "production-registry.json"


def _deploy(
    deploy_id: str,
    status: str,
    *,
    commit: str = CAND,
    trigger: str = TRIGGER_API,
) -> Dict[str, Any]:
    return {
        "id": deploy_id,
        "status": status,
        "commit": {"id": commit, "message": "sim"},
        "trigger": trigger,
        "createdAt": "2026-10-24T02:00:00Z",
    }


class SimMockState:
    def __init__(self) -> None:
        self.deploys: Dict[str, Dict[str, Dict[str, Any]]] = {WEB: {}, WORKER: {}}
        self.post_count: Dict[str, int] = {WEB: 0, WORKER: 0}
        self.pushed_sha: Optional[str] = None

    def seed(self, service_id: str, *rows: Dict[str, Any]) -> None:
        for row in rows:
            self.deploys[service_id][row["id"]] = dict(row)

    def list_payload(self, service_id: str) -> List[Dict[str, Any]]:
        rows = list(self.deploys[service_id].values())
        rows.sort(key=lambda r: r.get("createdAt", ""), reverse=True)
        return [{"deploy": r, "cursor": f"c-{r['id']}"} for r in rows]

    def set_status(self, service_id: str, deploy_id: str, status: str) -> None:
        self.deploys[service_id][deploy_id]["status"] = status

    def on_git_push(self, sha: str) -> None:
        self.pushed_sha = sha
        if "w-auto" not in self.deploys[WORKER]:
            self.seed(
                WORKER,
                _deploy("w-auto", DEPLOY_STATUS_BUILD_IN_PROGRESS, commit=sha, trigger=TRIGGER_NEW_COMMIT),
            )


def _service_id_from_url(url: str) -> str:
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    return parts[parts.index("services") + 1]


def _sim_route(state: SimMockState):
    def handler(method: str, url: str, *, headers, json_body=None) -> Optional[RenderApiResponse]:
        if not (headers.get("Authorization") or "").startswith("Bearer "):
            return RenderApiResponse(status_code=401, headers={}, body={"message": "auth"})
        path = urlparse(url).path
        sid = _service_id_from_url(url)

        if method == "GET" and "/deploys/" in path:
            deploy_id = path.rstrip("/").split("/deploys/")[-1]
            row = state.deploys.get(sid, {}).get(deploy_id)
            if not row:
                return RenderApiResponse(status_code=404, headers={}, body={"message": "missing"})
            return RenderApiResponse(status_code=200, headers={}, body=row)

        if method == "GET" and path.endswith("/deploys"):
            return RenderApiResponse(status_code=200, headers={}, body=state.list_payload(sid))

        if method == "POST" and path.endswith("/deploys"):
            state.post_count[sid] = state.post_count.get(sid, 0) + 1
            commit = (json_body or {}).get("commitId") or state.pushed_sha or CAND
            deploy_id = f"dep-{sid}-{state.post_count[sid]}"
            row = _deploy(deploy_id, DEPLOY_STATUS_QUEUED, commit=commit, trigger=TRIGGER_API)
            state.deploys.setdefault(sid, {})[deploy_id] = row
            return RenderApiResponse(status_code=201, headers={}, body=row)

        return RenderApiResponse(status_code=404, headers={}, body={"message": "unhandled"})

    return handler


def _advance_web_deploy_live(state: SimMockState):
    def handler(method: str, url: str, *, headers, json_body=None) -> Optional[RenderApiResponse]:
        path = urlparse(url).path
        if method == "GET" and "/deploys/dep-" in path and WEB in path:
            deploy_id = path.rstrip("/").split("/deploys/")[-1]
            cur = state.deploys[WEB][deploy_id]["status"]
            if cur == DEPLOY_STATUS_QUEUED:
                state.set_status(WEB, deploy_id, DEPLOY_STATUS_BUILD_IN_PROGRESS)
            else:
                state.set_status(WEB, deploy_id, DEPLOY_STATUS_LIVE)
        if method == "GET" and "/deploys/w-auto" in path:
            state.set_status(WORKER, "w-auto", DEPLOY_STATUS_LIVE)
        return None

    return handler


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


class RenderSimExecuteBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ace-render-sim-exec-")
        self.sessions_root = Path(self._tmp)
        os.environ["ACE_SESSIONS_ROOT"] = str(self.sessions_root)
        os.environ["ACE_PUBLISH_CANDIDATE_REGISTRY_PATH"] = str(
            Path(self._tmp) / "publish-candidate-registry.json"
        )
        os.environ["ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE"] = "1"
        self._registry_mtime_before = REGISTRY_REAL.stat().st_mtime if REGISTRY_REAL.is_file() else 0

    def tearDown(self) -> None:
        os.environ.pop("ACE_SESSIONS_ROOT", None)
        os.environ.pop("ACE_PUBLISH_CANDIDATE_REGISTRY_PATH", None)
        os.environ.pop("ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE", None)
        shutil.rmtree(self._tmp, ignore_errors=True)
        real = Path(__file__).resolve().parent.parent.parent / "_backups" / "sessions" / SID
        if real.is_dir():
            shutil.rmtree(real, ignore_errors=True)

    def _build_adapters(self, state: SimMockState, *, extra_handlers=None, **kwargs):
        transport = MockRenderApiTransport()
        transport.set_default_handler(_sim_route(state))
        transport.add_handler(_advance_web_deploy_live(state))
        if extra_handlers:
            for h in extra_handlers:
                transport.add_handler(h)
        adapters = build_simulated_render_publish_execute_adapters(
            transport,
            registry_simulation_path=Path(self._tmp) / "sim-registry-update.json",
            poll_interval_seconds=0.0,
            wait_deadline_seconds=10.0,
            **kwargs,
        )

        def _git_push_hook(ctx):
            state.on_git_push(ctx.precheck.candidate_backend_sha)

        orig = adapters.git_push_backend

        def wrapped(ctx):
            res = orig(ctx)
            if res.ok:
                state.on_git_push(ctx.precheck.candidate_backend_sha)
            return res

        adapters.git_push_backend = wrapped  # type: ignore[method-assign]
        return adapters, transport

    def _run_execute(self, adapters):
        bundle = _precheck_bundle()
        with patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.publish_real_execute_permitted",
            return_value=True,
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.resolve_active_registered_candidate",
            return_value=(SID, CAND, {}),
        ), patch(
            "engine.ace_local_dev_publish_uri_candidate_execute.run_publish_candidate_prechecks",
            return_value=bundle,
        ):
            return execute_publish_uri_candidate(
                confirm_execute="PUBLISH",
                publish_uri_preview_fingerprint=bundle.fingerprint,
                operation_id=OP,
                adapters=adapters,
            )


class RenderSimExecuteIntegrationTests(RenderSimExecuteBase):
    def test_full_simulated_chain_success(self) -> None:
        state = SimMockState()
        adapters, transport = self._build_adapters(state)
        out = self._run_execute(adapters)
        self.assertTrue(out["ok"])
        self.assertEqual(state.post_count[WORKER], 0)
        self.assertEqual(state.post_count[WEB], 1)
        self.assertEqual(adapters.stage_calls.get("git_push_backend"), 1)
        self.assertTrue(adapters.sim_state.registry_simulation_written)
        prog = load_public_publish_execute_progress(OP, session_id_hint=SID)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "COMPLETED")
        ids = [s["id"] for s in prog["completedSteps"]]
        for step in (
            STEP_GIT_PUSH_BACKEND,
            STEP_RENDER_WEB,
            STEP_RENDER_WORKER,
            STEP_PRODUCTION_VERIFY,
            STEP_REGISTRY_UPDATE,
        ):
            self.assertIn(step, ids)
        if REGISTRY_REAL.is_file():
            self.assertEqual(REGISTRY_REAL.stat().st_mtime, self._registry_mtime_before)

    def test_stage_order_preserved(self) -> None:
        state = SimMockState()
        adapters, _ = self._build_adapters(state)
        self._run_execute(adapters)
        required = [
            "prepare_isolated_release",
            "git_push_backend",
            "wait_render_web",
            "wait_render_worker",
            "verify_production",
            "update_production_registry",
        ]
        for name in required:
            self.assertGreaterEqual(adapters.stage_calls.get(name, 0), 1, msg=name)
        self.assertIsNone(adapters.stage_calls.get("deploy_github_pages"))

    def test_web_deploy_failure_partial(self) -> None:
        state = SimMockState()

        def fail_web(_m, url, *, headers, json_body=None) -> Optional[RenderApiResponse]:
            path = urlparse(url).path
            if _m == "GET" and "/deploys/dep-" in path and WEB in path:
                deploy_id = path.rstrip("/").split("/deploys/")[-1]
                state.set_status(WEB, deploy_id, DEPLOY_STATUS_BUILD_FAILED)
            return None

        adapters, _ = self._build_adapters(state, extra_handlers=[fail_web])
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)
        prog = load_public_publish_execute_progress(OP, session_id_hint=SID)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "PARTIAL")

    def test_worker_deploy_failure_partial(self) -> None:
        state = SimMockState()

        def fail_worker(_m, url, *, headers, json_body=None) -> Optional[RenderApiResponse]:
            if "/deploys/w-auto" in urlparse(url).path:
                state.set_status(WORKER, "w-auto", DEPLOY_STATUS_BUILD_FAILED)
            return None

        adapters, _ = self._build_adapters(state, extra_handlers=[fail_worker])
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)
        prog = load_public_publish_execute_progress(OP, session_id_hint=SID)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "PARTIAL")

    def test_worker_auto_deploy_missing_timeout(self) -> None:
        state = SimMockState()
        adapters, _ = self._build_adapters(state)

        def git_without_worker_seed(ctx):
            adapters._tick("git_push_backend")  # noqa: SLF001 — test-only
            state.pushed_sha = ctx.precheck.candidate_backend_sha
            return StageResult(ok=True, code="ace_publish_sim_git_push_ok")

        adapters.git_push_backend = git_without_worker_seed  # type: ignore[method-assign]
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)

    def test_verify_pair_sha_mismatch(self) -> None:
        state = SimMockState()
        state.seed(WEB, _deploy("wl", DEPLOY_STATUS_LIVE, commit=CAND))
        state.seed(WORKER, _deploy("wk", DEPLOY_STATUS_LIVE, commit="fadbddd4abb672fd2f4c436f2afddeeff5ea2f67"))
        adapters, _ = self._build_adapters(state)
        adapters.wait_render_web = lambda ctx: StageResult(ok=True, code="skip")  # type: ignore[method-assign]
        adapters.wait_render_worker = lambda ctx: StageResult(ok=True, code="skip")  # type: ignore[method-assign]
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)

    def test_competing_deploy_verify_fails(self) -> None:
        state = SimMockState()
        state.seed(
            WEB,
            _deploy("newer", DEPLOY_STATUS_BUILD_IN_PROGRESS, commit=CAND),
            _deploy("live", DEPLOY_STATUS_LIVE, commit=CAND),
        )
        state.seed(WORKER, _deploy("wk", DEPLOY_STATUS_LIVE, commit=CAND))
        adapters, _ = self._build_adapters(state)
        adapters.wait_render_web = lambda ctx: StageResult(ok=True, code="skip")  # type: ignore[method-assign]
        adapters.wait_render_worker = lambda ctx: StageResult(ok=True, code="skip")  # type: ignore[method-assign]
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)

    def test_render_auth_error(self) -> None:
        transport = MockRenderApiTransport()
        transport.set_default_handler(
            lambda *_a, **_k: RenderApiResponse(status_code=401, headers={}, body={"message": "denied"})
        )
        adapters = build_simulated_render_publish_execute_adapters(
            transport, poll_interval_seconds=0.0, wait_deadline_seconds=5.0
        )
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)

    def test_health_check_failure_blocks_success(self) -> None:
        state = SimMockState()
        adapters, _ = self._build_adapters(state, health_check=lambda: False)
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)
        prog = load_public_publish_execute_progress(OP, session_id_hint=SID)
        assert prog is not None
        self.assertEqual(prog["operationStatus"], "PARTIAL")

    def test_registry_update_not_called_on_verify_fail(self) -> None:
        state = SimMockState()
        adapters, _ = self._build_adapters(state, health_check=lambda: False)
        with self.assertRaises(PublishExecuteRejected):
            self._run_execute(adapters)
        self.assertFalse(adapters.sim_state.registry_simulation_written)

    def test_local_gate_still_blocks_without_permit(self) -> None:
        bundle = _precheck_bundle()
        state = SimMockState()
        adapters, _ = self._build_adapters(state)
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
                    operation_id="c" * 32,
                    adapters=adapters,
                )
        self.assertEqual(ctx.exception.code, "ace_publish_execute_not_allowed")
        self.assertEqual(adapters.stage_calls.get("git_push_backend"), None)


if __name__ == "__main__":
    unittest.main()
