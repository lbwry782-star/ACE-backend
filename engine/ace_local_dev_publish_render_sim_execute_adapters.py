"""Simulated Publish Execute adapters — Render API mock transport only (no real deploy)."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from engine.ace_local_dev_publish_execute_adapters import (
    BlockedPublishExecuteAdapters,
    PublishExecuteContext,
    StageResult,
)
from engine.ace_local_dev_render_api_client import RenderApiClient
from engine.ace_local_dev_render_api_errors import (
    RenderApiError,
    RenderDeployFailedError,
    RenderDeployTimeoutError,
)
from engine.ace_local_dev_render_api_transport import RenderApiTransport
from engine.ace_local_dev_render_api_types import deploy_commit_sha
from engine.ace_local_dev_render_publish_deploy import (
    ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
    ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
    execute_web_deploy,
    execute_worker_auto_deploy,
    verify_web_and_worker_pair,
)

# Fixed placeholder for mock-client tests only — not a Render account secret.
ACE_PUBLISH_RENDER_SIM_MOCK_API_KEY = "ace-publish-render-sim-mock"


@dataclass
class RenderPublishSimState:
    """In-memory coordination for mock transport handlers in tests."""

    last_simulated_git_push_sha: Optional[str] = None
    web_deploy_id: Optional[str] = None
    worker_deploy_id: Optional[str] = None
    registry_simulation_written: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def mark_git_push_simulated(self, commit_sha: str) -> None:
        self.last_simulated_git_push_sha = commit_sha


HealthCheck = Callable[[], bool]


class SimulatedRenderPublishExecuteAdapters:
    """
    PublishExecuteAdapters implementation using RenderApiClient + injectable transport.
    Never uses Urllib transport; never touches production registry or git remotes.
    """

    def __init__(
        self,
        client: RenderApiClient,
        *,
        web_service_id: str = ACE_RENDER_DEFAULT_WEB_SERVICE_ID,
        worker_service_id: str = ACE_RENDER_DEFAULT_WORKER_SERVICE_ID,
        sim_state: Optional[RenderPublishSimState] = None,
        registry_simulation_path: Optional[Path] = None,
        health_check: Optional[HealthCheck] = None,
        wait_deadline_seconds: float = 30.0,
        poll_interval_seconds: float = 0.01,
    ) -> None:
        self._client = client
        self._web_service_id = web_service_id
        self._worker_service_id = worker_service_id
        self._sim_state = sim_state or RenderPublishSimState()
        self._registry_simulation_path = registry_simulation_path
        self._health_check = health_check or (lambda: True)
        self._wait_deadline_seconds = wait_deadline_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._prepare_guard = BlockedPublishExecuteAdapters()
        self.allow_execute_gate_bypass_for_simulation = False
        self.stage_calls: Dict[str, int] = {}

    @property
    def sim_state(self) -> RenderPublishSimState:
        return self._sim_state

    def _tick(self, name: str) -> None:
        self.stage_calls[name] = self.stage_calls.get(name, 0) + 1

    def prepare_isolated_release(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("prepare_isolated_release")
        return self._prepare_guard.prepare_isolated_release(ctx)

    def git_push_backend(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("git_push_backend")
        sha = ctx.precheck.candidate_backend_sha
        if hasattr(self._sim_state, "on_git_push"):
            self._sim_state.on_git_push(sha)
        else:
            self._sim_state.mark_git_push_simulated(sha)
        ctx.notes["renderSimGitPush"] = {"simulated": True, "commitSha": sha}
        return StageResult(ok=True, code="ace_publish_sim_git_push_ok", details={"commitSha": sha})

    def wait_render_web(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("wait_render_web")
        sha = ctx.precheck.candidate_backend_sha
        try:
            deploy = execute_web_deploy(
                self._client,
                self._web_service_id,
                sha,
                wait_deadline_seconds=self._wait_deadline_seconds,
                poll_interval_seconds=self._poll_interval_seconds,
            )
        except (RenderApiError, RenderDeployFailedError, RenderDeployTimeoutError) as exc:
            code = getattr(exc, "code", "ace_publish_sim_render_web_failed")
            return StageResult(ok=False, code=str(code), message=getattr(exc, "message", str(exc)))
        self._sim_state.web_deploy_id = deploy.id
        commit = deploy_commit_sha(deploy) or sha
        ctx.notes["renderSimWeb"] = {"deployId": deploy.id, "commitSha": commit}
        return StageResult(
            ok=True,
            code="ace_publish_sim_render_web_ok",
            details={"deployId": deploy.id, "commitSha": commit},
        )

    def wait_render_worker(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("wait_render_worker")
        sha = ctx.precheck.candidate_backend_sha
        try:
            deploy = execute_worker_auto_deploy(
                self._client,
                self._worker_service_id,
                sha,
                auto_deploy_on_commit=True,
                wait_deadline_seconds=self._wait_deadline_seconds,
                poll_interval_seconds=self._poll_interval_seconds,
            )
        except (RenderApiError, RenderDeployFailedError, RenderDeployTimeoutError) as exc:
            code = getattr(exc, "code", "ace_publish_sim_render_worker_failed")
            return StageResult(ok=False, code=str(code), message=getattr(exc, "message", str(exc)))
        self._sim_state.worker_deploy_id = deploy.id
        commit = deploy_commit_sha(deploy) or sha
        ctx.notes["renderSimWorker"] = {"deployId": deploy.id, "commitSha": commit}
        return StageResult(
            ok=True,
            code="ace_publish_sim_render_worker_ok",
            details={"deployId": deploy.id, "commitSha": commit},
        )

    def deploy_github_pages(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("deploy_github_pages")
        ctx.notes["renderSimPages"] = {"simulated": True, "skipped": True}
        return StageResult(ok=True, code="ace_publish_sim_github_pages_skipped")

    def verify_production(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("verify_production")
        sha = ctx.precheck.candidate_backend_sha
        pair = verify_web_and_worker_pair(
            self._client,
            web_service_id=self._web_service_id,
            worker_service_id=self._worker_service_id,
            expected_sha=sha,
        )
        if not pair.ok:
            return StageResult(ok=False, code=pair.code, message=pair.message)
        if not self._health_check():
            return StageResult(
                ok=False,
                code="ace_publish_sim_health_check_failed",
                message="Simulated web health check failed.",
            )
        ctx.notes["renderSimVerify"] = {
            "commitSha": pair.expected_sha,
            "webDeployId": pair.web_deploy.id if pair.web_deploy else None,
            "workerDeployId": pair.worker_deploy.id if pair.worker_deploy else None,
        }
        return StageResult(ok=True, code="ace_publish_sim_verify_ok", details={"commitSha": sha})

    def update_production_registry(self, ctx: PublishExecuteContext) -> StageResult:
        self._tick("update_production_registry")
        payload = {
            "simulated": True,
            "operationId": ctx.operation_id,
            "candidateBackendSha": ctx.precheck.candidate_backend_sha,
            "note": "Not written to production-registry.json",
        }
        if self._registry_simulation_path is not None:
            path = self._registry_simulation_path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
            self._sim_state.registry_simulation_written = True
        ctx.notes["renderSimRegistry"] = payload
        return StageResult(ok=True, code="ace_publish_sim_registry_ok", details={"simulated": True})


def build_simulated_render_publish_execute_adapters(
    transport: RenderApiTransport,
    *,
    registry_simulation_path: Optional[Path] = None,
    sim_state: Optional[RenderPublishSimState] = None,
    health_check: Optional[HealthCheck] = None,
    wait_deadline_seconds: float = 30.0,
    poll_interval_seconds: float = 0.01,
) -> SimulatedRenderPublishExecuteAdapters:
    client = RenderApiClient(
        transport,
        api_key=ACE_PUBLISH_RENDER_SIM_MOCK_API_KEY,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    return SimulatedRenderPublishExecuteAdapters(
        client,
        sim_state=sim_state,
        registry_simulation_path=registry_simulation_path,
        health_check=health_check,
        wait_deadline_seconds=wait_deadline_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
