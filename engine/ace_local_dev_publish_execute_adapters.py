"""Publish Execute side-effect ports — blocked by default; mockable in tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from engine.ace_local_dev_publish_candidate_prechecks import PublishPrecheckResult


class PublishExecuteAdapterError(Exception):
    def __init__(self, code: str, *, message: Optional[str] = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


@dataclass
class PublishExecuteContext:
    precheck: PublishPrecheckResult
    operation_id: str
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    ok: bool
    code: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class PublishExecuteAdapters(Protocol):
    def prepare_isolated_release(self, ctx: PublishExecuteContext) -> StageResult: ...
    def git_push_backend(self, ctx: PublishExecuteContext) -> StageResult: ...
    def wait_render_web(self, ctx: PublishExecuteContext) -> StageResult: ...
    def wait_render_worker(self, ctx: PublishExecuteContext) -> StageResult: ...
    def deploy_github_pages(self, ctx: PublishExecuteContext) -> StageResult: ...
    def verify_production(self, ctx: PublishExecuteContext) -> StageResult: ...
    def update_production_registry(self, ctx: PublishExecuteContext) -> StageResult: ...


class BlockedPublishExecuteAdapters:
    """Default production adapter — never performs git/deploy/registry I/O."""

    def _blocked(self, stage: str) -> StageResult:
        return StageResult(
            ok=False,
            code=f"ace_publish_execute_{stage}_blocked",
            message="Real publish execute is disabled; adapter not configured.",
        )

    def prepare_isolated_release(self, ctx: PublishExecuteContext) -> StageResult:
        plan = ctx.precheck.execution_plan
        candidate = str(
            (plan.get("verifiedSources") or {}).get("candidateBackendSha") or ""
        )
        if candidate != ctx.precheck.candidate_backend_sha:
            return StageResult(
                ok=False,
                code="ace_publish_execute_candidate_sha_mismatch",
                message="Prepared release SHA does not match registered candidate.",
            )
        local_paths = list((plan.get("allowedProductFiles") or {}).get("backend") or [])
        for path in local_paths:
            token = str(path).lower()
            if "local-dev-infrastructure" in token or token.startswith("scripts/ori"):
                return StageResult(
                    ok=False,
                    code="ace_publish_execute_local_path_leak",
                    message="Local-only path in allowed publish set.",
                )
        return StageResult(ok=True, code="ace_publish_execute_isolated_release_ready")

    def git_push_backend(self, ctx: PublishExecuteContext) -> StageResult:
        return self._blocked("git_push")

    def wait_render_web(self, ctx: PublishExecuteContext) -> StageResult:
        return self._blocked("render_web")

    def wait_render_worker(self, ctx: PublishExecuteContext) -> StageResult:
        return self._blocked("render_worker")

    def deploy_github_pages(self, ctx: PublishExecuteContext) -> StageResult:
        return self._blocked("github_pages")

    def verify_production(self, ctx: PublishExecuteContext) -> StageResult:
        return self._blocked("production_verify")

    def update_production_registry(self, ctx: PublishExecuteContext) -> StageResult:
        return self._blocked("registry_update")
