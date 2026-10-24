"""Real Publish Execute adapters — port-based, not wired to LOCAL API by default.

Use only with explicit RealPublishExecutePorts (mocks in tests; future orchestration wiring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from engine.ace_local_dev_publish_execute_adapters import (
    BlockedPublishExecuteAdapters,
    PublishExecuteContext,
    StageResult,
)
from engine.ace_local_dev_publish_real_execute_ports import (
    RealPublishExecutePorts,
    load_real_publish_stage_snapshot,
    record_real_publish_stage,
    registry_update_allowed,
)


@dataclass
class RealPublishExecuteConfig:
    """Explicit targets — no production service ID defaults."""

    web_service_id: str
    worker_service_id: str
    backend_remote: str = "origin"
    backend_branch: str = "main"


class RealPublishExecuteAdapters:
    """
    PublishExecuteAdapters implementation delegating all external I/O to injectable ports.

    Not selectable from ace_local_dev_routes or execute_publish_uri_candidate defaults.
    """

    adapter_kind = "real_publish_execute_ports"

    def __init__(
        self,
        ports: RealPublishExecutePorts,
        *,
        config: Optional[RealPublishExecuteConfig] = None,
    ) -> None:
        if ports is None:
            raise ValueError("RealPublishExecuteAdapters requires RealPublishExecutePorts")
        self._ports = ports
        self._config = config
        self._prepare_guard = BlockedPublishExecuteAdapters()

    def prepare_isolated_release(self, ctx: PublishExecuteContext) -> StageResult:
        result = self._prepare_guard.prepare_isolated_release(ctx)
        record_real_publish_stage(ctx, "prepare_isolated_release", result)
        return result

    def git_push_backend(self, ctx: PublishExecuteContext) -> StageResult:
        candidate = ctx.precheck.candidate_backend_sha
        plan_sha = str(
            (ctx.precheck.execution_plan.get("verifiedSources") or {}).get("candidateBackendSha") or ""
        )
        if plan_sha and plan_sha != candidate:
            result = StageResult(
                ok=False,
                code="ace_publish_execute_candidate_sha_mismatch",
                message="Execution plan candidate SHA does not match registered candidate.",
            )
            record_real_publish_stage(ctx, "git_push_backend", result)
            return result
        outcome = self._ports.git_backend_push.push_backend_to_main(
            candidate_sha=candidate,
            execution_plan=dict(ctx.precheck.execution_plan or {}),
        )
        if outcome.ok:
            result = StageResult(
                ok=True,
                code="ace_publish_real_git_push_ok",
                message=outcome.message or "Backend push completed or already on remote.",
                details={
                    "commitSha": outcome.commit_sha or candidate,
                    "alreadyOnRemote": outcome.already_on_remote,
                    "pushed": outcome.pushed,
                    "idempotent": outcome.idempotent or outcome.already_on_remote,
                },
            )
        else:
            result = StageResult(
                ok=False,
                code=outcome.code or "ace_publish_real_git_push_failed",
                message=outcome.message or "Backend git push failed.",
            )
        record_real_publish_stage(ctx, "git_push_backend", result)
        return result

    def wait_render_web(self, ctx: PublishExecuteContext) -> StageResult:
        snapshot = load_real_publish_stage_snapshot(ctx)
        git_stage = snapshot.get("git_push_backend") or {}
        if ctx.precheck.readiness.get("hasBackendProductChanges") and git_stage.get("ok") is not True:
            result = StageResult(
                ok=False,
                code="ace_publish_real_render_web_prerequisite",
                message="Render Web deploy blocked until git push stage succeeded.",
            )
            record_real_publish_stage(ctx, "wait_render_web", result)
            return result
        candidate = ctx.precheck.candidate_backend_sha
        outcome = self._ports.render_web.wait_for_web_live(candidate_sha=candidate)
        result = _render_outcome_to_stage(outcome, ok_code="ace_publish_real_render_web_ok")
        record_real_publish_stage(ctx, "wait_render_web", result)
        return result

    def wait_render_worker(self, ctx: PublishExecuteContext) -> StageResult:
        snapshot = load_real_publish_stage_snapshot(ctx)
        git_stage = snapshot.get("git_push_backend") or {}
        if ctx.precheck.readiness.get("hasBackendProductChanges") and git_stage.get("ok") is not True:
            result = StageResult(
                ok=False,
                code="ace_publish_real_render_worker_prerequisite",
                message="Render Worker observe blocked until git push stage succeeded.",
            )
            record_real_publish_stage(ctx, "wait_render_worker", result)
            return result
        candidate = ctx.precheck.candidate_backend_sha
        outcome = self._ports.render_worker.wait_for_worker_live(candidate_sha=candidate)
        result = _render_outcome_to_stage(
            outcome,
            ok_code="ace_publish_real_render_worker_ok",
            observed=True,
        )
        record_real_publish_stage(ctx, "wait_render_worker", result)
        return result

    def deploy_github_pages(self, ctx: PublishExecuteContext) -> StageResult:
        if not ctx.precheck.readiness.get("hasFrontendProductChanges"):
            result = StageResult(
                ok=True,
                code="ace_publish_real_github_pages_skipped",
                message="No frontend product changes.",
                details={"skipped": True},
            )
            record_real_publish_stage(ctx, "deploy_github_pages", result)
            return result
        fe_plan = ctx.precheck.execution_plan.get("verifiedSources") or {}
        frontend_sha = str(fe_plan.get("proposedFrontendHead") or fe_plan.get("frontendHead") or "")
        if not frontend_sha:
            plan = ctx.precheck.plan if ctx.precheck else {}
            fe = plan.get("proposedChanges", {}).get("frontend", {}) if isinstance(plan, dict) else {}
            frontend_sha = str(fe.get("proposedHeadSha") or "")
        outcome = self._ports.github_pages.deploy_frontend(
            frontend_commit_sha=frontend_sha,
            execution_plan=dict(ctx.precheck.execution_plan or {}),
        )
        if outcome.ok:
            result = StageResult(
                ok=True,
                code="ace_publish_real_github_pages_ok",
                message=outcome.message or "GitHub Pages deploy completed.",
                details={
                    "frontendCommitSha": outcome.frontend_commit_sha or frontend_sha,
                    "workflowRunId": outcome.workflow_run_id,
                },
            )
        else:
            result = StageResult(
                ok=False,
                code=outcome.code or "ace_publish_real_github_pages_failed",
                message=outcome.message or "GitHub Pages deploy failed.",
            )
        record_real_publish_stage(ctx, "deploy_github_pages", result)
        return result

    def verify_production(self, ctx: PublishExecuteContext) -> StageResult:
        snapshot = load_real_publish_stage_snapshot(ctx)
        candidate = ctx.precheck.candidate_backend_sha
        frontend_required = bool(ctx.precheck.readiness.get("hasFrontendProductChanges"))
        if ctx.precheck.readiness.get("hasBackendProductChanges"):
            for stage_id in ("git_push_backend", "wait_render_web", "wait_render_worker"):
                entry = snapshot.get(stage_id) or {}
                if entry.get("ok") is not True:
                    result = StageResult(
                        ok=False,
                        code="ace_publish_real_verify_prerequisites_missing",
                        message=f"Production verify blocked; stage {stage_id} not successful.",
                        details={"missingStage": stage_id},
                    )
                    record_real_publish_stage(ctx, "verify_production", result)
                    return result
        if frontend_required:
            pages = snapshot.get("deploy_github_pages") or {}
            if pages.get("ok") is not True:
                result = StageResult(
                    ok=False,
                    code="ace_publish_real_verify_frontend_missing",
                    message="Production verify blocked; frontend deploy not successful.",
                )
                record_real_publish_stage(ctx, "verify_production", result)
                return result
        outcome = self._ports.production_verify.verify_production(
            candidate_backend_sha=candidate,
            frontend_deploy_required=frontend_required,
            stage_snapshot=snapshot,
        )
        if outcome.ok:
            ctx.notes["realPublishVerify"] = {
                "backendCommitSha": outcome.backend_commit_sha or candidate,
                "frontendVerified": outcome.frontend_verified,
            }
            result = StageResult(
                ok=True,
                code="ace_publish_real_verify_ok",
                message=outcome.message or "Production verification succeeded.",
                details={"backendCommitSha": outcome.backend_commit_sha or candidate},
            )
        else:
            result = StageResult(
                ok=False,
                code=outcome.code or "ace_publish_real_verify_failed",
                message=outcome.message or "Production verification failed.",
            )
        record_real_publish_stage(ctx, "verify_production", result)
        return result

    def update_production_registry(self, ctx: PublishExecuteContext) -> StageResult:
        blocked = registry_update_allowed(ctx)
        if blocked is not None:
            record_real_publish_stage(ctx, "update_production_registry", blocked)
            return blocked
        snapshot = load_real_publish_stage_snapshot(ctx)
        verify_raw = ctx.notes.get("realPublishVerify") or {}
        from engine.ace_local_dev_publish_real_execute_ports import ProductionVerifyOutcome

        verify_outcome = ProductionVerifyOutcome(
            ok=True,
            backend_commit_sha=str(verify_raw.get("backendCommitSha") or ctx.precheck.candidate_backend_sha),
            frontend_verified=bool(verify_raw.get("frontendVerified")),
        )
        outcome = self._ports.registry_update.update_production_registry(
            ctx=ctx,
            stage_snapshot=snapshot,
            verify_outcome=verify_outcome,
        )
        if outcome.ok:
            result = StageResult(
                ok=True,
                code="ace_publish_real_registry_ok",
                message=outcome.message or "Registry update completed.",
                details={"writtenPath": outcome.written_path},
            )
        else:
            result = StageResult(
                ok=False,
                code=outcome.code or "ace_publish_real_registry_failed",
                message=outcome.message or "Registry update failed.",
            )
        record_real_publish_stage(ctx, "update_production_registry", result)
        return result


def _render_outcome_to_stage(
    outcome,
    *,
    ok_code: str,
    observed: bool = False,
) -> StageResult:
    if outcome.ok:
        return StageResult(
            ok=True,
            code=ok_code,
            message=outcome.message or "Render stage completed.",
            details={
                "deployId": outcome.deploy_id,
                "commitSha": outcome.commit_sha,
                "alreadyLive": outcome.already_live,
                "observedOnly": observed or outcome.observed_only,
            },
        )
    return StageResult(
        ok=False,
        code=outcome.code or "ace_publish_real_render_failed",
        message=outcome.message or "Render stage failed.",
    )
