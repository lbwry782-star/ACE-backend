"""Injectable ports for RealPublishExecuteAdapters (tests and future wiring only).

No default production credentials, service IDs, or registry paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from engine.ace_local_dev_publish_execute_adapters import PublishExecuteContext, StageResult


@dataclass(frozen=True)
class GitBackendPushOutcome:
    ok: bool
    code: str = ""
    message: str = ""
    commit_sha: str = ""
    already_on_remote: bool = False
    pushed: bool = False
    idempotent: bool = False


@dataclass(frozen=True)
class RenderDeployOutcome:
    ok: bool
    code: str = ""
    message: str = ""
    deploy_id: str = ""
    commit_sha: str = ""
    already_live: bool = False
    observed_only: bool = False


@dataclass(frozen=True)
class GitHubPagesDeployOutcome:
    ok: bool
    code: str = ""
    message: str = ""
    workflow_run_id: str = ""
    frontend_commit_sha: str = ""


@dataclass(frozen=True)
class ProductionVerifyOutcome:
    ok: bool
    code: str = ""
    message: str = ""
    backend_commit_sha: str = ""
    frontend_verified: bool = False


@dataclass(frozen=True)
class ProductionRegistryUpdateOutcome:
    ok: bool
    code: str = ""
    message: str = ""
    written_path: str = ""


class GitBackendPushPort(Protocol):
    def push_backend_to_main(self, *, candidate_sha: str, execution_plan: Dict[str, Any]) -> GitBackendPushOutcome: ...


class RenderWebDeployPort(Protocol):
    def wait_for_web_live(self, *, candidate_sha: str) -> RenderDeployOutcome: ...


class RenderWorkerObservePort(Protocol):
    def wait_for_worker_live(self, *, candidate_sha: str) -> RenderDeployOutcome: ...


class GitHubPagesDeployPort(Protocol):
    def deploy_frontend(self, *, frontend_commit_sha: str, execution_plan: Dict[str, Any]) -> GitHubPagesDeployOutcome: ...


class ProductionVerifyPort(Protocol):
    def verify_production(
        self,
        *,
        candidate_backend_sha: str,
        frontend_deploy_required: bool,
        stage_snapshot: Dict[str, Any],
    ) -> ProductionVerifyOutcome: ...


class ProductionRegistryUpdatePort(Protocol):
    def update_production_registry(
        self,
        *,
        ctx: PublishExecuteContext,
        stage_snapshot: Dict[str, Any],
        verify_outcome: ProductionVerifyOutcome,
    ) -> ProductionRegistryUpdateOutcome: ...


@dataclass
class RealPublishExecutePorts:
    git_backend_push: GitBackendPushPort
    render_web: RenderWebDeployPort
    render_worker: RenderWorkerObservePort
    github_pages: GitHubPagesDeployPort
    production_verify: ProductionVerifyPort
    registry_update: ProductionRegistryUpdatePort


REAL_PUBLISH_STAGE_NOTES_KEY = "realPublishStages"


def record_real_publish_stage(ctx: PublishExecuteContext, stage_id: str, result: StageResult) -> None:
    bucket = ctx.notes.setdefault(REAL_PUBLISH_STAGE_NOTES_KEY, {})
    if not isinstance(bucket, dict):
        bucket = {}
        ctx.notes[REAL_PUBLISH_STAGE_NOTES_KEY] = bucket
    bucket[stage_id] = {
        "ok": result.ok,
        "code": result.code,
        "message": result.message,
        "details": dict(result.details or {}),
    }


def load_real_publish_stage_snapshot(ctx: PublishExecuteContext) -> Dict[str, Any]:
    raw = ctx.notes.get(REAL_PUBLISH_STAGE_NOTES_KEY)
    return dict(raw) if isinstance(raw, dict) else {}


def required_real_publish_stages(ctx: PublishExecuteContext) -> list[str]:
    readiness = ctx.precheck.readiness if ctx.precheck else {}
    required = ["verify_production"]
    if readiness.get("hasBackendProductChanges"):
        required = ["git_push_backend", "wait_render_web", "wait_render_worker", *required]
    if readiness.get("hasFrontendProductChanges"):
        if "deploy_github_pages" not in required:
            required.insert(-1, "deploy_github_pages")
    return required


def missing_prerequisite_stages(ctx: PublishExecuteContext) -> list[str]:
    required = required_real_publish_stages(ctx)
    snapshot = load_real_publish_stage_snapshot(ctx)
    missing: list[str] = []
    for stage_id in required:
        entry = snapshot.get(stage_id)
        if not isinstance(entry, dict) or entry.get("ok") is not True:
            missing.append(stage_id)
    return missing


def registry_update_allowed(ctx: PublishExecuteContext) -> Optional[StageResult]:
    missing = missing_prerequisite_stages(ctx)
    if missing:
        return StageResult(
            ok=False,
            code="ace_publish_real_registry_prerequisites_missing",
            message="Production registry update blocked until required stages succeed.",
            details={"missingStages": missing},
        )
    return None


def is_production_registry_path(path: Path) -> bool:
    """Heuristic guard — refuses accidental writes to live ACE registry file name."""
    name = path.name.lower()
    if name != "production-registry.json":
        return False
    parts = {p.lower() for p in path.parts}
    return "_registry" in parts and "ace-publish-test" not in parts and "ace-publish-real-test" not in parts
