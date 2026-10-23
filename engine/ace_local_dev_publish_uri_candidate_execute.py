"""Publish Uri Candidate Execute — orchestrator (blocked on LOCAL by default)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.ace_local_dev_cancel_progress import new_operation_id, resolve_operation_id
from engine.ace_local_dev_execute_guard import (
    LocalDevExecuteBusyError,
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_publish_candidate_prechecks import (
    PublishPrecheckRejected,
    resolve_active_registered_candidate,
    run_publish_candidate_prechecks,
    validate_fingerprint_token,
)
from engine.ace_local_dev_publish_execute_adapters import (
    BlockedPublishExecuteAdapters,
    PublishExecuteAdapters,
    PublishExecuteContext,
    StageResult,
)
from engine.ace_local_dev_publish_execute_progress import (
    STEP_CANDIDATE_SHA_MATCH,
    STEP_EXECUTION_PLAN,
    STEP_FINGERPRINT,
    STEP_FRESH_PREVIEW,
    STEP_GITHUB_PAGES,
    STEP_GIT_PUSH_BACKEND,
    STEP_ISOLATED_RELEASE,
    STEP_PRODUCTION_VERIFY,
    STEP_PRODUCT_SCOPE,
    STEP_REAL_EXECUTE_GATE,
    STEP_REGISTERED_CANDIDATE,
    STEP_REGISTRY_UPDATE,
    STEP_RENDER_WEB,
    STEP_RENDER_WORKER,
    STEP_WORKER_COUPLING,
    PublishExecuteProgressWriter,
    assert_publish_execute_operation_id_available,
)
from engine.ace_local_dev_publish_real_execute_gate import publish_real_execute_permitted

logger = logging.getLogger(__name__)

CONFIRM_EXECUTE_TOKEN = "PUBLISH"


class PublishExecuteRejected(Exception):
    def __init__(
        self,
        code: str,
        *,
        message: Optional[str] = None,
        operation_id: Optional[str] = None,
        blockers: Optional[list] = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code
        self.operation_id = operation_id
        self.blockers = blockers or []


def _run_precheck_steps(
    progress: PublishExecuteProgressWriter,
    *,
    expected_fingerprint: str,
) -> Any:
    progress.begin_step(STEP_REGISTERED_CANDIDATE)
    resolve_active_registered_candidate()
    progress.complete_step(STEP_REGISTERED_CANDIDATE)

    pre = run_publish_candidate_prechecks(expected_fingerprint=expected_fingerprint)
    progress.set_execution_plan(pre.execution_plan)
    for step_id in (
        STEP_FRESH_PREVIEW,
        STEP_FINGERPRINT,
        STEP_PRODUCT_SCOPE,
        STEP_WORKER_COUPLING,
        STEP_EXECUTION_PLAN,
    ):
        progress.begin_step(step_id)
        progress.complete_step(step_id)
    return pre


def _run_stage(
    progress: PublishExecuteProgressWriter,
    step_id: str,
    fn,
    ctx: PublishExecuteContext,
    *,
    partial_on_fail: bool = False,
) -> StageResult:
    progress.begin_step(step_id)
    result = fn(ctx)
    if result.ok:
        progress.complete_step(step_id, internal_note=result.code or None)
        return result
    progress.fail_step(
        step_id,
        error=result.message or result.code,
        internal_error=result.code,
        partial=partial_on_fail,
    )
    raise PublishExecuteRejected(result.code, message=result.message, operation_id=ctx.operation_id)


def execute_publish_uri_candidate(
    *,
    confirm_execute: str,
    publish_uri_preview_fingerprint: str,
    operation_id: Optional[str] = None,
    adapters: Optional[PublishExecuteAdapters] = None,
) -> Dict[str, Any]:
    if (confirm_execute or "").strip() != CONFIRM_EXECUTE_TOKEN:
        raise PublishExecuteRejected("ace_publish_execute_confirm_required")

    validate_fingerprint_token(publish_uri_preview_fingerprint)
    session_id, candidate_sha, _record = resolve_active_registered_candidate()

    try:
        op = resolve_operation_id(operation_id)
    except ValueError as exc:
        if str(exc) == "ace_cancel_progress_operation_id_invalid":
            raise PublishExecuteRejected("ace_publish_execute_operation_id_invalid") from None
        raise

    try:
        assert_publish_execute_operation_id_available(op, session_id=session_id)
    except ValueError as exc:
        code = str(exc)
        if code in (
            "ace_publish_execute_operation_id_invalid",
            "ace_publish_execute_operation_id_already_used",
        ):
            raise PublishExecuteRejected(code) from None
        raise

    progress = PublishExecuteProgressWriter(session_id=session_id, operation_id=op)
    adapter = adapters or BlockedPublishExecuteAdapters()

    try:
        try_begin_local_dev_execute()
    except LocalDevExecuteBusyError as exc:
        progress.fail_step(STEP_REGISTERED_CANDIDATE, error="ace_local_dev_execute_in_progress")
        raise PublishExecuteRejected(
            "ace_local_dev_execute_in_progress",
            operation_id=op,
        ) from exc

    try:
        try:
            pre = _run_precheck_steps(
                progress,
                expected_fingerprint=publish_uri_preview_fingerprint,
            )
        except PublishPrecheckRejected as exc:
            step = STEP_PRODUCT_SCOPE
            if exc.code == "ace_publish_uri_candidate_fingerprint_stale":
                step = STEP_FINGERPRINT
            elif exc.code in (
                "ace_publish_candidate_not_registered",
                "ace_publish_candidate_registry_stale",
                "ace_active_dev_cycle_invalid",
                "ace_dev_cycle_none",
                "ace_dev_cycle_not_open",
            ):
                step = STEP_REGISTERED_CANDIDATE
            elif exc.code == "ace_publish_uri_candidate_preview_failed":
                step = STEP_FRESH_PREVIEW
            progress.fail_step(step, error=exc.code, internal_error=exc.message)
            raise PublishExecuteRejected(
                exc.code,
                operation_id=op,
                blockers=exc.blockers,
                message=exc.message,
            ) from None

        ctx = PublishExecuteContext(precheck=pre, operation_id=op)

        _run_stage(progress, STEP_ISOLATED_RELEASE, adapter.prepare_isolated_release, ctx)

        progress.begin_step(STEP_CANDIDATE_SHA_MATCH)
        plan_sha = str(
            (pre.execution_plan.get("verifiedSources") or {}).get("candidateBackendSha") or ""
        )
        if plan_sha != pre.candidate_backend_sha:
            progress.fail_step(
                STEP_CANDIDATE_SHA_MATCH,
                error="ace_publish_execute_candidate_sha_mismatch",
            )
            raise PublishExecuteRejected(
                "ace_publish_execute_candidate_sha_mismatch",
                operation_id=op,
            )
        progress.complete_step(STEP_CANDIDATE_SHA_MATCH)

        progress.begin_step(STEP_REAL_EXECUTE_GATE)
        if not publish_real_execute_permitted():
            progress.block_step(
                STEP_REAL_EXECUTE_GATE,
                error="ace_publish_execute_not_allowed",
                internal_error="ACE_LOCAL_DEV_ALLOW_REAL_PUBLISH_EXECUTE required; LOCAL layout blocked.",
            )
            raise PublishExecuteRejected("ace_publish_execute_not_allowed", operation_id=op)
        progress.complete_step(STEP_REAL_EXECUTE_GATE)

        _run_stage(
            progress,
            STEP_GIT_PUSH_BACKEND,
            adapter.git_push_backend,
            ctx,
            partial_on_fail=True,
        )
        _run_stage(progress, STEP_RENDER_WEB, adapter.wait_render_web, ctx, partial_on_fail=True)
        _run_stage(
            progress,
            STEP_RENDER_WORKER,
            adapter.wait_render_worker,
            ctx,
            partial_on_fail=True,
        )

        if pre.readiness.get("hasFrontendProductChanges"):
            _run_stage(
                progress,
                STEP_GITHUB_PAGES,
                adapter.deploy_github_pages,
                ctx,
                partial_on_fail=True,
            )
        else:
            progress.skip_step(STEP_GITHUB_PAGES, reason="no_frontend_product_changes")

        _run_stage(
            progress,
            STEP_PRODUCTION_VERIFY,
            adapter.verify_production,
            ctx,
            partial_on_fail=True,
        )
        _run_stage(
            progress,
            STEP_REGISTRY_UPDATE,
            adapter.update_production_registry,
            ctx,
            partial_on_fail=True,
        )

        progress.finish_success()
        return {
            "ok": True,
            "executeMode": "PUBLISH_CANDIDATE_EXECUTE",
            "realExecuteBlocked": False,
            "executePublishAvailable": True,
            "operationId": op,
            "sessionId": session_id,
            "candidateBackendSha": candidate_sha,
            "publishUriPreviewFingerprint": pre.fingerprint,
            "progressPath": str(progress.path),
            "note": "Execute completed via injected adapters (tests) or configured remote ports.",
        }
    except PublishExecuteRejected:
        raise
    except Exception as exc:
        logger.exception(
            "LOCAL_DEV_PUBLISH_EXECUTE_FAILED sessionId=%s operationId=%s",
            session_id,
            op,
        )
        progress.fail_step(STEP_PRODUCTION_VERIFY, error="ace_publish_execute_failed", internal_error=str(exc))
        raise PublishExecuteRejected(
            "ace_publish_execute_failed",
            operation_id=op,
            message=str(exc)[:240],
        ) from exc
    finally:
        end_local_dev_execute()
