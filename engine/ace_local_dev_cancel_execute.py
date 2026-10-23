"""Local cancel-development Execute (PowerShell engine, two-step confirm)."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from engine.ace_local_dev_cancel_preview import run_cancel_dev_preview_powershell
from engine.ace_local_dev_cancel_progress import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_PARTIAL,
    STEP_PRE_CHECKS,
    CancelExecuteProgressWriter,
    assert_operation_id_available,
    resolve_operation_id,
)
from engine.ace_local_dev_cycle_store import (
    resolve_active_dev_cycle,
    resolve_active_dev_session_id_for_api,
    resolve_sessions_root,
    write_active_dev_cycle_pointer,
)
from engine.ace_local_dev_execute_guard import (
    LocalDevExecuteBusyError,
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_powershell import run_cancel_execute_json
from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint
from engine.ace_local_dev_post_cancel_auto_start import (
    ensure_post_cancel_auto_start_record,
    run_post_cancel_auto_start,
)
from engine.ace_local_dev_real_execute_gate import is_real_ace_cancel_execute_guarded

logger = logging.getLogger(__name__)

CONFIRM_EXECUTE_TOKEN = "CANCEL"


class CancelExecuteRejected(ValueError):
    """Validation failure after progress reservation (includes operationId for clients)."""

    def __init__(self, code: str, *, operation_id: Optional[str] = None) -> None:
        super().__init__(code)
        self.operation_id = operation_id


def _allow_real_execute_from_env() -> bool:
    raw = (os.environ.get("ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _fail_prechecks(progress: CancelExecuteProgressWriter, exc: BaseException) -> None:
    progress.fail_step(STEP_PRE_CHECKS, error=str(exc))


def execute_cancel_development(
    *,
    confirm_execute: str,
    preview_fingerprint: str,
    operation_id: Optional[str] = None,
) -> Dict[str, Any]:
    if (confirm_execute or "").strip() != CONFIRM_EXECUTE_TOKEN:
        raise ValueError("ace_cancel_execute_confirm_required")

    fp = (preview_fingerprint or "").strip().lower()
    if len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp):
        raise ValueError("ace_cancel_preview_fingerprint_invalid")

    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        raise ValueError("ace_active_dev_cycle_invalid")
    session_id = resolution.session_id
    if not session_id:
        raise ValueError("ace_dev_cycle_none")
    if (
        resolution.cycle_status in ("CANCELLED", "CANCELLED_AWAITING_NEW_CYCLE")
        or "CYCLE_ALREADY_CANCELLED" in resolution.blockers
    ):
        raise ValueError("ace_dev_cycle_already_cancelled")
    if resolution.cycle_status != "OPEN":
        raise ValueError("ace_dev_cycle_not_open")

    try:
        op = resolve_operation_id(operation_id)
        assert_operation_id_available(op, session_id=session_id)
    except ValueError as exc:
        code = str(exc)
        if code == "ace_cancel_progress_operation_id_invalid":
            raise CancelExecuteRejected(code) from None
        if code == "ace_cancel_progress_operation_id_already_used":
            raise CancelExecuteRejected(code) from None
        raise

    progress = CancelExecuteProgressWriter(session_id=session_id, operation_id=op)
    progress.begin_step(STEP_PRE_CHECKS)

    try:
        if not _allow_real_execute_from_env() and is_real_ace_cancel_execute_guarded():
            raise CancelExecuteRejected("ace_cancel_execute_not_allowed", operation_id=op)

        raw_preview = run_cancel_dev_preview_powershell(session_id)
        if not raw_preview.get("ok"):
            raise CancelExecuteRejected("ace_cancel_preview_failed", operation_id=op)

        plan = raw_preview.get("plan") if isinstance(raw_preview.get("plan"), dict) else {}
        blockers = [str(b) for b in (plan.get("blockers") or [])]
        if blockers:
            raise CancelExecuteRejected(
                "ace_cancel_execute_blockers:" + ",".join(blockers),
                operation_id=op,
            )
        if not plan.get("canExecute"):
            raise CancelExecuteRejected("ace_cancel_execute_not_allowed", operation_id=op)

        fresh_fp = compute_preview_fingerprint(raw_preview)
        if fresh_fp.lower() != fp:
            raise CancelExecuteRejected("ace_cancel_preview_fingerprint_stale", operation_id=op)

        progress.complete_step(STEP_PRE_CHECKS)
    except CancelExecuteRejected as exc:
        _fail_prechecks(progress, exc)
        raise
    except RuntimeError as exc:
        _fail_prechecks(progress, exc)
        raise CancelExecuteRejected("ace_cancel_preview_failed", operation_id=op) from exc

    try:
        try_begin_local_dev_execute()
    except LocalDevExecuteBusyError:
        progress.fail_operation(error="ace_local_dev_execute_in_progress")
        raise CancelExecuteRejected("ace_local_dev_execute_in_progress", operation_id=op) from None

    try:
        result = run_cancel_execute_json(
            session_id,
            allow_real_execute=_allow_real_execute_from_env(),
            progress_path=str(progress.path),
            operation_id=op,
        )
    except RuntimeError:
        logger.exception("LOCAL_DEV_CANCEL_EXECUTE_FAILED sessionId=%s", session_id)
        progress.reload()
        if str(progress._state.get("operationStatus") or "") == STATUS_IN_PROGRESS:
            progress.fail_operation(error="ace_cancel_execute_failed")
        raise
    finally:
        end_local_dev_execute()

    progress.reload()

    if not result.get("ok"):
        err = str(result.get("error") or "ace_cancel_execute_failed")
        if err.startswith("REAL_EXECUTE_BLOCKED"):
            progress.fail_operation(error="ace_cancel_execute_not_allowed")
            raise CancelExecuteRejected("ace_cancel_execute_not_allowed", operation_id=op)
        progress.fail_operation(error=err)
        payload: Dict[str, Any] = {
            "ok": False,
            "error": "ace_cancel_execute_failed",
            "message": err[:500],
            "operationId": op,
        }
        recovery = result.get("recovery")
        if isinstance(recovery, dict):
            payload["recovery"] = {
                "backupRoot": recovery.get("backupRoot"),
                "recoveryGuideFile": recovery.get("recoveryMarkdown"),
            }
        return payload

    exec_status = None
    if isinstance(result.get("result"), dict):
        exec_status = str((result.get("result") or {}).get("status") or "")

    cancel_status = str(progress._state.get("cancelStatus") or STATUS_IN_PROGRESS)
    if exec_status == "PARTIAL" and cancel_status != STATUS_PARTIAL:
        progress.set_cancel_status(STATUS_PARTIAL)
        progress.finish(
            operation_status=STATUS_FAILED,
            cancel_status=STATUS_PARTIAL,
            error=progress._state.get("error"),
        )
    elif cancel_status == STATUS_COMPLETED:
        progress._state["operationStatus"] = STATUS_IN_PROGRESS
        progress._flush()
    elif cancel_status not in (STATUS_COMPLETED, STATUS_PARTIAL, STATUS_FAILED):
        progress.set_cancel_status(STATUS_COMPLETED)
        progress._state["operationStatus"] = STATUS_IN_PROGRESS
        progress._flush()

    try:
        write_active_dev_cycle_pointer(session_id=session_id)
    except ValueError as exc:
        logger.warning(
            "LOCAL_DEV_POINTER_REFRESH_AFTER_EXECUTE sessionId=%s err=%s",
            session_id,
            exc,
        )

    ensure_post_cancel_auto_start_record(
        resolve_sessions_root() / session_id,
        session_id,
    )
    progress.begin_post_cancel_auto_start()
    auto_start = run_post_cancel_auto_start(cancelled_session_id=session_id)
    progress.apply_post_cancel_auto_start_result(auto_start)

    after = resolve_active_dev_cycle()
    status = after.to_status_payload()
    progress.reload()
    return {
        "ok": True,
        "mode": "EXECUTE",
        "sessionId": session_id,
        "operationId": op,
        "executeResult": {
            "status": exec_status,
        },
        "postCancelAutoStart": auto_start,
        "activeCycle": status.get("activeCycle"),
        "canStartNewCycle": status.get("canStartNewCycle"),
    }
