"""Local cancel-development Execute (PowerShell engine, two-step confirm)."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from engine.ace_local_dev_cancel_preview import run_cancel_dev_preview_powershell
from engine.ace_local_dev_cycle_store import (
    resolve_active_dev_session_id_for_api,
    write_active_dev_cycle_pointer,
)
from engine.ace_local_dev_execute_guard import (
    LocalDevExecuteBusyError,
    end_local_dev_execute,
    try_begin_local_dev_execute,
)
from engine.ace_local_dev_powershell import run_cancel_execute_json
from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint

logger = logging.getLogger(__name__)

CONFIRM_EXECUTE_TOKEN = "CANCEL"


def _allow_real_execute_from_env() -> bool:
    raw = (os.environ.get("ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def execute_cancel_development(
    *,
    confirm_execute: str,
    preview_fingerprint: str,
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
    if resolution.cycle_status == "CANCELLED" or "CYCLE_ALREADY_CANCELLED" in resolution.blockers:
        raise ValueError("ace_dev_cycle_already_cancelled")
    if resolution.cycle_status != "OPEN":
        raise ValueError("ace_dev_cycle_not_open")

    raw_preview = run_cancel_dev_preview_powershell(session_id)
    if not raw_preview.get("ok"):
        raise RuntimeError("ace_cancel_preview_failed")

    plan = raw_preview.get("plan") if isinstance(raw_preview.get("plan"), dict) else {}
    blockers = [str(b) for b in (plan.get("blockers") or [])]
    if blockers:
        raise ValueError("ace_cancel_execute_blockers:" + ",".join(blockers))
    if not plan.get("canExecute"):
        raise ValueError("ace_cancel_execute_not_allowed")

    fresh_fp = compute_preview_fingerprint(raw_preview)
    if fresh_fp.lower() != fp:
        raise ValueError("ace_cancel_preview_fingerprint_stale")

    try:
        try_begin_local_dev_execute()
    except LocalDevExecuteBusyError:
        raise ValueError("ace_local_dev_execute_in_progress") from None

    try:
        result = run_cancel_execute_json(
            session_id,
            allow_real_execute=_allow_real_execute_from_env(),
        )
    except RuntimeError:
        logger.exception("LOCAL_DEV_CANCEL_EXECUTE_FAILED sessionId=%s", session_id)
        raise
    finally:
        end_local_dev_execute()

    if not result.get("ok"):
        err = str(result.get("error") or "ace_cancel_execute_failed")
        payload: Dict[str, Any] = {
            "ok": False,
            "error": "ace_cancel_execute_failed",
            "message": err[:500],
        }
        recovery = result.get("recovery")
        if isinstance(recovery, dict):
            payload["recovery"] = {
                "backupRoot": recovery.get("backupRoot"),
                "recoveryGuideFile": recovery.get("recoveryMarkdown"),
            }
        return payload

    try:
        write_active_dev_cycle_pointer(session_id=session_id)
    except ValueError as exc:
        logger.warning(
            "LOCAL_DEV_POINTER_REFRESH_AFTER_EXECUTE sessionId=%s err=%s",
            session_id,
            exc,
        )

    after = resolve_active_dev_session_id_for_api()
    return {
        "ok": True,
        "mode": "EXECUTE",
        "sessionId": session_id,
        "executeResult": {
            "status": (result.get("result") or {}).get("status") if isinstance(result.get("result"), dict) else None,
        },
        "activeCycle": after.to_status_payload().get("activeCycle"),
        "canStartNewCycle": after.to_status_payload().get("canStartNewCycle"),
    }
