"""Shutdown gate hook for LOCAL dev route entry points (loopback API only)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from engine.ace_local_dev_publish_candidate_registry import resolve_ace_root
from engine.ace_local_dev_shutdown_gate import (
    local_dev_operation_start_allowed,
    read_managed_launcher_session,
    read_shutdown_gate_state,
    shutdown_gate_state_path,
)

logger = logging.getLogger(__name__)

ERROR_SHUTDOWN_GATE_DENIED = "ace_shutdown_gate_operations_frozen"
ERROR_SHUTDOWN_GATE_MISMATCH = "ace_shutdown_gate_session_mismatch"
ERROR_SHUTDOWN_GATE_SHUTTING_DOWN = "ace_shutdown_gate_shutting_down"
ERROR_SHUTDOWN_GATE_STATE_UNREADABLE = "ace_shutdown_gate_state_unreadable"


def local_dev_shutdown_blocks_new_operation(
    *,
    ace_root: Optional[Path] = None,
    expected_launcher_session: Optional[str] = None,
) -> Optional[str]:
    """
    Return a stable error code when a new LOCAL operation must not start, else None.
    Treats unreadable gate state file as a block (fail closed).
    """
    root = ace_root if ace_root is not None else resolve_ace_root()
    gate_path = shutdown_gate_state_path(root)
    if gate_path.is_file():
        state = read_shutdown_gate_state(root)
        if state is None:
            logger.warning("LOCAL_DEV_SHUTDOWN_GATE_UNREADABLE path=%s", gate_path)
            return ERROR_SHUTDOWN_GATE_STATE_UNREADABLE

    session = expected_launcher_session
    if not session:
        session = read_managed_launcher_session(root)

    allowed, reason = local_dev_operation_start_allowed(
        root,
        expected_launcher_session=session,
    )
    if allowed:
        return None
    if reason in (
        ERROR_SHUTDOWN_GATE_DENIED,
        "ace_shutdown_gate_operations_frozen",
    ):
        return ERROR_SHUTDOWN_GATE_DENIED
    if reason == "ace_shutdown_gate_session_mismatch":
        return ERROR_SHUTDOWN_GATE_MISMATCH
    if reason == "ace_shutdown_gate_shutting_down":
        return ERROR_SHUTDOWN_GATE_SHUTTING_DOWN
    return reason or ERROR_SHUTDOWN_GATE_DENIED


def shutdown_gate_denied_json_response(code: str) -> Tuple[Dict[str, Any], int]:
    messages = {
        ERROR_SHUTDOWN_GATE_DENIED: "LOCAL shutdown is in progress; new operations are temporarily frozen.",
        ERROR_SHUTDOWN_GATE_MISMATCH: "LOCAL shutdown session mismatch; operation refused.",
        ERROR_SHUTDOWN_GATE_SHUTTING_DOWN: "LOCAL shutdown is completing; new operations are not allowed.",
        ERROR_SHUTDOWN_GATE_STATE_UNREADABLE: "LOCAL shutdown gate state could not be read; new operations blocked.",
    }
    return (
        {
            "ok": False,
            "error": code,
            "message": messages.get(code, "LOCAL shutdown gate blocked this operation."),
            "shutdownGateBlocked": True,
        },
        503,
    )
