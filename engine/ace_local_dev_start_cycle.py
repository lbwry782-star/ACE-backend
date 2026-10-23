"""Start a new local dev cycle: validate, baseline, atomic pointer."""
from __future__ import annotations

import logging
from typing import Any, Dict

from engine.ace_local_dev_cycle_store import (
    resolve_active_dev_cycle,
    write_active_dev_cycle_pointer,
)
from engine.ace_local_dev_powershell import run_baseline_validate_json, run_new_session_baseline_json

logger = logging.getLogger(__name__)

CONFIRM_START_TOKEN = "START"


def start_dev_cycle(*, confirm_start: str) -> Dict[str, Any]:
    if (confirm_start or "").strip() != CONFIRM_START_TOKEN:
        raise ValueError("ace_dev_cycle_start_confirm_required")

    resolution = resolve_active_dev_cycle()
    if resolution.cycle_status == "OPEN" and resolution.session_id:
        raise ValueError("ace_dev_cycle_open_exists")
    if resolution.cycle_status == "INVALID":
        raise ValueError("ace_active_dev_cycle_invalid")

    validate = run_baseline_validate_json()
    if not validate.get("ok"):
        raise RuntimeError("ace_dev_cycle_baseline_validate_failed")

    baseline = run_new_session_baseline_json()
    if not baseline.get("ok"):
        raise RuntimeError("ace_dev_cycle_baseline_failed")

    session_id = str(baseline.get("sessionId") or "").strip()
    if not session_id or str(baseline.get("status") or "").upper() != "READY":
        raise RuntimeError("ace_dev_cycle_baseline_not_ready")

    try:
        pointer_path = write_active_dev_cycle_pointer(session_id=session_id)
    except ValueError as exc:
        logger.exception("LOCAL_DEV_START_POINTER_FAILED sessionId=%s", session_id)
        raise RuntimeError(str(exc)) from exc

    after = resolve_active_dev_cycle()
    return {
        "ok": True,
        "mode": "START_CYCLE",
        "sessionId": session_id,
        "sessionRoot": baseline.get("sessionRoot"),
        "pointerPath": str(pointer_path),
        "activeCycle": after.to_status_payload().get("activeCycle"),
    }
