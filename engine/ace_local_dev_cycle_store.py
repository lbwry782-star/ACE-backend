"""
Local active dev cycle pointer (_backups/sessions/active-dev-cycle.json).

Read on each API request; pointer is primary. ACE_ORI_DEV_SESSION_ID is legacy fallback
only when the pointer file is absent (not a competing source of truth).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ACTIVE_CYCLE_FILENAME = "active-dev-cycle.json"
POINTER_SCHEMA_VERSION = 1
_SESSION_ID_PATTERN = re.compile(r"^ori-\d{8}-\d{6}-[0-9a-f]{8}$", re.IGNORECASE)


def resolve_sessions_root() -> Path:
    override = (os.environ.get("ACE_SESSIONS_ROOT") or "").strip()
    if override:
        return Path(override)
    backend_root = Path(__file__).resolve().parent.parent
    return backend_root.parent / "_backups" / "sessions"


def active_cycle_pointer_path(sessions_root: Optional[Path] = None) -> Path:
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    return root / ACTIVE_CYCLE_FILENAME


def _validate_session_id(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text or not _SESSION_ID_PATTERN.match(text):
        return None
    return text


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_session_record(session_root: Path) -> Dict[str, Any]:
    session_file = session_root / "session.json"
    meta = _read_json_file(session_file)
    if meta is None:
        return {"present": False, "ready": False, "status": None}
    status = str(meta.get("status") or "").strip().upper()
    return {
        "present": True,
        "ready": status == "READY",
        "status": status or None,
    }


def _load_cycle_status(session_root: Path) -> str:
    cycle_file = session_root / "cancel-dev-cycle.state.json"
    cycle = _read_json_file(cycle_file)
    if cycle is None:
        return "OPEN"
    return str(cycle.get("status") or "OPEN").strip().upper()


def is_cancelled_cycle_status(status: str) -> bool:
    text = (status or "").strip().upper()
    return text in ("CANCELLED", "CANCELLED_AWAITING_NEW_CYCLE")


def scan_open_cycle_session_ids(sessions_root: Path) -> List[str]:
    if not sessions_root.is_dir():
        return []
    open_ids: List[str] = []
    for entry in sessions_root.iterdir():
        if not entry.is_dir():
            continue
        sid = entry.name
        if not _validate_session_id(sid):
            continue
        if _load_cycle_status(entry) == "OPEN":
            open_ids.append(sid)
    return sorted(set(open_ids))


def write_active_dev_cycle_pointer(
    *,
    session_id: str,
    sessions_root: Optional[Path] = None,
) -> Path:
    """
    Atomic write of the active cycle pointer. Validates single OPEN cycle policy.
    Intended for tooling/tests — not invoked for real migration in this phase.
    """
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    sid = _validate_session_id(session_id)
    if not sid:
        raise ValueError("ace_active_cycle_session_id_invalid")

    session_root = root / sid
    session_info = _load_session_record(session_root)
    if not session_info["present"]:
        raise ValueError("ace_active_cycle_session_missing")
    if not session_info["ready"]:
        raise ValueError("ace_active_cycle_session_not_ready")

    open_ids = scan_open_cycle_session_ids(root)
    other_open = [x for x in open_ids if x.lower() != sid.lower()]
    if other_open:
        raise ValueError("ace_active_cycle_multiple_open")

    cycle_status = _load_cycle_status(session_root)
    payload = {
        "schemaVersion": POINTER_SCHEMA_VERSION,
        "sessionId": sid,
        "cycleStatus": cycle_status,
        "baselineReady": bool(session_info["ready"]),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    root.mkdir(parents=True, exist_ok=True)
    target = active_cycle_pointer_path(root)
    tmp = target.with_name(f"{target.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


@dataclass
class ActiveDevCycleResolution:
    session_id: Optional[str] = None
    cycle_status: str = "NONE"
    baseline_ready: bool = False
    source: str = "none"
    pointer_present: bool = False
    pointer_valid: bool = False
    blockers: List[str] = field(default_factory=list)
    pointer_errors: List[str] = field(default_factory=list)

    def to_status_payload(self) -> Dict[str, Any]:
        if self.cycle_status == "OPEN" and self.session_id:
            can_start = False
        elif self.cycle_status == "CANCELLED_AWAITING_NEW_CYCLE":
            can_start = False
        elif self.cycle_status == "INVALID" or "MULTIPLE_OPEN_CYCLES" in self.blockers:
            can_start = False
        else:
            can_start = True
        blockers = list(self.blockers)
        if self.cycle_status == "CANCELLED_AWAITING_NEW_CYCLE":
            blockers.append("POST_CANCEL_AUTO_START_IN_PROGRESS")
        return {
            "ok": True,
            "activeCycle": {
                "status": self.cycle_status,
                "sessionId": self.session_id,
                "baselineReady": self.baseline_ready,
                "source": self.source,
            },
            "blockers": blockers,
            "canStartNewCycle": can_start,
            "pointer": {
                "present": self.pointer_present,
                "valid": self.pointer_valid,
                "errors": list(self.pointer_errors),
            },
        }


def resolve_active_dev_cycle(
    sessions_root: Optional[Path] = None,
) -> ActiveDevCycleResolution:
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    result = ActiveDevCycleResolution()
    pointer_path = active_cycle_pointer_path(root)
    result.pointer_present = pointer_path.is_file()

    pointer_data = _read_json_file(pointer_path) if result.pointer_present else None
    env_sid = _validate_session_id((os.environ.get("ACE_ORI_DEV_SESSION_ID") or "").strip())

    if result.pointer_present:
        if pointer_data is None:
            result.pointer_errors.append("POINTER_JSON_INVALID")
            result.cycle_status = "INVALID"
            result.blockers.append("ACTIVE_CYCLE_POINTER_INVALID")
            return result
        if pointer_data.get("schemaVersion") != POINTER_SCHEMA_VERSION:
            result.pointer_errors.append("POINTER_SCHEMA_UNSUPPORTED")
        ptr_sid = _validate_session_id(str(pointer_data.get("sessionId") or ""))
        if not ptr_sid:
            result.pointer_errors.append("POINTER_SESSION_ID_INVALID")
        if result.pointer_errors:
            result.cycle_status = "INVALID"
            result.blockers.append("ACTIVE_CYCLE_POINTER_INVALID")
            return result

        assert ptr_sid is not None
        session_root = root / ptr_sid
        session_info = _load_session_record(session_root)
        if not session_info["present"]:
            result.pointer_errors.append("SESSION_JSON_MISSING")
        elif not session_info["ready"]:
            result.pointer_errors.append("SESSION_NOT_READY")

        cycle_status = _load_cycle_status(session_root) if session_info["present"] else "INVALID"
        ptr_cycle = str(pointer_data.get("cycleStatus") or "").strip().upper()
        if ptr_cycle and ptr_cycle != cycle_status:
            result.pointer_errors.append("POINTER_CYCLE_STATUS_STALE")

        open_ids = scan_open_cycle_session_ids(root)
        if len(open_ids) > 1:
            result.blockers.append("MULTIPLE_OPEN_CYCLES")
        if ptr_sid not in open_ids and cycle_status == "OPEN":
            result.pointer_errors.append("CYCLE_NOT_OPEN_ON_DISK")

        if result.pointer_errors or result.blockers:
            result.cycle_status = "INVALID"
            result.session_id = ptr_sid
            result.blockers.append("ACTIVE_CYCLE_POINTER_INVALID")
            return result

        result.pointer_valid = True
        result.session_id = ptr_sid
        result.cycle_status = cycle_status if cycle_status else "OPEN"
        result.baseline_ready = bool(session_info["ready"])
        result.source = "pointer"
        return result

    if env_sid:
        session_root = root / env_sid
        session_info = _load_session_record(session_root)
        if not session_info["present"]:
            result.blockers.append("LEGACY_ENV_SESSION_MISSING")
            result.cycle_status = "NONE"
            return result
        if not session_info["ready"]:
            result.blockers.append("LEGACY_ENV_SESSION_NOT_READY")
            result.cycle_status = "NONE"
            return result
        open_ids = scan_open_cycle_session_ids(root)
        if len(open_ids) > 1:
            result.blockers.append("MULTIPLE_OPEN_CYCLES")
            result.cycle_status = "INVALID"
            return result
        result.session_id = env_sid
        result.cycle_status = _load_cycle_status(session_root)
        result.baseline_ready = True
        result.source = "env_fallback"
        return result

    result.cycle_status = "NONE"
    return result


def resolve_active_dev_session_id_for_api(
    sessions_root: Optional[Path] = None,
) -> ActiveDevCycleResolution:
    resolution = resolve_active_dev_cycle(sessions_root=sessions_root)
    if resolution.cycle_status == "INVALID":
        return resolution
    if resolution.cycle_status == "NONE" or not resolution.session_id:
        return resolution
    if is_cancelled_cycle_status(resolution.cycle_status):
        resolution.blockers.append("CYCLE_ALREADY_CANCELLED")
    return resolution
