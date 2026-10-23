"""Server-side new dev cycle after successful cancel Execute (durable, idempotent)."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from engine.ace_local_dev_cycle_store import (
    _load_cycle_status,
    _read_json_file,
    _validate_session_id,
    resolve_active_dev_cycle,
    resolve_sessions_root,
    scan_open_cycle_session_ids,
    write_active_dev_cycle_pointer,
)
from engine.ace_local_dev_start_cycle import run_start_dev_cycle_core

logger = logging.getLogger(__name__)

POST_CANCEL_AUTO_START_SCHEMA = 1
POST_CANCEL_AUTO_START_FILENAME = "post-cancel-auto-start.json"
CANCEL_CYCLE_STATE_FILENAME = "cancel-dev-cycle.state.json"
AUTO_START_LOCK_FILENAME = ".post-cancel-auto-start.lock"

PHASE_COMPLETED = "COMPLETED"
PHASE_FAILED = "FAILED"
PHASE_IN_PROGRESS = "IN_PROGRESS"
PHASE_PENDING = "PENDING"

STATUS_CANCELLED = "CANCELLED"
STATUS_AWAITING = "CANCELLED_AWAITING_NEW_CYCLE"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_start_record_path(session_root: Path) -> Path:
    return session_root / POST_CANCEL_AUTO_START_FILENAME


def _read_auto_start_record(session_root: Path) -> Optional[Dict[str, Any]]:
    data = _read_json_file(_auto_start_record_path(session_root))
    return data if isinstance(data, dict) else None


def _write_auto_start_record(session_root: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["schemaVersion"] = POST_CANCEL_AUTO_START_SCHEMA
    payload["updatedAt"] = _utc_now_iso()
    path = _auto_start_record_path(session_root)
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _set_cancel_cycle_status(session_root: Path, status: str) -> None:
    path = session_root / CANCEL_CYCLE_STATE_FILENAME
    data = _read_json_file(path) or {"schemaVersion": 1}
    data["status"] = status
    if status == STATUS_AWAITING:
        data["awaitingNewCycleSince"] = _utc_now_iso()
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalize_cancelled_status(raw: str) -> str:
    text = (raw or "").strip().upper()
    if text == STATUS_AWAITING:
        return STATUS_AWAITING
    if text == STATUS_CANCELLED:
        return STATUS_CANCELLED
    return text


@contextmanager
def _sessions_auto_start_lock(sessions_root: Path, *, timeout_sec: float = 120.0) -> Iterator[None]:
    lock_path = sessions_root / AUTO_START_LOCK_FILENAME
    sessions_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    acquired = False
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()} at={_utc_now_iso()}\n")
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
    if not acquired:
        raise RuntimeError("ace_post_cancel_auto_start_lock_timeout")
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("POST_CANCEL_AUTO_START_LOCK_RELEASE_FAILED path=%s", lock_path)


def _completed_payload(
    *,
    cancelled_session_id: str,
    new_session_id: str,
    resumed: bool = False,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "phase": PHASE_COMPLETED,
        "cancelledSessionId": cancelled_session_id,
        "newSessionId": new_session_id,
        "resumed": resumed,
    }


def run_post_cancel_auto_start(
    *,
    cancelled_session_id: str,
    sessions_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Start a new dev cycle after cancel Execute succeeded. Idempotent and disk-backed.
    """
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    sid = _validate_session_id(cancelled_session_id)
    if not sid:
        raise ValueError("ace_post_cancel_auto_start_session_invalid")

    session_root = root / sid
    if not session_root.is_dir():
        raise ValueError("ace_post_cancel_auto_start_session_missing")

    with _sessions_auto_start_lock(root):
        resolution = resolve_active_dev_cycle(sessions_root=root)
        open_ids = scan_open_cycle_session_ids(root)

        record = _read_auto_start_record(session_root) or {
            "cancelledSessionId": sid,
            "phase": PHASE_PENDING,
            "newSessionId": None,
            "lastError": None,
        }

        existing_new = _validate_session_id(str(record.get("newSessionId") or ""))
        if record.get("phase") == PHASE_COMPLETED and existing_new:
            if resolution.cycle_status == "OPEN" and resolution.session_id == existing_new:
                return _completed_payload(
                    cancelled_session_id=sid,
                    new_session_id=existing_new,
                    resumed=True,
                )
            if existing_new in open_ids:
                write_active_dev_cycle_pointer(session_id=existing_new, sessions_root=root)
                _write_auto_start_record(
                    session_root,
                    {**record, "phase": PHASE_COMPLETED, "newSessionId": existing_new},
                )
                return _completed_payload(
                    cancelled_session_id=sid,
                    new_session_id=existing_new,
                    resumed=True,
                )

        if resolution.cycle_status == "OPEN" and resolution.session_id and resolution.session_id != sid:
            new_sid = resolution.session_id
            _write_auto_start_record(
                session_root,
                {
                    **record,
                    "phase": PHASE_COMPLETED,
                    "newSessionId": new_sid,
                    "lastError": None,
                },
            )
            _set_cancel_cycle_status(session_root, STATUS_CANCELLED)
            return _completed_payload(cancelled_session_id=sid, new_session_id=new_sid, resumed=True)

        if open_ids:
            if len(open_ids) == 1 and open_ids[0] != sid:
                new_sid = open_ids[0]
                _write_auto_start_record(
                    session_root,
                    {
                        **record,
                        "phase": PHASE_COMPLETED,
                        "newSessionId": new_sid,
                        "lastError": None,
                    },
                )
                write_active_dev_cycle_pointer(session_id=new_sid, sessions_root=root)
                return _completed_payload(cancelled_session_id=sid, new_session_id=new_sid, resumed=True)
            raise RuntimeError("ace_post_cancel_auto_start_unexpected_open_cycles")

        cycle_status = _normalize_cancelled_status(_load_cycle_status(session_root))
        if cycle_status not in (STATUS_CANCELLED, STATUS_AWAITING):
            raise RuntimeError("ace_post_cancel_auto_start_cycle_not_cancelled")

        _set_cancel_cycle_status(session_root, STATUS_AWAITING)
        write_active_dev_cycle_pointer(session_id=sid, sessions_root=root)
        _write_auto_start_record(
            session_root,
            {
                **record,
                "cancelledSessionId": sid,
                "phase": PHASE_IN_PROGRESS,
                "newSessionId": None,
                "lastError": None,
            },
        )

        try:
            start_payload = run_start_dev_cycle_core(sessions_root=root)
        except Exception as exc:
            err = str(exc)[:500]
            logger.exception(
                "POST_CANCEL_AUTO_START_FAILED cancelledSessionId=%s",
                sid,
            )
            _write_auto_start_record(
                session_root,
                {
                    **record,
                    "cancelledSessionId": sid,
                    "phase": PHASE_FAILED,
                    "newSessionId": None,
                    "lastError": err,
                },
            )
            _set_cancel_cycle_status(session_root, STATUS_AWAITING)
            write_active_dev_cycle_pointer(session_id=sid, sessions_root=root)
            return {
                "ok": False,
                "phase": PHASE_FAILED,
                "cancelledSessionId": sid,
                "newSessionId": None,
                "error": "ace_post_cancel_auto_start_failed",
                "message": err,
            }

        new_sid = _validate_session_id(str(start_payload.get("sessionId") or ""))
        if not new_sid:
            _write_auto_start_record(
                session_root,
                {
                    **record,
                    "phase": PHASE_FAILED,
                    "lastError": "ace_dev_cycle_baseline_not_ready",
                },
            )
            _set_cancel_cycle_status(session_root, STATUS_AWAITING)
            write_active_dev_cycle_pointer(session_id=sid, sessions_root=root)
            return {
                "ok": False,
                "phase": PHASE_FAILED,
                "cancelledSessionId": sid,
                "error": "ace_post_cancel_auto_start_failed",
                "message": "ace_dev_cycle_baseline_not_ready",
            }

        _set_cancel_cycle_status(session_root, STATUS_CANCELLED)
        _write_auto_start_record(
            session_root,
            {
                **record,
                "cancelledSessionId": sid,
                "phase": PHASE_COMPLETED,
                "newSessionId": new_sid,
                "lastError": None,
            },
        )
        return _completed_payload(cancelled_session_id=sid, new_session_id=new_sid, resumed=False)


def try_resume_post_cancel_auto_start(
    sessions_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Recover in-flight or failed auto-start after restart or client disconnect.
    Returns payload when a resume attempt ran; None when idle.
    """
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    if not root.is_dir():
        return None

    resolution = resolve_active_dev_cycle(sessions_root=root)
    if resolution.cycle_status == "OPEN":
        return None

    ptr_sid = resolution.session_id
    if not ptr_sid:
        return None

    session_root = root / ptr_sid
    record = _read_auto_start_record(session_root)
    if not record:
        cycle = _normalize_cancelled_status(_load_cycle_status(session_root))
        if cycle in (STATUS_CANCELLED, STATUS_AWAITING) and resolution.cycle_status in (
            STATUS_CANCELLED,
            STATUS_AWAITING,
        ):
            return run_post_cancel_auto_start(cancelled_session_id=ptr_sid, sessions_root=root)
        return None

    phase = str(record.get("phase") or "").upper()
    if phase == PHASE_COMPLETED:
        return None
    if phase in (PHASE_PENDING, PHASE_IN_PROGRESS, PHASE_FAILED):
        return run_post_cancel_auto_start(cancelled_session_id=ptr_sid, sessions_root=root)
    return None
