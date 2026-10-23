"""Durable on-disk progress for cancel Execute (atomic JSON, read-only API)."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_cycle_store import _validate_session_id, resolve_sessions_root

PROGRESS_SCHEMA = 1
PROGRESS_FILENAME_PREFIX = "cancel-execute-progress-"

PHASE_CANCEL_EXECUTE = "CANCEL_EXECUTE"
PHASE_POST_CANCEL_AUTO_START = "POST_CANCEL_AUTO_START"

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_PARTIAL = "PARTIAL"
STATUS_PENDING = "PENDING"

STEP_PRE_CHECKS = "pre_checks"
STEP_BACKUP_CREATION = "backup_creation"
STEP_BACKEND_RESTORE = "backend_restore"
STEP_FRONTEND_RESTORE = "frontend_restore"
STEP_ENV_RESTORE = "env_restore"
STEP_FINAL_VERIFY = "final_verify"
STEP_CLOSE_CANCEL_CYCLE = "close_cancel_cycle"
STEP_POST_CANCEL_AUTO_START = "post_cancel_auto_start"

_PATH_LIKE = re.compile(
    r"([A-Za-z]:\\|\\\\|/|\\|\.env|_backups\\|cancel-dev-backups)",
    re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress_path(session_root: Path, operation_id: str) -> Path:
    op = validate_operation_id(operation_id)
    return session_root / f"{PROGRESS_FILENAME_PREFIX}{op}.json"


def validate_operation_id(value: str) -> str:
    text = (value or "").strip().lower()
    if len(text) != 32 or not all(c in "0123456789abcdef" for c in text):
        raise ValueError("ace_cancel_progress_operation_id_invalid")
    return text


def new_operation_id() -> str:
    return uuid.uuid4().hex


def resolve_operation_id(client_value: Optional[str]) -> str:
    raw = (client_value or "").strip()
    if not raw:
        return new_operation_id()
    return validate_operation_id(raw)


def assert_operation_id_available(
    operation_id: str,
    *,
    session_id: str,
    sessions_root: Optional[Path] = None,
) -> None:
    op = validate_operation_id(operation_id)
    sid = _validate_session_id(session_id)
    if not sid:
        raise ValueError("ace_cancel_progress_session_invalid")
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    existing = find_progress_file(op, sessions_root=root)
    if existing is not None:
        raise ValueError("ace_cancel_progress_operation_id_already_used")
    path = _progress_path(root / sid, op)
    if path.is_file():
        raise ValueError("ace_cancel_progress_operation_id_already_used")


def sanitize_progress_error(message: Optional[str], *, max_len: int = 240) -> Optional[str]:
    if message is None:
        return None
    text = str(message).strip()
    if not text:
        return None
    text = _PATH_LIKE.sub("[path]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _default_post_cancel_auto_start() -> Dict[str, Any]:
    return {
        "phase": STATUS_PENDING,
        "newSessionId": None,
        "lastError": None,
    }


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _read_progress_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


class CancelExecuteProgressWriter:
    """In-process writer bound to one operation under a session folder."""

    def __init__(self, *, session_id: str, operation_id: str, sessions_root: Optional[Path] = None) -> None:
        sid = _validate_session_id(session_id)
        if not sid:
            raise ValueError("ace_cancel_progress_session_invalid")
        self.operation_id = validate_operation_id(operation_id)
        self.session_id = sid
        root = sessions_root if sessions_root is not None else resolve_sessions_root()
        self._path = _progress_path(root / sid, self.operation_id)
        now = _utc_now_iso()
        self._state: Dict[str, Any] = {
            "schemaVersion": PROGRESS_SCHEMA,
            "operationId": self.operation_id,
            "sessionId": self.session_id,
            "operationStatus": STATUS_IN_PROGRESS,
            "cancelStatus": STATUS_IN_PROGRESS,
            "phase": PHASE_CANCEL_EXECUTE,
            "currentStep": None,
            "completedSteps": [],
            "postCancelAutoStart": _default_post_cancel_auto_start(),
            "startedAt": now,
            "updatedAt": now,
            "error": None,
        }
        self._flush()

    @property
    def path(self) -> Path:
        return self._path

    def _flush(self) -> None:
        self._state["updatedAt"] = _utc_now_iso()
        _write_atomic(self._path, self._state)

    def begin_step(self, step_id: str) -> None:
        sid = (step_id or "").strip()
        if not sid:
            raise ValueError("ace_cancel_progress_step_invalid")
        now = _utc_now_iso()
        self._state["currentStep"] = {
            "id": sid,
            "status": STATUS_IN_PROGRESS,
            "startedAt": now,
        }
        self._flush()

    def complete_step(self, step_id: str, *, status: str = "SUCCESS") -> None:
        sid = (step_id or "").strip()
        current = self._state.get("currentStep")
        if isinstance(current, dict) and current.get("id") == sid:
            started = str(current.get("startedAt") or _utc_now_iso())
        else:
            started = _utc_now_iso()
        finished = _utc_now_iso()
        steps: List[Dict[str, Any]] = list(self._state.get("completedSteps") or [])
        steps.append(
            {
                "id": sid,
                "status": status,
                "startedAt": started,
                "finishedAt": finished,
            }
        )
        self._state["completedSteps"] = steps
        self._state["currentStep"] = None
        self._flush()

    def fail_step(self, step_id: str, *, error: Optional[str] = None) -> None:
        self.complete_step(step_id, status="FAILED")
        self._state["cancelStatus"] = STATUS_FAILED
        self._state["operationStatus"] = STATUS_FAILED
        self._state["error"] = sanitize_progress_error(error)
        self._flush()

    def set_cancel_status(self, status: str) -> None:
        self._state["cancelStatus"] = (status or "").strip() or STATUS_IN_PROGRESS
        self._flush()

    def set_phase(self, phase: str) -> None:
        self._state["phase"] = (phase or "").strip() or PHASE_CANCEL_EXECUTE
        self._flush()

    def begin_post_cancel_auto_start(self) -> None:
        self.set_phase(PHASE_POST_CANCEL_AUTO_START)
        self.begin_step(STEP_POST_CANCEL_AUTO_START)
        block = dict(self._state.get("postCancelAutoStart") or _default_post_cancel_auto_start())
        block["phase"] = STATUS_IN_PROGRESS
        block["lastError"] = None
        self._state["postCancelAutoStart"] = block
        self._flush()

    def apply_post_cancel_auto_start_result(self, payload: Dict[str, Any]) -> None:
        block = dict(self._state.get("postCancelAutoStart") or _default_post_cancel_auto_start())
        phase = str(payload.get("phase") or "").upper()
        if payload.get("ok"):
            block["phase"] = STATUS_COMPLETED
            new_sid = payload.get("newSessionId")
            block["newSessionId"] = str(new_sid) if new_sid else None
            block["lastError"] = None
            self.complete_step(STEP_POST_CANCEL_AUTO_START)
            self._state["operationStatus"] = STATUS_COMPLETED
            self._state["error"] = None
        else:
            block["phase"] = STATUS_FAILED
            block["newSessionId"] = None
            block["lastError"] = sanitize_progress_error(
                str(payload.get("message") or payload.get("error") or "ace_post_cancel_auto_start_failed")
            )
            self.complete_step(STEP_POST_CANCEL_AUTO_START, status="FAILED")
            self._state["operationStatus"] = STATUS_FAILED
            self._state["error"] = block["lastError"]
        self._state["postCancelAutoStart"] = block
        self._flush()

    def finish(
        self,
        *,
        operation_status: str,
        error: Optional[str] = None,
        cancel_status: Optional[str] = None,
    ) -> None:
        self._state["operationStatus"] = operation_status
        if cancel_status:
            self._state["cancelStatus"] = cancel_status
        self._state["currentStep"] = None
        if error:
            self._state["error"] = sanitize_progress_error(error)
        self._flush()

    def fail_operation(self, *, error: Optional[str] = None, partial: bool = False) -> None:
        cancel = STATUS_PARTIAL if partial else STATUS_FAILED
        self.finish(
            operation_status=cancel,
            error=error,
            cancel_status=cancel,
        )

    def reload(self) -> None:
        data = _read_progress_file(self._path)
        if isinstance(data, dict):
            self._state = data


def find_progress_file(
    operation_id: str,
    *,
    sessions_root: Optional[Path] = None,
    session_id_hint: Optional[str] = None,
) -> Optional[Path]:
    op = validate_operation_id(operation_id)
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    hint = _validate_session_id(session_id_hint or "")
    if hint:
        path = _progress_path(root / hint, op)
        if path.is_file():
            return path
    pattern = f"{PROGRESS_FILENAME_PREFIX}{op}.json"
    for path in root.glob(f"*/{pattern}"):
        if path.is_file():
            return path
    return None


def load_public_progress(
    operation_id: str,
    *,
    sessions_root: Optional[Path] = None,
    session_id_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = find_progress_file(
        operation_id,
        sessions_root=sessions_root,
        session_id_hint=session_id_hint,
    )
    if path is None:
        return None
    data = _read_progress_file(path)
    if not data:
        return None
    if str(data.get("operationId") or "").lower() != validate_operation_id(operation_id):
        return None
    return public_progress_payload(data)


def public_progress_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    err = sanitize_progress_error(raw.get("error"))
    current = raw.get("currentStep")
    if current is not None and not isinstance(current, dict):
        current = None
    completed = raw.get("completedSteps")
    if not isinstance(completed, list):
        completed = []
    safe_completed: List[Dict[str, Any]] = []
    for item in completed:
        if not isinstance(item, dict):
            continue
        safe_completed.append(
            {
                "id": str(item.get("id") or ""),
                "status": str(item.get("status") or ""),
                "startedAt": item.get("startedAt"),
                "finishedAt": item.get("finishedAt"),
            }
        )
    safe_current = None
    if isinstance(current, dict):
        safe_current = {
            "id": str(current.get("id") or ""),
            "status": str(current.get("status") or ""),
            "startedAt": current.get("startedAt"),
        }
    pc_raw = raw.get("postCancelAutoStart")
    post_cancel: Dict[str, Any] = {
        "phase": STATUS_PENDING,
        "newSessionId": None,
        "lastError": None,
    }
    if isinstance(pc_raw, dict):
        post_cancel = {
            "phase": str(pc_raw.get("phase") or STATUS_PENDING),
            "newSessionId": pc_raw.get("newSessionId"),
            "lastError": sanitize_progress_error(pc_raw.get("lastError")),
        }
    return {
        "ok": True,
        "operationId": str(raw.get("operationId") or ""),
        "sessionId": str(raw.get("sessionId") or ""),
        "operationStatus": str(raw.get("operationStatus") or ""),
        "cancelStatus": str(raw.get("cancelStatus") or raw.get("operationStatus") or ""),
        "phase": str(raw.get("phase") or ""),
        "currentStep": safe_current,
        "completedSteps": safe_completed,
        "postCancelAutoStart": post_cancel,
        "startedAt": raw.get("startedAt"),
        "updatedAt": raw.get("updatedAt"),
        "error": err,
    }
