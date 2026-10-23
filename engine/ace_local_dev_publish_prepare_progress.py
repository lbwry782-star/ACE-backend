"""Durable on-disk progress for publish prepare (simulation only; no deploy)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_cancel_progress import sanitize_progress_error, validate_operation_id
from engine.ace_local_dev_cycle_store import _validate_session_id, resolve_sessions_root

PROGRESS_SCHEMA = 1
PROGRESS_FILENAME_PREFIX = "publish-prepare-progress-"

PHASE_PUBLISH_PREPARE_SIMULATE = "PUBLISH_PREPARE_SIMULATE"

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_SIMULATED = "SIMULATED"

STEP_REGISTERED_CANDIDATE = "registered_candidate"
STEP_FRESH_PREVIEW = "fresh_preview"
STEP_FINGERPRINT = "fingerprint_match"
STEP_PRODUCT_SCOPE = "product_scope"
STEP_WORKER_COUPLING = "worker_coupling"
STEP_EXECUTION_PLAN = "execution_plan"
STEP_SIMULATE_GIT_BACKEND = "simulate_git_push_backend"
STEP_SIMULATE_GIT_FRONTEND = "simulate_git_push_frontend"
STEP_SIMULATE_RENDER = "simulate_render_deploy"
STEP_SIMULATE_GITHUB_PAGES = "simulate_github_pages"
STEP_SIMULATE_REGISTRY = "simulate_production_registry_update"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress_path(session_root: Path, operation_id: str) -> Path:
    op = validate_operation_id(operation_id)
    return session_root / f"{PROGRESS_FILENAME_PREFIX}{op}.json"


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


def assert_publish_prepare_operation_id_available(
    operation_id: str,
    *,
    session_id: str,
    sessions_root: Optional[Path] = None,
) -> None:
    op = validate_operation_id(operation_id)
    sid = _validate_session_id(session_id)
    if not sid:
        raise ValueError("ace_publish_prepare_session_invalid")
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    path = _progress_path(root / sid, op)
    if path.is_file():
        raise ValueError("ace_publish_prepare_operation_id_already_used")


class PublishPrepareProgressWriter:
    """Journal for one publish-prepare simulation under a session folder."""

    def __init__(
        self,
        *,
        session_id: str,
        operation_id: str,
        sessions_root: Optional[Path] = None,
    ) -> None:
        sid = _validate_session_id(session_id)
        if not sid:
            raise ValueError("ace_publish_prepare_session_invalid")
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
            "prepareStatus": STATUS_IN_PROGRESS,
            "phase": PHASE_PUBLISH_PREPARE_SIMULATE,
            "simulationOnly": True,
            "realExecuteBlocked": True,
            "currentStep": None,
            "completedSteps": [],
            "startedAt": now,
            "updatedAt": now,
            "error": None,
            "executionPlan": None,
        }
        self._flush()

    @property
    def path(self) -> Path:
        return self._path

    def _flush(self) -> None:
        self._state["updatedAt"] = _utc_now_iso()
        _write_atomic(self._path, self._state)

    def set_execution_plan(self, plan: Dict[str, Any]) -> None:
        self._state["executionPlan"] = plan
        self._flush()

    def begin_step(self, step_id: str) -> None:
        sid = (step_id or "").strip()
        if not sid:
            raise ValueError("ace_publish_prepare_step_invalid")
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
        self._state["prepareStatus"] = STATUS_FAILED
        self._state["operationStatus"] = STATUS_FAILED
        self._state["error"] = sanitize_progress_error(error)
        self._flush()

    def finish_success(self) -> None:
        self._state["prepareStatus"] = STATUS_COMPLETED
        self._state["operationStatus"] = STATUS_COMPLETED
        self._state["currentStep"] = None
        self._state["error"] = None
        self._flush()

    def fail_operation(self, *, error: Optional[str] = None) -> None:
        self._state["prepareStatus"] = STATUS_FAILED
        self._state["operationStatus"] = STATUS_FAILED
        self._state["currentStep"] = None
        self._state["error"] = sanitize_progress_error(error)
        self._flush()

    def reload(self) -> None:
        data = _read_progress_file(self._path)
        if isinstance(data, dict):
            self._state = data


def find_publish_prepare_progress_file(
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


def load_public_publish_prepare_progress(
    operation_id: str,
    *,
    sessions_root: Optional[Path] = None,
    session_id_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    path = find_publish_prepare_progress_file(
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
    return public_publish_prepare_progress_payload(data)


def public_publish_prepare_progress_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    err = sanitize_progress_error(raw.get("error"))
    current = raw.get("currentStep")
    if current is not None and not isinstance(current, dict):
        current = None
    plan = raw.get("executionPlan")
    if plan is not None and not isinstance(plan, dict):
        plan = None
    return {
        "ok": True,
        "operationId": str(raw.get("operationId") or ""),
        "sessionId": str(raw.get("sessionId") or ""),
        "operationStatus": str(raw.get("operationStatus") or ""),
        "prepareStatus": str(raw.get("prepareStatus") or ""),
        "phase": str(raw.get("phase") or ""),
        "simulationOnly": bool(raw.get("simulationOnly")),
        "realExecuteBlocked": bool(raw.get("realExecuteBlocked", True)),
        "currentStep": current,
        "completedSteps": list(raw.get("completedSteps") or []),
        "startedAt": raw.get("startedAt"),
        "updatedAt": raw.get("updatedAt"),
        "error": err,
        "executionPlanSummary": _execution_plan_summary(plan) if plan else None,
    }


def _execution_plan_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
    verified = plan.get("verifiedSources") if isinstance(plan.get("verifiedSources"), dict) else {}
    return {
        "candidateBackendSha": verified.get("candidateBackendSha"),
        "productionBackendWeb": verified.get("productionBackendWeb"),
        "backendProductCommitCount": len(
            (plan.get("allowedProductCommits") or {}).get("backend") or []
        ),
        "frontendProductCommitCount": len(
            (plan.get("allowedProductCommits") or {}).get("frontend") or []
        ),
        "simulationOnly": bool(plan.get("simulationOnly")),
    }
