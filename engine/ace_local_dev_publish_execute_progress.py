"""Durable progress for full Publish Execute (prechecks + release + deploy + verify)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_cancel_progress import sanitize_progress_error, validate_operation_id
from engine.ace_local_dev_cycle_store import _validate_session_id, resolve_sessions_root

PROGRESS_SCHEMA = 2
PROGRESS_FILENAME_PREFIX = "publish-execute-progress-"

PHASE_PUBLISH_EXECUTE = "PUBLISH_EXECUTE"

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_PARTIAL = "PARTIAL"
STATUS_SKIPPED = "SKIPPED"
STATUS_BLOCKED = "BLOCKED"

STEP_REGISTERED_CANDIDATE = "registered_candidate"
STEP_FRESH_PREVIEW = "fresh_preview"
STEP_FINGERPRINT = "fingerprint_match"
STEP_PRODUCT_SCOPE = "product_scope"
STEP_WORKER_COUPLING = "worker_coupling"
STEP_EXECUTION_PLAN = "execution_plan"
STEP_ISOLATED_RELEASE = "isolated_release_prepare"
STEP_CANDIDATE_SHA_MATCH = "candidate_sha_match"
STEP_REAL_EXECUTE_GATE = "real_execute_gate"
STEP_GIT_PUSH_BACKEND = "git_push_backend"
STEP_RENDER_WEB = "render_web_deploy"
STEP_RENDER_WORKER = "render_worker_deploy"
STEP_GITHUB_PAGES = "github_pages_deploy"
STEP_PRODUCTION_VERIFY = "production_verify"
STEP_REGISTRY_UPDATE = "production_registry_update"

STEP_UI_HE: Dict[str, str] = {
    STEP_REGISTERED_CANDIDATE: "בודק את השינויים...",
    STEP_FRESH_PREVIEW: "בודק את השינויים...",
    STEP_FINGERPRINT: "מוודא שהשינויים בטוחים לפרסום...",
    STEP_PRODUCT_SCOPE: "מוודא שהשינויים בטוחים לפרסום...",
    STEP_WORKER_COUPLING: "מוודא שהשינויים בטוחים לפרסום...",
    STEP_EXECUTION_PLAN: "מוודא שהשינויים בטוחים לפרסום...",
    STEP_ISOLATED_RELEASE: "מכין את הגרסה החדשה...",
    STEP_CANDIDATE_SHA_MATCH: "מכין את הגרסה החדשה...",
    STEP_REAL_EXECUTE_GATE: "מכין את הגרסה החדשה...",
    STEP_GIT_PUSH_BACKEND: "מעלה את השינויים...",
    STEP_RENDER_WEB: "מעדכן את האתר והשירותים...",
    STEP_RENDER_WORKER: "מעדכן את האתר והשירותים...",
    STEP_GITHUB_PAGES: "מעדכן את האתר והשירותים...",
    STEP_PRODUCTION_VERIFY: "מוודא שהפרסום הושלם...",
    STEP_REGISTRY_UPDATE: "מוודא שהפרסום הושלם...",
}

ORDERED_STEPS: List[str] = [
    STEP_REGISTERED_CANDIDATE,
    STEP_FRESH_PREVIEW,
    STEP_FINGERPRINT,
    STEP_PRODUCT_SCOPE,
    STEP_WORKER_COUPLING,
    STEP_EXECUTION_PLAN,
    STEP_ISOLATED_RELEASE,
    STEP_CANDIDATE_SHA_MATCH,
    STEP_REAL_EXECUTE_GATE,
    STEP_GIT_PUSH_BACKEND,
    STEP_RENDER_WEB,
    STEP_RENDER_WORKER,
    STEP_GITHUB_PAGES,
    STEP_PRODUCTION_VERIFY,
    STEP_REGISTRY_UPDATE,
]


def step_ui_message(step_id: str) -> str:
    return STEP_UI_HE.get(step_id, "מעבד...")


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


def assert_publish_execute_operation_id_available(
    operation_id: str,
    *,
    session_id: str,
    sessions_root: Optional[Path] = None,
) -> None:
    op = validate_operation_id(operation_id)
    sid = _validate_session_id(session_id)
    if not sid:
        raise ValueError("ace_publish_execute_session_invalid")
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    path = _progress_path(root / sid, op)
    if path.is_file():
        raise ValueError("ace_publish_execute_operation_id_already_used")


class PublishExecuteProgressWriter:
    def __init__(
        self,
        *,
        session_id: str,
        operation_id: str,
        sessions_root: Optional[Path] = None,
    ) -> None:
        sid = _validate_session_id(session_id)
        if not sid:
            raise ValueError("ace_publish_execute_session_invalid")
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
            "publishStatus": STATUS_IN_PROGRESS,
            "phase": PHASE_PUBLISH_EXECUTE,
            "realExecuteBlockedByDefault": True,
            "currentStep": None,
            "completedSteps": [],
            "stepCatalog": [
                {"id": s, "uiMessageHe": step_ui_message(s)} for s in ORDERED_STEPS
            ],
            "partialFailure": None,
            "executionPlan": None,
            "startedAt": now,
            "updatedAt": now,
            "error": None,
            "internalError": None,
        }
        self._flush()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    def _flush(self) -> None:
        self._state["updatedAt"] = _utc_now_iso()
        _write_atomic(self._path, self._state)

    def set_execution_plan(self, plan: Dict[str, Any]) -> None:
        self._state["executionPlan"] = plan
        self._flush()

    def begin_step(self, step_id: str) -> None:
        sid = (step_id or "").strip()
        now = _utc_now_iso()
        self._state["currentStep"] = {
            "id": sid,
            "status": STATUS_IN_PROGRESS,
            "uiMessageHe": step_ui_message(sid),
            "startedAt": now,
        }
        self._flush()

    def complete_step(
        self,
        step_id: str,
        *,
        status: str = STATUS_COMPLETED,
        internal_note: Optional[str] = None,
    ) -> None:
        sid = (step_id or "").strip()
        current = self._state.get("currentStep")
        if isinstance(current, dict) and current.get("id") == sid:
            started = str(current.get("startedAt") or _utc_now_iso())
        else:
            started = _utc_now_iso()
        finished = _utc_now_iso()
        row: Dict[str, Any] = {
            "id": sid,
            "status": status,
            "uiMessageHe": step_ui_message(sid),
            "startedAt": started,
            "finishedAt": finished,
        }
        if internal_note:
            row["internalNote"] = sanitize_progress_error(internal_note, max_len=120)
        steps: List[Dict[str, Any]] = list(self._state.get("completedSteps") or [])
        steps.append(row)
        self._state["completedSteps"] = steps
        self._state["currentStep"] = None
        self._flush()

    def skip_step(self, step_id: str, *, reason: str = "not_required") -> None:
        self.begin_step(step_id)
        self.complete_step(step_id, status=STATUS_SKIPPED, internal_note=reason)

    def fail_step(
        self,
        step_id: str,
        *,
        error: Optional[str] = None,
        internal_error: Optional[str] = None,
        partial: bool = False,
    ) -> None:
        self.complete_step(step_id, status=STATUS_FAILED, internal_note=internal_error or error)
        self._state["publishStatus"] = STATUS_PARTIAL if partial else STATUS_FAILED
        self._state["operationStatus"] = STATUS_PARTIAL if partial else STATUS_FAILED
        self._state["error"] = sanitize_progress_error(error)
        self._state["internalError"] = sanitize_progress_error(internal_error or error, max_len=400)
        if partial:
            self._state["partialFailure"] = {
                "lastCompletedStepId": step_id,
                "at": _utc_now_iso(),
            }
        self._flush()

    def block_step(self, step_id: str, *, error: str, internal_error: Optional[str] = None) -> None:
        self.begin_step(step_id)
        self.complete_step(step_id, status=STATUS_BLOCKED, internal_note=internal_error or error)
        self._state["publishStatus"] = STATUS_FAILED
        self._state["operationStatus"] = STATUS_FAILED
        self._state["error"] = sanitize_progress_error(error)
        self._state["internalError"] = sanitize_progress_error(internal_error or error, max_len=400)
        self._flush()

    def finish_success(self) -> None:
        self._state["publishStatus"] = STATUS_COMPLETED
        self._state["operationStatus"] = STATUS_COMPLETED
        self._state["currentStep"] = None
        self._state["error"] = None
        self._state["internalError"] = None
        self._flush()


def load_public_publish_execute_progress(
    operation_id: str,
    *,
    sessions_root: Optional[Path] = None,
    session_id_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    op = validate_operation_id(operation_id)
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    hint = _validate_session_id(session_id_hint or "")
    paths: List[Path] = []
    if hint:
        paths.append(_progress_path(root / hint, op))
    if not paths or not paths[0].is_file():
        for path in root.glob(f"*/{PROGRESS_FILENAME_PREFIX}{op}.json"):
            paths.append(path)
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return public_publish_execute_progress_payload(data)
    return None


def public_publish_execute_progress_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    err = sanitize_progress_error(raw.get("error"))
    current = raw.get("currentStep")
    if current is not None and not isinstance(current, dict):
        current = None
    return {
        "ok": True,
        "operationId": str(raw.get("operationId") or ""),
        "sessionId": str(raw.get("sessionId") or ""),
        "operationStatus": str(raw.get("operationStatus") or ""),
        "publishStatus": str(raw.get("publishStatus") or ""),
        "phase": str(raw.get("phase") or ""),
        "realExecuteBlockedByDefault": bool(raw.get("realExecuteBlockedByDefault", True)),
        "currentStep": current,
        "completedSteps": list(raw.get("completedSteps") or []),
        "stepCatalog": list(raw.get("stepCatalog") or []),
        "partialFailure": raw.get("partialFailure"),
        "startedAt": raw.get("startedAt"),
        "updatedAt": raw.get("updatedAt"),
        "error": err,
        "executionPlanSummary": _plan_summary(raw.get("executionPlan")),
    }


def _plan_summary(plan: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None
    verified = plan.get("verifiedSources") if isinstance(plan.get("verifiedSources"), dict) else {}
    return {
        "candidateBackendSha": verified.get("candidateBackendSha"),
        "simulationOnly": plan.get("simulationOnly"),
    }
