"""Durable atomic journal for external PUBLISH stages (shutdown gate / recovery; not wired to execute)."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_cancel_progress import validate_operation_id
from engine.ace_local_dev_cycle_store import _validate_session_id, resolve_sessions_root

JOURNAL_SCHEMA = 1
JOURNAL_FILENAME_PREFIX = "publish-external-effects-journal-"

STAGE_PRECHECKS = "prechecks"
STAGE_GIT_PUSH_BACKEND = "git_push_backend"
STAGE_RENDER_WEB = "render_web"
STAGE_RENDER_WORKER = "render_worker"
STAGE_GITHUB_PAGES = "github_pages"
STAGE_PRODUCTION_VERIFY = "production_verify"
STAGE_REGISTRY_UPDATE = "registry_update"

STAGE_PENDING = "PENDING"
STAGE_RUNNING = "RUNNING"
STAGE_SUCCEEDED = "SUCCEEDED"
STAGE_FAILED = "FAILED"
STAGE_UNKNOWN = "UNKNOWN"

_SECRET_KEY = re.compile(r"(token|secret|password|api[_-]?key|authorization)", re.I)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _journal_path(session_root: Path, operation_id: str) -> Path:
    op = validate_operation_id(operation_id)
    return session_root / f"{JOURNAL_FILENAME_PREFIX}{op}.json"


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _sanitize_evidence(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    safe: Dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key)
        if _SECRET_KEY.search(key_text):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key_text] = value
        elif isinstance(value, dict):
            nested = _sanitize_evidence(value)
            if nested:
                safe[key_text] = nested
    return safe


def load_publish_external_effects_journal(
    operation_id: str,
    *,
    session_id: str,
    sessions_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    sid = _validate_session_id(session_id)
    if not sid:
        return None
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    path = _journal_path(root / sid, validate_operation_id(operation_id))
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


class PublishExternalEffectsJournalWriter:
    def __init__(
        self,
        *,
        session_id: str,
        operation_id: str,
        candidate_backend_sha: str = "",
        sessions_root: Optional[Path] = None,
    ) -> None:
        sid = _validate_session_id(session_id)
        if not sid:
            raise ValueError("ace_publish_external_journal_session_invalid")
        self.operation_id = validate_operation_id(operation_id)
        self.session_id = sid
        root = sessions_root if sessions_root is not None else resolve_sessions_root()
        self._path = _journal_path(root / sid, self.operation_id)
        now = _utc_now_iso()
        self._state: Dict[str, Any] = {
            "schemaVersion": JOURNAL_SCHEMA,
            "operationId": self.operation_id,
            "sessionId": self.session_id,
            "candidateBackendSha": (candidate_backend_sha or "").strip().lower(),
            "startedAt": now,
            "updatedAt": now,
            "stages": {},
        }
        self._flush()

    @property
    def path(self) -> Path:
        return self._path

    def _flush(self) -> None:
        self._state["updatedAt"] = _utc_now_iso()
        _write_atomic(self._path, self._state)

    def set_stage(
        self,
        stage_id: str,
        status: str,
        *,
        evidence: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        sid = (stage_id or "").strip()
        if not sid:
            raise ValueError("ace_publish_external_journal_stage_invalid")
        st = (status or "").strip().upper()
        if st not in (
            STAGE_PENDING,
            STAGE_RUNNING,
            STAGE_SUCCEEDED,
            STAGE_FAILED,
            STAGE_UNKNOWN,
        ):
            raise ValueError("ace_publish_external_journal_status_invalid")
        stages: Dict[str, Any] = dict(self._state.get("stages") or {})
        prev = stages.get(sid) if isinstance(stages.get(sid), dict) else {}
        started = str(prev.get("startedAt") or _utc_now_iso())
        if st == STAGE_RUNNING and prev.get("status") != STAGE_RUNNING:
            started = _utc_now_iso()
        entry: Dict[str, Any] = {
            "status": st,
            "startedAt": started,
            "updatedAt": _utc_now_iso(),
            "evidence": _sanitize_evidence(evidence),
        }
        if st in (STAGE_SUCCEEDED, STAGE_FAILED, STAGE_UNKNOWN):
            entry["finishedAt"] = _utc_now_iso()
        if error:
            entry["error"] = str(error)[:240]
        stages[sid] = entry
        self._state["stages"] = stages
        self._flush()


def map_publish_execute_step_to_journal_stage(step_id: str) -> Optional[str]:
    token = (step_id or "").strip()
    mapping = {
        "registered_candidate": STAGE_PRECHECKS,
        "fresh_preview": STAGE_PRECHECKS,
        "fingerprint_match": STAGE_PRECHECKS,
        "product_scope": STAGE_PRECHECKS,
        "worker_coupling": STAGE_PRECHECKS,
        "execution_plan": STAGE_PRECHECKS,
        "isolated_release_prepare": STAGE_PRECHECKS,
        "candidate_sha_match": STAGE_PRECHECKS,
        "real_execute_gate": STAGE_PRECHECKS,
        "git_push_backend": STAGE_GIT_PUSH_BACKEND,
        "render_web_deploy": STAGE_RENDER_WEB,
        "render_worker_deploy": STAGE_RENDER_WORKER,
        "github_pages_deploy": STAGE_GITHUB_PAGES,
        "production_verify": STAGE_PRODUCTION_VERIFY,
        "production_registry_update": STAGE_REGISTRY_UPDATE,
    }
    return mapping.get(token)
