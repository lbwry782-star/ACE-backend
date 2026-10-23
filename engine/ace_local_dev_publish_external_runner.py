"""External PUBLISH runner — simulation only, separate from Flask LOCAL."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_cancel_progress import validate_operation_id
from engine.ace_local_dev_cycle_store import (
    active_cycle_pointer_path,
    resolve_sessions_root,
    scan_open_cycle_session_ids,
)
from engine.ace_local_dev_publish_candidate_registry import (
    evaluate_registered_candidate,
    load_publish_candidate_registry,
    load_production_backend_baseline_sha,
)
from engine.ace_local_dev_publish_execute_progress import (
    load_public_publish_execute_progress,
)
from engine.ace_local_dev_publish_render_sim_execute_adapters import (
    build_simulated_render_publish_execute_adapters,
)
from engine.ace_local_dev_publish_render_sim_transport import (
    PublishRenderSimState,
    build_publish_sim_mock_transport,
)
from engine.ace_local_dev_publish_uri_candidate_execute import (
    PublishExecuteRejected,
    execute_publish_uri_candidate,
)
from engine.ace_local_dev_render_api_transport import RenderApiTransport

PUBLISH_RUNNER_MODE_SIMULATION = "SIMULATION"
_FORBIDDEN_CREDENTIAL_ENV = frozenset(
    {
        "ACE_RENDER_API_KEY",
        "RENDER_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)
_LOCK_FILENAME = "publish-external-runner.lock"
_JOURNAL_PREFIX = "publish-external-runner-journal-"


class PublishExternalRunnerError(Exception):
    def __init__(self, code: str, *, message: Optional[str] = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


@dataclass
class PublishExternalRunnerJournal:
    session_id: str
    operation_id: str
    mode: str = PUBLISH_RUNNER_MODE_SIMULATION
    steps: List[Dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    finished_at: Optional[str] = None
    status: str = "IN_PROGRESS"
    error: Optional[str] = None

    def append(self, name: str, status: str, *, detail: str = "") -> None:
        self.steps.append(
            {
                "at": _utc_now_iso(),
                "name": name,
                "status": status,
                "detail": detail[:500],
            }
        )

    def flush(self, session_root: Path) -> Path:
        self.finished_at = _utc_now_iso()
        path = session_root / f"{_JOURNAL_PREFIX}{self.operation_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "sessionId": self.session_id,
            "operationId": self.operation_id,
            "mode": self.mode,
            "steps": self.steps,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "status": self.status,
            "error": self.error,
        }
        tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_simulation_only_mode(requested_mode: str) -> None:
    token = (requested_mode or "").strip().upper()
    if token and token != PUBLISH_RUNNER_MODE_SIMULATION:
        raise PublishExternalRunnerError(
            "ace_publish_external_runner_real_mode_forbidden",
            message="Only SIMULATION mode is supported.",
        )


def assert_no_forbidden_credentials_in_env() -> None:
    present = [key for key in _FORBIDDEN_CREDENTIAL_ENV if (os.environ.get(key) or "").strip()]
    if present:
        raise PublishExternalRunnerError(
            "ace_publish_external_runner_credentials_forbidden",
            message="Credential environment variables must not be set for simulation runner.",
        )


def validate_active_session(session_id: str, *, sessions_root: Optional[Path] = None) -> None:
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    sid = (session_id or "").strip()
    pointer = active_cycle_pointer_path(root)
    if pointer.is_file():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublishExternalRunnerError("ace_publish_external_runner_active_cycle_invalid") from exc
        active = str(data.get("sessionId") or "").strip()
        cycle_status = str(data.get("cycleStatus") or "").strip().upper()
        if active != sid:
            raise PublishExternalRunnerError(
                "ace_publish_external_runner_session_not_active",
                message="Session id does not match active dev cycle pointer.",
            )
        if cycle_status != "OPEN":
            raise PublishExternalRunnerError(
                "ace_publish_external_runner_cycle_not_open",
                message="Active dev cycle is not OPEN.",
            )
    open_ids = scan_open_cycle_session_ids(root)
    if len(open_ids) > 1:
        raise PublishExternalRunnerError(
            "ace_publish_external_runner_multiple_open_cycles",
            message="Multiple OPEN dev cycles detected.",
        )
    if open_ids and sid not in open_ids:
        raise PublishExternalRunnerError(
            "ace_publish_external_runner_session_not_open",
            message="Session is not the single OPEN dev cycle.",
        )


def validate_active_publish_candidate(session_id: str) -> str:
    record = load_publish_candidate_registry()
    if not record:
        raise PublishExternalRunnerError("ace_publish_candidate_not_registered")
    reg_session = str(record.get("sessionId") or "").strip()
    if reg_session != session_id:
        raise PublishExternalRunnerError(
            "ace_publish_external_runner_candidate_session_mismatch",
            message="Active publish candidate belongs to a different session.",
        )
    baseline = load_production_backend_baseline_sha() or ""
    effective, reasons = evaluate_registered_candidate(
        record=record,
        active_session_id=session_id,
        cycle_status="OPEN",
        current_production_baseline_sha=baseline,
    )
    if effective != "ACTIVE":
        raise PublishExternalRunnerError(
            "ace_publish_candidate_registry_stale",
            message=";".join(reasons) if reasons else "Candidate not ACTIVE.",
        )
    sha = str(record.get("candidateBackendSha") or "").strip()
    if not sha:
        raise PublishExternalRunnerError("ace_publish_uri_candidate_sha_invalid")
    return sha


class _RunnerSessionLock:
    def __init__(self, session_root: Path, operation_id: str) -> None:
        self._path = session_root / _LOCK_FILENAME
        self._operation_id = operation_id
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        if self._path.is_file():
            try:
                existing = self._path.read_text(encoding="utf-8").strip()
            except OSError:
                existing = ""
            raise PublishExternalRunnerError(
                "ace_publish_external_runner_session_busy",
                message=f"Runner lock held (operation={existing or 'unknown'}).",
            )
        try:
            self._fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, self._operation_id.encode("utf-8"))
        except FileExistsError as exc:
            raise PublishExternalRunnerError(
                "ace_publish_external_runner_session_busy",
                message="Parallel runner invocation blocked.",
            ) from exc

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            if self._path.is_file():
                self._path.unlink()
        except OSError:
            pass


def run_external_publish_simulation(
    *,
    session_id: str,
    operation_id: str,
    publish_uri_preview_fingerprint: str,
    confirm_execute: str = "PUBLISH",
    transport: Optional[RenderApiTransport] = None,
    sim_state: Optional[PublishRenderSimState] = None,
    registry_simulation_path: Optional[Path] = None,
    sessions_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run full publish execute chain with simulated Render adapters only.
    Does not read API keys or mutate production registry.
    """
    assert_simulation_only_mode(PUBLISH_RUNNER_MODE_SIMULATION)
    assert_no_forbidden_credentials_in_env()

    try:
        op = validate_operation_id(operation_id)
    except ValueError as exc:
        raise PublishExternalRunnerError("ace_publish_execute_operation_id_invalid") from exc

    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    session_root = root / session_id
    if not session_root.is_dir():
        raise PublishExternalRunnerError("ace_publish_external_runner_session_missing")

    journal = PublishExternalRunnerJournal(
        session_id=session_id,
        operation_id=op,
        started_at=_utc_now_iso(),
    )

    lock = _RunnerSessionLock(session_root, op)
    lock.acquire()
    try:
        journal.append("validate_session", "START")
        validate_active_session(session_id, sessions_root=root)
        journal.append("validate_session", "SUCCESS")

        journal.append("validate_candidate", "START")
        validate_active_publish_candidate(session_id)
        journal.append("validate_candidate", "SUCCESS")

        state = sim_state or PublishRenderSimState()
        mock_transport = transport or build_publish_sim_mock_transport(state)
        sim_registry = registry_simulation_path or (session_root / f"sim-registry-{op}.json")

        adapters = build_simulated_render_publish_execute_adapters(
            mock_transport,
            registry_simulation_path=sim_registry,
            sim_state=state,
            poll_interval_seconds=0.0,
            wait_deadline_seconds=15.0,
        )
        adapters.allow_execute_gate_bypass_for_simulation = True

        orig_git = adapters.git_push_backend

        def _git_push(ctx):
            res = orig_git(ctx)
            if res.ok:
                state.on_git_push(ctx.precheck.candidate_backend_sha)
            return res

        adapters.git_push_backend = _git_push  # type: ignore[method-assign]

        journal.append("execute_simulation", "START")
        try:
            result = execute_publish_uri_candidate(
                confirm_execute=confirm_execute,
                publish_uri_preview_fingerprint=publish_uri_preview_fingerprint,
                operation_id=op,
                adapters=adapters,
            )
        except PublishExecuteRejected as exc:
            journal.append("execute_simulation", "FAILED", detail=exc.code)
            journal.status = _progress_status(session_root, op, root)
            journal.error = exc.code
            journal.flush(session_root)
            raise PublishExternalRunnerError(exc.code, message=exc.message) from exc

        journal.append("execute_simulation", "SUCCESS")
        journal.status = "SUCCESS"
        journal_path = journal.flush(session_root)
        result["externalRunnerMode"] = PUBLISH_RUNNER_MODE_SIMULATION
        result["externalRunnerJournalPath"] = str(journal_path)
        return result
    finally:
        lock.release()


def _progress_status(session_root: Path, operation_id: str, sessions_root: Path) -> str:
    prog = load_public_publish_execute_progress(
        operation_id,
        sessions_root=sessions_root,
        session_id_hint=session_root.name,
    )
    if not prog:
        return "FAILED"
    return str(prog.get("operationStatus") or "FAILED")
