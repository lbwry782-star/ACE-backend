"""
LOCAL shutdown gate — isolated state machine (not wired to routes, X, or Stop-AceLocal).

Evaluates whether stack shutdown is safe while NEW/CANCEL/PUBLISH may be active.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_cancel_progress import (
    PROGRESS_FILENAME_PREFIX as CANCEL_PROGRESS_PREFIX,
    STEP_BACKEND_RESTORE,
    STEP_CLOSE_CANCEL_CYCLE,
    STEP_ENV_RESTORE,
    STEP_FRONTEND_RESTORE,
    _read_progress_file,
    _validate_session_id,
    public_progress_payload,
)
from engine.ace_local_dev_cycle_store import resolve_sessions_root
from engine.ace_local_dev_execute_guard import is_local_dev_execute_in_progress
from engine.ace_local_dev_publish_external_effects_journal import (
    STAGE_FAILED,
    STAGE_GIT_PUSH_BACKEND,
    STAGE_GITHUB_PAGES,
    STAGE_PENDING,
    STAGE_PRODUCTION_VERIFY,
    STAGE_REGISTRY_UPDATE,
    STAGE_RENDER_WEB,
    STAGE_RENDER_WORKER,
    STAGE_RUNNING,
    STAGE_SUCCEEDED,
    STAGE_UNKNOWN,
    load_publish_external_effects_journal,
    map_publish_execute_step_to_journal_stage,
)
from engine.ace_local_dev_publish_prepare_progress import (
    PROGRESS_FILENAME_PREFIX as PREPARE_PROGRESS_PREFIX,
)
from engine.ace_local_dev_publish_execute_progress import (
    PROGRESS_FILENAME_PREFIX as EXECUTE_PROGRESS_PREFIX,
)
from engine.ace_local_dev_shutdown_builder_activity import (
    merge_builder_observations_into_shutdown_scan,
    observe_builder_shutdown_activity,
)

logger = logging.getLogger(__name__)

GATE_SCHEMA_VERSION = 1
GATE_STATE_FILENAME = "shutdown-gate-state.json"

GATE_RUNNING = "RUNNING"
GATE_SHUTDOWN_REQUESTED = "SHUTDOWN_REQUESTED"
GATE_WAITING_FOR_OPERATION = "WAITING_FOR_OPERATION"
GATE_SAFE_TO_STOP = "SAFE_TO_STOP"
GATE_SHUTDOWN_BLOCKED = "SHUTDOWN_BLOCKED"

CANCEL_LOCK_FILENAME = ".ori-cancel-dev.lock"
CANCEL_JOURNAL_EXECUTE_PREFIX = "cancel-journal-execute-"

CANCEL_CRITICAL_STEPS = frozenset(
    {
        STEP_BACKEND_RESTORE,
        STEP_FRONTEND_RESTORE,
        STEP_ENV_RESTORE,
        STEP_CLOSE_CANCEL_CYCLE,
    }
)

PUBLISH_EXTERNAL_CRITICAL_STAGES = frozenset(
    {
        STAGE_GIT_PUSH_BACKEND,
        STAGE_RENDER_WEB,
        STAGE_RENDER_WORKER,
        STAGE_GITHUB_PAGES,
        STAGE_REGISTRY_UPDATE,
    }
)

TERMINAL_PROGRESS = frozenset({"COMPLETED", "FAILED", "CANCELLED", "SIMULATED", "BLOCKED"})
ACTIVE_PROGRESS = frozenset({"IN_PROGRESS", "PARTIAL"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_local_dev_state_dir(ace_root: Path) -> Path:
    return ace_root / "_backups" / "local-dev"


def shutdown_gate_state_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / GATE_STATE_FILENAME


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def read_shutdown_gate_state(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = shutdown_gate_state_path(ace_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_managed_launcher_session(ace_root: Path) -> Optional[str]:
    reg_path = resolve_local_dev_state_dir(ace_root) / "managed-processes.json"
    if not reg_path.is_file():
        return None
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    session = str(data.get("launcherSession") or "").strip()
    return session or None


def local_dev_operation_start_allowed(
    ace_root: Path,
    *,
    expected_launcher_session: Optional[str] = None,
) -> Tuple[bool, str]:
    """Route hook — returns (allowed, reason_code). Wired via ace_local_dev_shutdown_route_gate."""
    state = read_shutdown_gate_state(ace_root)
    if not state:
        return True, "gate_not_requested"
    gate_status = str(state.get("gateStatus") or GATE_RUNNING).upper()
    if gate_status == GATE_RUNNING:
        return True, "gate_running"
    if expected_launcher_session:
        bound = str(state.get("launcherSession") or "")
        if bound and bound != expected_launcher_session:
            return False, "ace_shutdown_gate_session_mismatch"
    if gate_status in (GATE_SHUTDOWN_REQUESTED, GATE_WAITING_FOR_OPERATION, GATE_SHUTDOWN_BLOCKED):
        return False, "ace_shutdown_gate_operations_frozen"
    if gate_status == GATE_SAFE_TO_STOP:
        return False, "ace_shutdown_gate_shutting_down"
    return False, "ace_shutdown_gate_operations_frozen"


def request_local_shutdown(
    ace_root: Path,
    *,
    launcher_session: str,
    reason: str = "user_request",
) -> Dict[str, Any]:
    session = (launcher_session or "").strip()
    if not session:
        raise ValueError("ace_shutdown_gate_launcher_session_required")
    path = shutdown_gate_state_path(ace_root)
    existing = read_shutdown_gate_state(ace_root)
    if existing:
        prev_status = str(existing.get("gateStatus") or "").upper()
        prev_session = str(existing.get("launcherSession") or "")
        if prev_status in (GATE_SHUTDOWN_REQUESTED, GATE_WAITING_FOR_OPERATION, GATE_SAFE_TO_STOP):
            if prev_session and prev_session != session:
                payload = dict(existing)
                payload["gateStatus"] = GATE_SHUTDOWN_BLOCKED
                payload["updatedAt"] = _utc_now_iso()
                payload["blockReason"] = "concurrent_shutdown_request"
                _write_json_atomic(path, payload)
                return payload
            return existing
    payload: Dict[str, Any] = {
        "schemaVersion": GATE_SCHEMA_VERSION,
        "gateStatus": GATE_SHUTDOWN_REQUESTED,
        "launcherSession": session,
        "requestedAt": _utc_now_iso(),
        "updatedAt": _utc_now_iso(),
        "reason": (reason or "")[:240],
        "evaluation": None,
    }
    _write_json_atomic(path, payload)
    return payload


@dataclass
class OperationObservation:
    kind: str
    session_id: Optional[str] = None
    operation_id: Optional[str] = None
    status: str = "UNKNOWN"
    current_step: Optional[str] = None
    cancel_critical: bool = False
    publish_external_unknown: bool = False
    publish_external_running: bool = False
    notes: List[str] = field(default_factory=list)


def _parse_progress_file(path: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    data = _read_progress_file(path)
    if data is None:
        return None, True
    return data, False


def _cancel_progress_active(raw: Dict[str, Any]) -> Tuple[bool, Optional[str], bool]:
    payload = public_progress_payload(raw)
    op_status = str(payload.get("operationStatus") or "").upper()
    cancel_status = str(payload.get("cancelStatus") or "").upper()
    if op_status in TERMINAL_PROGRESS and cancel_status in TERMINAL_PROGRESS:
        if cancel_status == "PARTIAL" or op_status == "PARTIAL":
            return False, None, True
        return False, None, False
    if op_status not in ACTIVE_PROGRESS and cancel_status not in ACTIVE_PROGRESS:
        if op_status or cancel_status:
            return False, None, False
    current = payload.get("currentStep")
    step_id = None
    if isinstance(current, dict):
        step_id = str(current.get("id") or "") or None
    critical = bool(step_id and step_id in CANCEL_CRITICAL_STEPS)
    post = payload.get("postCancelAutoStart") or {}
    if isinstance(post, dict) and str(post.get("phase") or "").upper() == "IN_PROGRESS":
        critical = True
    return True, step_id, critical


def _publish_progress_active(
    raw: Dict[str, Any],
    *,
    execute: bool,
    session_id: str,
    operation_id: str,
    sessions_root: Path,
    stage_timeout_seconds: Optional[float],
) -> OperationObservation:
    op_status = str(raw.get("operationStatus") or "").upper()
    publish_status = str(raw.get("publishStatus") or raw.get("prepareStatus") or "").upper()
    obs = OperationObservation(
        kind="PUBLISH_EXECUTE" if execute else "PUBLISH_PREPARE",
        session_id=session_id,
        operation_id=operation_id,
        status=op_status or publish_status or "UNKNOWN",
    )
    if op_status in TERMINAL_PROGRESS and publish_status in TERMINAL_PROGRESS:
        if publish_status == "PARTIAL" or op_status == "PARTIAL":
            obs.notes.append("publish_partial")
        return obs

    if op_status in ACTIVE_PROGRESS or publish_status in ACTIVE_PROGRESS:
        obs.status = "ACTIVE"
    current = raw.get("currentStep")
    step_id = str(current.get("id") or "") if isinstance(current, dict) else ""
    obs.current_step = step_id or None

    if not execute:
        return obs

    journal = load_publish_external_effects_journal(
        operation_id,
        session_id=session_id,
        sessions_root=sessions_root,
    )
    stage_states = _resolve_publish_stage_states(
        raw,
        journal,
        stage_timeout_seconds=stage_timeout_seconds,
    )
    for stage_id, st in stage_states.items():
        if st == STAGE_UNKNOWN and stage_id in PUBLISH_EXTERNAL_CRITICAL_STAGES:
            obs.publish_external_unknown = True
            obs.notes.append(f"publish_unknown:{stage_id}")
        if st == STAGE_RUNNING and stage_id in PUBLISH_EXTERNAL_CRITICAL_STAGES:
            obs.publish_external_running = True
    return obs


def _resolve_publish_stage_states(
    progress_raw: Dict[str, Any],
    journal: Optional[Dict[str, Any]],
    *,
    stage_timeout_seconds: Optional[float],
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    journal_stages = {}
    if isinstance(journal, dict):
        js = journal.get("stages")
        if isinstance(js, dict):
            journal_stages = js

    completed = progress_raw.get("completedSteps")
    completed_ids: Dict[str, str] = {}
    if isinstance(completed, list):
        for row in completed:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or "")
            stage = map_publish_execute_step_to_journal_stage(sid)
            if not stage:
                continue
            st = str(row.get("status") or "").upper()
            if st in ("COMPLETED", "SUCCESS", "SIMULATED", "SKIPPED"):
                completed_ids[stage] = STAGE_SUCCEEDED
            elif st == "FAILED":
                completed_ids[stage] = STAGE_FAILED

    current = progress_raw.get("currentStep")
    current_stage = None
    if isinstance(current, dict):
        current_stage = map_publish_execute_step_to_journal_stage(str(current.get("id") or ""))

    for stage_id in (
        STAGE_GIT_PUSH_BACKEND,
        STAGE_RENDER_WEB,
        STAGE_RENDER_WORKER,
        STAGE_GITHUB_PAGES,
        STAGE_PRODUCTION_VERIFY,
        STAGE_REGISTRY_UPDATE,
    ):
        if stage_id in completed_ids:
            result[stage_id] = completed_ids[stage_id]
            continue
        j_entry = journal_stages.get(stage_id) if isinstance(journal_stages.get(stage_id), dict) else None
        if j_entry:
            jst = str(j_entry.get("status") or "").upper()
            if jst == STAGE_SUCCEEDED:
                ev = j_entry.get("evidence")
                if isinstance(ev, dict) and ev:
                    result[stage_id] = STAGE_SUCCEEDED
                    continue
                result[stage_id] = STAGE_UNKNOWN
                continue
            if jst == STAGE_FAILED:
                result[stage_id] = STAGE_FAILED
                continue
            if jst == STAGE_RUNNING:
                if _journal_stage_timed_out(j_entry, stage_timeout_seconds):
                    result[stage_id] = STAGE_UNKNOWN
                else:
                    result[stage_id] = STAGE_RUNNING
                continue
            if jst == STAGE_UNKNOWN:
                result[stage_id] = STAGE_UNKNOWN
                continue
        if current_stage == stage_id:
            result[stage_id] = STAGE_RUNNING
            continue
        result[stage_id] = STAGE_PENDING
    return result


def _journal_stage_timed_out(entry: Dict[str, Any], timeout_seconds: Optional[float]) -> bool:
    if timeout_seconds is None or timeout_seconds <= 0:
        return False
    started = str(entry.get("startedAt") or "")
    if not started:
        return False
    try:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - start_dt).total_seconds() > timeout_seconds
    except ValueError:
        return True


def _scan_new_cycle_in_progress(sessions_root: Path) -> List[OperationObservation]:
    observations: List[OperationObservation] = []
    if not sessions_root.is_dir():
        return observations
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        sid = session_dir.name
        if not _validate_session_id(sid):
            continue
        session_file = session_dir / "session.json"
        if not session_file.is_file():
            continue
        try:
            meta = json.loads(session_file.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            observations.append(
                OperationObservation(kind="NEW", session_id=sid, status="UNKNOWN", notes=["session_json_invalid"])
            )
            continue
        if not isinstance(meta, dict):
            continue
        status = str(meta.get("status") or "").upper()
        if status == "IN_PROGRESS":
            observations.append(
                OperationObservation(kind="NEW", session_id=sid, status="ACTIVE", notes=["baseline_in_progress"])
            )
    return observations


def _scan_cancel_lock(sessions_root: Path) -> Optional[OperationObservation]:
    lock_path = sessions_root / CANCEL_LOCK_FILENAME
    if not lock_path.is_file():
        return None
    try:
        raw = lock_path.read_text(encoding="utf-8-sig")
        json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return OperationObservation(kind="CANCEL", status="UNKNOWN", notes=["cancel_lock_unreadable"])
    return OperationObservation(kind="CANCEL", status="ACTIVE", notes=["cancel_lock_held"])


def _scan_cancel_journals(session_dir: Path, session_id: str) -> List[OperationObservation]:
    observations: List[OperationObservation] = []
    for path in session_dir.glob(f"{CANCEL_JOURNAL_EXECUTE_PREFIX}*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            observations.append(
                OperationObservation(
                    kind="CANCEL",
                    session_id=session_id,
                    status="UNKNOWN",
                    notes=["cancel_journal_corrupt"],
                )
            )
            continue
        if not isinstance(data, dict):
            continue
        status = str(data.get("status") or "").upper()
        if status == "IN_PROGRESS":
            observations.append(
                OperationObservation(
                    kind="CANCEL",
                    session_id=session_id,
                    status="ACTIVE",
                    notes=["cancel_journal_in_progress"],
                )
            )
        elif status == "PARTIAL":
            observations.append(
                OperationObservation(
                    kind="CANCEL",
                    session_id=session_id,
                    status="PARTIAL",
                    notes=["cancel_journal_partial"],
                )
            )
    return observations


def observe_active_local_dev_operations(
    *,
    ace_root: Path,
    sessions_root: Optional[Path] = None,
    execute_busy: Optional[bool] = None,
    backend_reachable: bool = True,
    stage_timeout_seconds: Optional[float] = None,
    redis_reachable: Optional[bool] = None,
    worker_reachable: Optional[bool] = None,
    builder_activity_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = sessions_root if sessions_root is not None else resolve_sessions_root()
    observations: List[OperationObservation] = []

    if execute_busy is None:
        execute_busy = is_local_dev_execute_in_progress()
    if execute_busy:
        observations.append(
            OperationObservation(kind="EXECUTE_GUARD", status="ACTIVE", notes=["in_process_execute_guard"])
        )

    if not backend_reachable:
        observations.append(
            OperationObservation(kind="BACKEND", status="UNKNOWN", notes=["backend_unreachable"])
        )

    lock_obs = _scan_cancel_lock(root)
    if lock_obs:
        observations.append(lock_obs)

    observations.extend(_scan_new_cycle_in_progress(root))

    if root.is_dir():
        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue
            sid = session_dir.name
            if not _validate_session_id(sid):
                continue
            observations.extend(_scan_cancel_journals(session_dir, sid))

            for path in session_dir.glob(f"{CANCEL_PROGRESS_PREFIX}*.json"):
                raw, corrupt = _parse_progress_file(path)
                op_id = path.stem.replace(CANCEL_PROGRESS_PREFIX, "")
                if corrupt or raw is None:
                    observations.append(
                        OperationObservation(
                            kind="CANCEL",
                            session_id=sid,
                            operation_id=op_id,
                            status="UNKNOWN",
                            notes=["cancel_progress_corrupt"],
                        )
                    )
                    continue
                active, step, critical = _cancel_progress_active(raw)
                if active:
                    observations.append(
                        OperationObservation(
                            kind="CANCEL",
                            session_id=sid,
                            operation_id=op_id,
                            status="ACTIVE",
                            current_step=step,
                            cancel_critical=critical,
                        )
                    )
                elif critical:
                    observations.append(
                        OperationObservation(
                            kind="CANCEL",
                            session_id=sid,
                            operation_id=op_id,
                            status="UNKNOWN",
                            notes=["cancel_progress_inconsistent"],
                        )
                    )

            for path in session_dir.glob(f"{PREPARE_PROGRESS_PREFIX}*.json"):
                raw, corrupt = _parse_progress_file(path)
                op_id = path.stem.replace(PREPARE_PROGRESS_PREFIX, "")
                if corrupt or raw is None:
                    observations.append(
                        OperationObservation(
                            kind="PUBLISH_PREPARE",
                            session_id=sid,
                            operation_id=op_id,
                            status="UNKNOWN",
                            notes=["prepare_progress_corrupt"],
                        )
                    )
                    continue
                obs = _publish_progress_active(
                    raw,
                    execute=False,
                    session_id=sid,
                    operation_id=op_id,
                    sessions_root=root,
                    stage_timeout_seconds=stage_timeout_seconds,
                )
                if obs.status == "ACTIVE":
                    observations.append(obs)

            for path in session_dir.glob(f"{EXECUTE_PROGRESS_PREFIX}*.json"):
                raw, corrupt = _parse_progress_file(path)
                op_id = path.stem.replace(EXECUTE_PROGRESS_PREFIX, "")
                if corrupt or raw is None:
                    observations.append(
                        OperationObservation(
                            kind="PUBLISH_EXECUTE",
                            session_id=sid,
                            operation_id=op_id,
                            status="UNKNOWN",
                            notes=["execute_progress_corrupt"],
                        )
                    )
                    continue
                obs = _publish_progress_active(
                    raw,
                    execute=True,
                    session_id=sid,
                    operation_id=op_id,
                    sessions_root=root,
                    stage_timeout_seconds=stage_timeout_seconds,
                )
                if obs.status == "ACTIVE" or obs.publish_external_unknown or obs.publish_external_running:
                    observations.append(obs)

    scan: Dict[str, Any] = {
        "observations": [_observation_to_dict(o) for o in observations],
        "executeBusy": bool(execute_busy),
        "backendReachable": bool(backend_reachable),
    }
    builder_scan = observe_builder_shutdown_activity(
        ace_root=ace_root,
        redis_reachable=redis_reachable,
        worker_reachable=worker_reachable,
        builder_activity_override=builder_activity_override,
    )
    merge_builder_observations_into_shutdown_scan(scan, builder_scan)
    return scan


def _observation_to_dict(obs: OperationObservation) -> Dict[str, Any]:
    return {
        "kind": obs.kind,
        "sessionId": obs.session_id,
        "operationId": obs.operation_id,
        "status": obs.status,
        "currentStep": obs.current_step,
        "cancelCritical": obs.cancel_critical,
        "publishExternalUnknown": obs.publish_external_unknown,
        "publishExternalRunning": obs.publish_external_running,
        "notes": list(obs.notes),
    }


_BUILDER_BLOCKING_KINDS = frozenset(
    {"BUILDER1", "BUILDER2", "BUILDER2_QUEUE", "BUILDER2_WORKER", "REDIS"}
)


def evaluate_shutdown_gate_status(
    observations: List[Dict[str, Any]],
    *,
    shutdown_requested: bool,
    scan_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if not shutdown_requested:
        return GATE_RUNNING, reasons

    has_unknown = False
    has_active = False
    has_cancel_critical = False
    has_publish_external = False

    if isinstance(scan_meta, dict):
        if scan_meta.get("redisReachable") is False:
            has_unknown = True
            reasons.append("redis_unreachable")
        worker_ok = scan_meta.get("workerReachable")
        if worker_ok is False:
            has_unknown = True
            reasons.append("worker_unreachable")

    for row in observations:
        kind = str(row.get("kind") or "")
        status = str(row.get("status") or "").upper()
        notes = row.get("notes") or []

        if status == "UNKNOWN" or "corrupt" in " ".join(str(n) for n in notes):
            has_unknown = True
            reasons.append(f"unknown:{kind}")
            continue
        if status in ("ACTIVE", "PARTIAL") or row.get("publishExternalRunning") or row.get("cancelCritical"):
            has_active = True
        if row.get("cancelCritical"):
            has_cancel_critical = True
            reasons.append("cancel_critical_step")
        if row.get("publishExternalUnknown"):
            has_unknown = True
            reasons.append("publish_external_unknown")
        if row.get("publishExternalRunning"):
            has_publish_external = True
            reasons.append("publish_external_running")
        if kind == "BACKEND" and status == "UNKNOWN":
            has_unknown = True
            reasons.append("backend_unknown")
        if kind in _BUILDER_BLOCKING_KINDS:
            if status == "UNKNOWN":
                has_unknown = True
                reasons.append(f"builder_unknown:{kind}")
            elif status == "ACTIVE":
                has_active = True
                reasons.append(f"builder_active:{kind}")

    if has_unknown:
        return GATE_SHUTDOWN_BLOCKED, reasons
    if has_active or has_cancel_critical or has_publish_external:
        return GATE_WAITING_FOR_OPERATION, reasons
    return GATE_SAFE_TO_STOP, reasons


def refresh_local_shutdown_gate(
    ace_root: Path,
    *,
    launcher_session: Optional[str] = None,
    execute_busy: Optional[bool] = None,
    backend_reachable: bool = True,
    sessions_root: Optional[Path] = None,
    stage_timeout_seconds: Optional[float] = None,
    redis_reachable: Optional[bool] = None,
    worker_reachable: Optional[bool] = None,
    builder_activity_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = shutdown_gate_state_path(ace_root)
    state = read_shutdown_gate_state(ace_root)
    if not state:
        state = {
            "schemaVersion": GATE_SCHEMA_VERSION,
            "gateStatus": GATE_RUNNING,
            "launcherSession": launcher_session or read_managed_launcher_session(ace_root) or "",
            "requestedAt": None,
            "updatedAt": _utc_now_iso(),
            "reason": "",
            "evaluation": None,
        }
        _write_json_atomic(path, state)
        return state

    requested = str(state.get("gateStatus") or GATE_RUNNING).upper() != GATE_RUNNING
    if launcher_session:
        state["launcherSession"] = launcher_session

    scan = observe_active_local_dev_operations(
        ace_root=ace_root,
        sessions_root=sessions_root,
        execute_busy=execute_busy,
        backend_reachable=backend_reachable,
        stage_timeout_seconds=stage_timeout_seconds,
        redis_reachable=redis_reachable,
        worker_reachable=worker_reachable,
        builder_activity_override=builder_activity_override,
    )
    gate_status, reasons = evaluate_shutdown_gate_status(
        list(scan.get("observations") or []),
        shutdown_requested=requested
        or str(state.get("gateStatus") or "").upper()
        in (GATE_SHUTDOWN_REQUESTED, GATE_WAITING_FOR_OPERATION, GATE_SAFE_TO_STOP, GATE_SHUTDOWN_BLOCKED),
        scan_meta=scan,
    )
    if str(state.get("gateStatus") or "").upper() == GATE_SHUTDOWN_REQUESTED or requested:
        requested = True
    if not requested:
        gate_status = GATE_RUNNING

    state["gateStatus"] = gate_status
    state["updatedAt"] = _utc_now_iso()
    state["evaluation"] = {
        "reasons": reasons,
        "observationCount": len(scan.get("observations") or []),
        "scan": scan,
    }
    _write_json_atomic(path, state)
    return state
