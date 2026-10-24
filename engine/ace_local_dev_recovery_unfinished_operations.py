"""Read-only reconciliation for unfinished CANCEL / PUBLISH / NEW (no retries)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_publish_external_effects_journal import (
    JOURNAL_FILENAME_PREFIX,
    load_publish_external_effects_journal,
)
from engine.ace_local_dev_startup_recovery_check import _observation_is_critical

_BLOCK_DEV_KINDS = frozenset({"CANCEL", "PUBLISH_EXECUTE", "PUBLISH_PREPARE", "NEW"})


@dataclass
class IsolatedOperation:
    kind: str
    session_id: str
    operation_id: str
    status: str
    reconciliation: str
    notes: List[str] = field(default_factory=list)
    external_effects: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "sessionId": self.session_id,
            "operationId": self.operation_id,
            "status": self.status,
            "reconciliation": self.reconciliation,
            "notes": list(self.notes),
            "externalEffects": self.external_effects,
        }


@dataclass
class UnfinishedOperationsAssessment:
    isolated: List[IsolatedOperation] = field(default_factory=list)
    blocked_dev_kinds: List[str] = field(default_factory=list)
    publish_unknown_external: bool = False
    cancel_partial_or_unknown: bool = False
    new_incomplete: bool = False
    evidence_snapshot: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "isolatedOperations": [row.to_dict() for row in self.isolated],
            "blockedDevOperationKinds": list(self.blocked_dev_kinds),
            "publishUnknownExternal": self.publish_unknown_external,
            "cancelPartialOrUnknown": self.cancel_partial_or_unknown,
            "newIncomplete": self.new_incomplete,
            "evidenceSnapshot": list(self.evidence_snapshot),
        }


def _publish_journal_summary(sessions_root: Path, session_id: str, operation_id: str) -> Optional[Dict[str, Any]]:
    path = sessions_root / session_id / f"{JOURNAL_FILENAME_PREFIX}{operation_id}.json"
    if not path.is_file():
        return None
    try:
        journal = load_publish_external_effects_journal(
            operation_id,
            session_id=session_id,
            sessions_root=sessions_root,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return {"loadError": True, "path": str(path)}
    if journal is None:
        return {"missing": True, "path": str(path)}
    stages = journal.get("stages") if isinstance(journal, dict) else None
    summary: Dict[str, Any] = {"path": str(path), "stages": {}}
    if isinstance(stages, dict):
        for name, row in stages.items():
            if not isinstance(row, dict):
                continue
            summary["stages"][name] = str(row.get("status") or "").upper()
    unknown = any(v == "UNKNOWN" for v in summary["stages"].values())
    running = any(v == "RUNNING" for v in summary["stages"].values())
    summary["hasUnknown"] = unknown
    summary["hasRunning"] = running
    return summary


def _reconcile_cancel(obs: Dict[str, Any]) -> IsolatedOperation:
    status = str(obs.get("status") or "").upper()
    notes = list(obs.get("notes") or [])
    critical = bool(obs.get("cancelCritical"))
    if status == "ACTIVE":
        route = "isolate_no_blind_retry"
        notes.append("cancel_active_or_lock_present")
    elif critical or status in ("UNKNOWN", "PARTIAL"):
        route = "isolate_preserve_state"
    else:
        route = "read_only_review"
    return IsolatedOperation(
        kind="CANCEL",
        session_id=str(obs.get("sessionId") or ""),
        operation_id=str(obs.get("operationId") or ""),
        status=status,
        reconciliation=route,
        notes=notes,
    )


def _reconcile_publish(obs: Dict[str, Any], sessions_root: Optional[Path]) -> IsolatedOperation:
    kind = str(obs.get("kind") or "PUBLISH_EXECUTE")
    status = str(obs.get("status") or "").upper()
    notes = list(obs.get("notes") or [])
    sid = str(obs.get("sessionId") or "")
    op = str(obs.get("operationId") or "")
    external: Optional[Dict[str, Any]] = None
    route = "read_only_reconciliation"

    if obs.get("publishExternalUnknown"):
        route = "isolate_no_push_or_deploy"
        notes.append("external_effect_unknown")
    elif obs.get("publishExternalRunning"):
        route = "isolate_no_push_or_deploy"
        notes.append("external_effect_running")
    elif status in ("UNKNOWN", "PARTIAL", "ACTIVE"):
        route = "isolate_no_push_or_deploy"

    if sessions_root and sid and op:
        external = _publish_journal_summary(sessions_root, sid, op)
        if external and external.get("hasUnknown"):
            route = "isolate_no_push_or_deploy"
            notes.append("journal_stage_unknown")
        elif external and external.get("hasRunning"):
            route = "isolate_no_push_or_deploy"
            notes.append("journal_stage_running")

    return IsolatedOperation(
        kind=kind,
        session_id=sid,
        operation_id=op,
        status=status,
        reconciliation=route,
        notes=notes,
        external_effects=external,
    )


def _reconcile_new(obs: Dict[str, Any]) -> IsolatedOperation:
    status = str(obs.get("status") or "").upper()
    notes = list(obs.get("notes") or [])
    route = "read_only_review"
    if status in ("ACTIVE", "UNKNOWN", "PARTIAL"):
        route = "isolate_no_open_active_reinit"
        notes.append("new_cycle_incomplete")
    return IsolatedOperation(
        kind="NEW",
        session_id=str(obs.get("sessionId") or ""),
        operation_id=str(obs.get("operationId") or ""),
        status=status,
        reconciliation=route,
        notes=notes,
    )


def assess_unfinished_dev_operations(
    observations: List[Dict[str, Any]],
    *,
    sessions_root: Optional[Path] = None,
) -> UnfinishedOperationsAssessment:
    assessment = UnfinishedOperationsAssessment()
    assessment.evidence_snapshot = [dict(o) for o in observations if isinstance(o, dict)]

    seen: set[tuple[str, str, str]] = set()
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        kind = str(obs.get("kind") or "")
        if kind not in _BLOCK_DEV_KINDS and not _observation_is_critical(obs):
            continue
        if kind == "EXECUTE_GUARD":
            continue
        key = (kind, str(obs.get("sessionId") or ""), str(obs.get("operationId") or ""))
        if key in seen:
            continue
        seen.add(key)

        if kind == "CANCEL":
            row = _reconcile_cancel(obs)
            assessment.isolated.append(row)
            assessment.cancel_partial_or_unknown = True
        elif kind in ("PUBLISH_EXECUTE", "PUBLISH_PREPARE"):
            row = _reconcile_publish(obs, sessions_root)
            assessment.isolated.append(row)
            if row.reconciliation == "isolate_no_push_or_deploy":
                assessment.publish_unknown_external = True
        elif kind == "NEW":
            row = _reconcile_new(obs)
            assessment.isolated.append(row)
            assessment.new_incomplete = True

    kinds = sorted({row.kind for row in assessment.isolated})
    if assessment.publish_unknown_external:
        kinds = sorted(set(kinds) | {"PUBLISH_EXECUTE", "PUBLISH_PREPARE"})
    assessment.blocked_dev_kinds = kinds
    return assessment
