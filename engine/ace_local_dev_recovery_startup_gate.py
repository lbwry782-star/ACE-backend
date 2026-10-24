"""
Recovery startup gate (phase 5.5) — coordinator check + decision report; no restore execution.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_BLOCKED,
    RecoveryStartupCoordinatorDecision,
    coordinator_decision_to_dict,
    evaluate_recovery_startup_coordination,
)
from engine.ace_local_dev_shutdown_gate import _write_json_atomic, resolve_local_dev_state_dir

REPORT_SCHEMA_VERSION = 1
REPORTS_DIR_NAME = "recovery-startup-coordinator-reports"
LATEST_REPORT_FILENAME = "recovery-startup-decision-latest.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def startup_decision_reports_dir(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / REPORTS_DIR_NAME


def planned_future_actions_from_decision(
    decision: RecoveryStartupCoordinatorDecision,
) -> Dict[str, Any]:
    return {
        "autoRestoreEngine": {
            "wouldRun": decision.coordinator_outcome
            in ("AUTO_RECOVERY_ALLOWED", "START_WITH_ISOLATED_OPERATIONS"),
            "executedInPhase55": False,
            "note": "Restore engine not invoked in phase 5.5",
        },
        "periodicSaveService": {
            "wouldActivate": bool(decision.allow_periodic_save_service),
            "executedInPhase55": False,
        },
        "newLauncherSession": {
            "wouldCreate": bool(decision.allow_new_launcher_session),
            "executedInPhase55": False,
        },
        "ledgerInitialization": {
            "wouldInitialize": bool(decision.allow_ledger_initialization),
            "executedInPhase55": False,
        },
        "blockedDevOperationKinds": list(decision.blocked_dev_operation_kinds),
        "isolatedOperations": list(decision.isolated_operations),
        "guiRecoveryMode": decision.gui_recovery_mode,
    }


def build_startup_gate_report(
    decision: RecoveryStartupCoordinatorDecision,
    *,
    check_only: bool = True,
    validate_only: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "phase": "5.5",
        "writtenAt": _utc_now_iso(),
        "reportId": uuid.uuid4().hex,
        "checkOnly": check_only,
        "validateOnly": validate_only,
        "checkFailed": bool(error),
        "error": error,
        "coordinatorDecision": coordinator_decision_to_dict(decision),
        "plannedFutureActions": planned_future_actions_from_decision(decision),
        "evidencePreservationNote": (
            "Separate report file; does not overwrite recovery ledger or termination evidence."
        ),
    }


def write_startup_gate_report(ace_root: Path, report: Dict[str, Any]) -> Dict[str, str]:
    ace_root = Path(ace_root)
    reports_dir = startup_decision_reports_dir(ace_root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = str(report.get("reportId") or uuid.uuid4().hex)
    archive_path = reports_dir / f"recovery-startup-decision-{stamp}-{rid[:8]}.json"
    latest_path = reports_dir / LATEST_REPORT_FILENAME
    _write_json_atomic(archive_path, report)
    _write_json_atomic(latest_path, report)
    return {"archivePath": str(archive_path), "latestPath": str(latest_path)}


def evaluate_startup_recovery_gate(
    ace_root: Path,
    *,
    sessions_root: Optional[Path] = None,
    inspect_live_processes: bool = True,
    acquire_coordinator_lock: bool = True,
    validate_only: bool = False,
) -> Dict[str, Any]:
    try:
        decision = evaluate_recovery_startup_coordination(
            ace_root,
            sessions_root=sessions_root,
            inspect_live_processes=inspect_live_processes,
            acquire_coordinator_lock=acquire_coordinator_lock,
        )
        report = build_startup_gate_report(
            decision,
            validate_only=validate_only,
        )
        paths = write_startup_gate_report(ace_root, report)
        return {
            "ok": True,
            "checkFailed": False,
            "coordinatorOutcome": decision.coordinator_outcome,
            "startupAllowed": decision.startup_allowed,
            "continueLocalStackStart": _gate_allows_stack_continue(decision),
            "assignNewLauncherSession": decision.coordinator_outcome == COORDINATOR_START_ALLOWED,
            "reportPaths": paths,
            "report": report,
        }
    except Exception as exc:
        fallback = RecoveryStartupCoordinatorDecision(
            coordinator_outcome=COORDINATOR_START_BLOCKED,
            recovery_status="GATE_CHECK_FAILED",
            termination_type="UNKNOWN",
            confidence="LOW",
            evidence=[],
            previous_launcher_session=None,
            last_verified_checkpoint=None,
            allow_local_processes=False,
            allow_dev_work=False,
            allow_new_launcher_session=False,
            allow_ledger_initialization=False,
            allow_periodic_save_service=False,
            reasons=[f"gate_check_exception:{type(exc).__name__}"],
            gui_recovery_mode="blocked",
        )
        report = build_startup_gate_report(
            fallback,
            validate_only=validate_only,
            error=str(exc)[:480],
        )
        paths = write_startup_gate_report(ace_root, report)
        return {
            "ok": False,
            "checkFailed": True,
            "coordinatorOutcome": COORDINATOR_START_BLOCKED,
            "startupAllowed": False,
            "continueLocalStackStart": False,
            "assignNewLauncherSession": False,
            "reportPaths": paths,
            "report": report,
            "error": str(exc)[:480],
        }


def _gate_allows_stack_continue(decision: RecoveryStartupCoordinatorDecision) -> bool:
    if decision.coordinator_outcome == COORDINATOR_START_BLOCKED:
        return False
    if not decision.allow_local_processes:
        return False
    return True
