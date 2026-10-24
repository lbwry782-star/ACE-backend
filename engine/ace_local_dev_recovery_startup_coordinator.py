"""
Recovery startup coordinator — automatic recovery policy (phase 5.1; not wired to Start-AceLocal).

No user approval gate before recovery. Does not perform real restore on production ACE paths.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_recovery_auto_restore_plan import (
    AutoRestorePlan,
    build_auto_restore_plan,
)
from engine.ace_local_dev_recovery_checkpoint_validate import (
    assess_recovery_checkpoints,
    recovery_checkpoints_root,
    validate_recovery_checkpoint_root,
)
from engine.ace_local_dev_recovery_unfinished_operations import (
    UnfinishedOperationsAssessment,
    assess_unfinished_dev_operations,
)
from engine.ace_local_dev_shutdown_gate import (
    observe_active_local_dev_operations,
    resolve_local_dev_state_dir,
)
from engine.ace_local_dev_startup_recovery_check import (
    STARTUP_CRITICAL_RECOVERY,
    STARTUP_FIRST_START,
    STARTUP_LEGACY_INSTALL_RECOVERY_BOOTSTRAP,
    STARTUP_NORMAL_START,
    STARTUP_RECOVERY_REQUIRED,
    STARTUP_RECOVERY_REVIEW,
    StartupRecoveryDecision,
    evaluate_previous_local_startup,
    read_recovery_ledger,
)

logger = logging.getLogger(__name__)

COORDINATOR_START_ALLOWED = "START_ALLOWED"
COORDINATOR_AUTO_RECOVERY_ALLOWED = "AUTO_RECOVERY_ALLOWED"
COORDINATOR_START_WITH_ISOLATED_OPERATIONS = "START_WITH_ISOLATED_OPERATIONS"
COORDINATOR_RECOVERY_UNSAFE = "RECOVERY_UNSAFE"
COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP = "LEGACY_RECOVERY_BOOTSTRAP"
COORDINATOR_START_BLOCKED = "START_BLOCKED"

# Legacy phase-5 names (deprecated; kept for import stability in tests/docs)
COORDINATOR_RECOVERY_REQUIRED = COORDINATOR_AUTO_RECOVERY_ALLOWED
COORDINATOR_RECOVERY_REVIEW = COORDINATOR_START_WITH_ISOLATED_OPERATIONS
COORDINATOR_CRITICAL_RECOVERY = COORDINATOR_START_WITH_ISOLATED_OPERATIONS

STARTUP_COORDINATOR_LOCK = "recovery-startup-coordinator.lock"

_LOCAL_START_OUTCOMES = frozenset(
    {
        COORDINATOR_START_ALLOWED,
        COORDINATOR_AUTO_RECOVERY_ALLOWED,
        COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
        COORDINATOR_RECOVERY_UNSAFE,
        COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP,
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def coordinator_lock_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / STARTUP_COORDINATOR_LOCK


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except AttributeError:
        return False
    return True


def try_acquire_startup_coordinator_lock(ace_root: Path) -> Dict[str, Any]:
    ace_root = Path(ace_root)
    path = coordinator_lock_path(ace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8-sig"))
            pid = int(existing.get("pid") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {"acquired": False, "reason": "coordinator_lock_unreadable"}
        if _pid_alive(pid):
            return {"acquired": False, "reason": "coordinator_lock_live", "pid": pid}
        path.unlink(missing_ok=True)
    payload = {
        "pid": os.getpid(),
        "startedAt": _utc_now_iso(),
        "token": uuid.uuid4().hex,
    }
    tmp = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return {"acquired": True, "reason": "ok", "token": payload["token"]}


def release_startup_coordinator_lock(ace_root: Path) -> None:
    coordinator_lock_path(Path(ace_root)).unlink(missing_ok=True)


def _workspace_roots(ace_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    backend = ace_root / "ACE-Backend"
    frontend = ace_root / "ACE-Frontend"
    return (
        backend if backend.is_dir() else None,
        frontend if frontend.is_dir() else None,
    )


def _recommended_checkpoint_root(
    ace_root: Path,
    checkpoint_assessment: Dict[str, Any],
) -> Optional[Path]:
    cid = str(checkpoint_assessment.get("recommendedCheckpointId") or "").strip()
    if not cid:
        return None
    root = recovery_checkpoints_root(ace_root) / cid
    if root.is_dir():
        return root
    return None


def _build_restore_plan_if_possible(
    ace_root: Path,
    checkpoint_assessment: Dict[str, Any],
) -> Optional[AutoRestorePlan]:
    cp_root = _recommended_checkpoint_root(ace_root, checkpoint_assessment)
    if cp_root is None:
        return None
    backend, frontend = _workspace_roots(ace_root)
    if backend is None or frontend is None:
        plan = AutoRestorePlan(
            checkpoint_id=cp_root.name,
            checkpoint_root=str(cp_root),
            checkpoint_valid=validate_recovery_checkpoint_root(cp_root).ok,
        )
        plan.notes.append("workspace_roots_missing_file_recovery_deferred")
        plan.safe_auto_apply = plan.checkpoint_valid
        return plan
    cid = str(checkpoint_assessment.get("recommendedCheckpointId") or cp_root.name)
    return build_auto_restore_plan(
        ace_root=ace_root,
        checkpoint_id=cid,
        backend_root=backend,
        frontend_root=frontend,
    )


def _gui_recovery_mode(outcome: str) -> str:
    if outcome == COORDINATOR_START_ALLOWED:
        return "none"
    if outcome == COORDINATOR_AUTO_RECOVERY_ALLOWED:
        return "auto"
    if outcome == COORDINATOR_START_WITH_ISOLATED_OPERATIONS:
        return "isolated"
    if outcome == COORDINATOR_RECOVERY_UNSAFE:
        return "unsafe"
    if outcome == COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP:
        return "legacy_bootstrap"
    return "blocked"


def _legacy_bootstrap_baseline_established(
    ace_root: Path,
    ledger: Optional[Dict[str, Any]],
    unfinished: UnfinishedOperationsAssessment,
) -> bool:
    if not ledger or not isinstance(ledger, dict):
        return False
    boot = ledger.get("legacyRecoveryBootstrap")
    if not isinstance(boot, dict):
        return False
    if str(boot.get("status") or "").upper() != "COMPLETED":
        return False
    if boot.get("noVerifiedCheckpointYet"):
        return False
    lvc = ledger.get("lastVerifiedCheckpoint")
    if not isinstance(lvc, dict):
        return False
    cid = str(lvc.get("checkpointId") or "").strip()
    if not cid:
        return False
    if unfinished.isolated or unfinished.publish_unknown_external:
        return False
    root = recovery_checkpoints_root(ace_root) / cid
    if not root.is_dir():
        return False
    return validate_recovery_checkpoint_root(root).ok


def _derive_coordinator_outcome(
    *,
    startup: StartupRecoveryDecision,
    ledger_corrupt: bool,
    ledger: Optional[Dict[str, Any]],
    ace_root: Path,
    checkpoint_assessment: Dict[str, Any],
    unfinished: UnfinishedOperationsAssessment,
    restore_plan: Optional[AutoRestorePlan],
) -> Tuple[str, List[str]]:
    extra: List[str] = []
    startup_outcome = startup.outcome
    has_isolated = bool(unfinished.isolated)
    cp_ok = bool(checkpoint_assessment.get("recommendedValidationOk")) or bool(
        checkpoint_assessment.get("lastValidationOk")
    )
    if checkpoint_assessment.get("recommendedCheckpointId"):
        cp_ok = cp_ok or bool(checkpoint_assessment.get("recommendedValidationOk") is not False)

    if startup_outcome in (STARTUP_FIRST_START, STARTUP_NORMAL_START):
        return COORDINATOR_START_ALLOWED, extra

    if startup_outcome == STARTUP_LEGACY_INSTALL_RECOVERY_BOOTSTRAP:
        legacy_checks = dict(startup.checks.get("legacyBootstrap") or {})
        if legacy_checks.get("bootstrapAllowed"):
            extra.append("legacy_one_time_bootstrap")
            return COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP, extra
        extra.append("legacy_bootstrap_not_allowed")
        return COORDINATOR_RECOVERY_UNSAFE, extra

    if ledger_corrupt:
        extra.append("ledger_corrupt_preserve_evidence")
        return COORDINATOR_RECOVERY_UNSAFE, extra

    if _legacy_bootstrap_baseline_established(ace_root, ledger, unfinished):
        extra.append("legacy_bootstrap_completed_baseline")
        return COORDINATOR_START_ALLOWED, extra

    if startup_outcome == STARTUP_CRITICAL_RECOVERY or has_isolated:
        extra.append("isolated_dev_operations")
        return COORDINATOR_START_WITH_ISOLATED_OPERATIONS, extra

    if startup_outcome == STARTUP_RECOVERY_REVIEW:
        if restore_plan and restore_plan.safe_auto_apply and not restore_plan.has_conflicts:
            extra.append("auto_recovery_with_review_context")
            return COORDINATOR_AUTO_RECOVERY_ALLOWED, extra
        extra.append("recovery_review_isolated_start")
        return COORDINATOR_START_WITH_ISOLATED_OPERATIONS, extra

    if startup_outcome == STARTUP_RECOVERY_REQUIRED:
        if restore_plan and restore_plan.safe_auto_apply:
            extra.append("auto_recovery_safe_plan")
            return COORDINATOR_AUTO_RECOVERY_ALLOWED, extra
        if restore_plan and restore_plan.checkpoint_valid and restore_plan.has_conflicts:
            extra.append("file_conflicts_preserve_both")
            return COORDINATOR_START_WITH_ISOLATED_OPERATIONS, extra
        if cp_ok or (restore_plan and restore_plan.checkpoint_valid):
            extra.append("checkpoint_valid_manual_file_merge_may_be_needed")
            return COORDINATOR_START_WITH_ISOLATED_OPERATIONS, extra
        extra.append("no_verified_checkpoint_for_auto_recovery")
        return COORDINATOR_RECOVERY_UNSAFE, extra

    extra.append("unclassified_startup_outcome")
    return COORDINATOR_RECOVERY_UNSAFE, extra


def _flags_for_outcome(
    outcome: str,
    *,
    ledger_corrupt: bool,
    unfinished: UnfinishedOperationsAssessment,
    restore_plan: Optional[AutoRestorePlan],
) -> Dict[str, bool]:
    if outcome == COORDINATOR_START_BLOCKED:
        return {
            "allow_local_processes": False,
            "allow_dev_work": False,
            "allow_new_launcher_session": False,
            "allow_ledger_initialization": False,
            "allow_periodic_save_service": False,
        }

    if outcome == COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP:
        return {
            "allow_local_processes": True,
            "allow_dev_work": True,
            "allow_new_launcher_session": False,
            "allow_ledger_initialization": True,
            "allow_periodic_save_service": False,
        }

    allow_local = outcome in _LOCAL_START_OUTCOMES
    allow_dev = outcome in (
        COORDINATOR_START_ALLOWED,
        COORDINATOR_AUTO_RECOVERY_ALLOWED,
        COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
        COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP,
    )
    allow_new_session = outcome in _LOCAL_START_OUTCOMES and not ledger_corrupt
    allow_ledger = (allow_new_session and outcome != COORDINATOR_RECOVERY_UNSAFE) or (
        outcome == COORDINATOR_LEGACY_RECOVERY_BOOTSTRAP
    )

    periodic = False
    if outcome in (COORDINATOR_START_ALLOWED, COORDINATOR_AUTO_RECOVERY_ALLOWED):
        periodic = not unfinished.isolated and not unfinished.publish_unknown_external
        if restore_plan is not None and not restore_plan.checkpoint_valid:
            periodic = False

    return {
        "allow_local_processes": allow_local,
        "allow_dev_work": allow_dev,
        "allow_new_launcher_session": allow_new_session,
        "allow_ledger_initialization": allow_ledger,
        "allow_periodic_save_service": periodic,
    }


@dataclass
class RecoveryStartupCoordinatorDecision:
    coordinator_outcome: str
    recovery_status: str
    termination_type: str
    confidence: str
    evidence: List[Dict[str, Any]]
    previous_launcher_session: Optional[str]
    last_verified_checkpoint: Optional[Dict[str, Any]]
    allow_local_processes: bool
    allow_dev_work: bool
    allow_new_launcher_session: bool
    allow_ledger_initialization: bool
    allow_periodic_save_service: bool
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    checkpoint_assessment: Dict[str, Any] = field(default_factory=dict)
    periodic_save_contract: Dict[str, Any] = field(default_factory=dict)
    auto_restore_plan: Dict[str, Any] = field(default_factory=dict)
    unfinished_operations: Dict[str, Any] = field(default_factory=dict)
    isolated_operations: List[Dict[str, Any]] = field(default_factory=list)
    blocked_dev_operation_kinds: List[str] = field(default_factory=list)
    evidence_preservation: Dict[str, Any] = field(default_factory=dict)
    gui_recovery_mode: str = "none"

    @property
    def startup_allowed(self) -> bool:
        return self.coordinator_outcome in _LOCAL_START_OUTCOMES


def build_periodic_save_activation_contract(
    decision: RecoveryStartupCoordinatorDecision,
    *,
    launcher_session: str = "",
) -> Dict[str, Any]:
    allowed = decision.allow_periodic_save_service and bool(launcher_session.strip())
    return {
        "allowed": allowed,
        "requiresCoordinatorOutcomeIn": [
            COORDINATOR_START_ALLOWED,
            COORDINATOR_AUTO_RECOVERY_ALLOWED,
        ],
        "requiresMatchingLauncherSession": True,
        "launcherSession": (launcher_session or "").strip(),
        "mustNotInitializeLedger": True,
        "mustNotRunTwice": True,
        "controlledStopRequired": True,
        "blockedWhenRecoveryUnresolved": True,
        "blockedWhenIsolatedOperations": bool(decision.isolated_operations),
        "note": "Contract only; service not started in phase 5.1.",
    }


def evaluate_recovery_startup_coordination(
    ace_root: Path,
    *,
    sessions_root: Optional[Path] = None,
    inspect_live_processes: bool = True,
    backend_reachable: bool = True,
    acquire_coordinator_lock: bool = True,
    termination_evidence_override: Optional[Dict[str, Any]] = None,
) -> RecoveryStartupCoordinatorDecision:
    ace_root = Path(ace_root)
    lock_token: Optional[str] = None
    if acquire_coordinator_lock:
        lock = try_acquire_startup_coordinator_lock(ace_root)
        if not lock.get("acquired"):
            contract = build_periodic_save_activation_contract(
                RecoveryStartupCoordinatorDecision(
                    coordinator_outcome=COORDINATOR_START_BLOCKED,
                    recovery_status=COORDINATOR_START_BLOCKED,
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
                    reasons=[str(lock.get("reason") or "coordinator_lock_held")],
                    gui_recovery_mode="blocked",
                )
            )
            return RecoveryStartupCoordinatorDecision(
                coordinator_outcome=COORDINATOR_START_BLOCKED,
                recovery_status=COORDINATOR_START_BLOCKED,
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
                reasons=[str(lock.get("reason") or "coordinator_lock_held")],
                checks={"coordinatorLock": lock},
                periodic_save_contract=contract,
                gui_recovery_mode="blocked",
            )
        lock_token = str(lock.get("token") or "")

    try:
        startup: StartupRecoveryDecision = evaluate_previous_local_startup(
            ace_root,
            sessions_root=sessions_root,
            inspect_live_processes=inspect_live_processes,
            backend_reachable=backend_reachable,
            termination_evidence_override=termination_evidence_override,
        )
        ledger, ledger_corrupt = read_recovery_ledger(ace_root)
        checkpoint = assess_recovery_checkpoints(ace_root, ledger=ledger if not ledger_corrupt else None)
        cp_dict = checkpoint.to_dict()

        root = sessions_root if sessions_root is not None else (ace_root / "_backups" / "sessions")
        scan = observe_active_local_dev_operations(
            ace_root=ace_root,
            sessions_root=root,
            backend_reachable=backend_reachable,
        )
        observations = list(scan.get("observations") or [])
        unfinished = assess_unfinished_dev_operations(observations, sessions_root=root)
        restore_plan = _build_restore_plan_if_possible(ace_root, cp_dict)

        coordinator_outcome, policy_notes = _derive_coordinator_outcome(
            startup=startup,
            ledger_corrupt=ledger_corrupt,
            ledger=ledger if not ledger_corrupt else None,
            ace_root=ace_root,
            checkpoint_assessment=cp_dict,
            unfinished=unfinished,
            restore_plan=restore_plan,
        )
        flags = _flags_for_outcome(
            coordinator_outcome,
            ledger_corrupt=ledger_corrupt,
            unfinished=unfinished,
            restore_plan=restore_plan,
        )

        last_cp: Optional[Dict[str, Any]] = None
        if ledger and isinstance(ledger.get("lastVerifiedCheckpoint"), dict):
            last_cp = dict(ledger["lastVerifiedCheckpoint"])

        reasons = list(startup.reasons) + policy_notes
        if checkpoint.notes:
            reasons.extend(checkpoint.notes)

        evidence_preservation = {
            "preservePriorLauncherSession": startup.previous_launcher_session,
            "preserveLedgerUntilIncidentRecorded": ledger_corrupt
            or coordinator_outcome == COORDINATOR_RECOVERY_UNSAFE,
            "startupOutcome": startup.outcome,
            "observationEvidence": list(unfinished.evidence_snapshot),
            "automaticRecoveryPolicy": "phase_5_1_no_user_prompt",
        }

        decision = RecoveryStartupCoordinatorDecision(
            coordinator_outcome=coordinator_outcome,
            recovery_status=startup.recovery_status,
            termination_type=startup.termination_type,
            confidence=startup.confidence,
            evidence=list(startup.evidence),
            previous_launcher_session=startup.previous_launcher_session,
            last_verified_checkpoint=last_cp,
            allow_local_processes=flags["allow_local_processes"],
            allow_dev_work=flags["allow_dev_work"],
            allow_new_launcher_session=flags["allow_new_launcher_session"],
            allow_ledger_initialization=flags["allow_ledger_initialization"],
            allow_periodic_save_service=flags["allow_periodic_save_service"],
            reasons=reasons,
            checks={
                **startup.checks,
                "startupOutcome": startup.outcome,
                "coordinatorOutcome": coordinator_outcome,
                "coordinatorLockToken": lock_token,
                "observationCount": len(observations),
            },
            checkpoint_assessment=cp_dict,
            auto_restore_plan=restore_plan.to_dict() if restore_plan else {},
            unfinished_operations=unfinished.to_dict(),
            isolated_operations=[row.to_dict() for row in unfinished.isolated],
            blocked_dev_operation_kinds=list(unfinished.blocked_dev_kinds),
            evidence_preservation=evidence_preservation,
            gui_recovery_mode=_gui_recovery_mode(coordinator_outcome),
        )
        decision.periodic_save_contract = build_periodic_save_activation_contract(decision)
        return decision
    finally:
        if acquire_coordinator_lock and lock_token:
            release_startup_coordinator_lock(ace_root)


def coordinator_decision_to_dict(decision: RecoveryStartupCoordinatorDecision) -> Dict[str, Any]:
    return {
        "coordinatorOutcome": decision.coordinator_outcome,
        "recoveryStatus": decision.recovery_status,
        "terminationType": decision.termination_type,
        "confidence": decision.confidence,
        "evidence": list(decision.evidence),
        "previousLauncherSession": decision.previous_launcher_session,
        "lastVerifiedCheckpoint": decision.last_verified_checkpoint,
        "allowLocalProcesses": decision.allow_local_processes,
        "allowDevWork": decision.allow_dev_work,
        "allowNewLauncherSession": decision.allow_new_launcher_session,
        "allowLedgerInitialization": decision.allow_ledger_initialization,
        "allowPeriodicSaveService": decision.allow_periodic_save_service,
        "startupAllowed": decision.startup_allowed,
        "guiRecoveryMode": decision.gui_recovery_mode,
        "reasons": list(decision.reasons),
        "checks": decision.checks,
        "checkpointAssessment": decision.checkpoint_assessment,
        "autoRestorePlan": decision.auto_restore_plan,
        "unfinishedOperations": decision.unfinished_operations,
        "isolatedOperations": list(decision.isolated_operations),
        "blockedDevOperationKinds": list(decision.blocked_dev_operation_kinds),
        "evidencePreservation": decision.evidence_preservation,
        "periodicSaveContract": decision.periodic_save_contract,
    }


def assert_coordinator_allows_new_launcher_session(
    decision: RecoveryStartupCoordinatorDecision,
) -> None:
    if not decision.allow_new_launcher_session:
        raise RuntimeError(
            f"ace_recovery_startup_coordinator_blocked:{decision.coordinator_outcome}"
        )


def assert_coordinator_allows_periodic_save(
    decision: RecoveryStartupCoordinatorDecision,
    *,
    launcher_session: str,
) -> None:
    if not decision.allow_periodic_save_service:
        raise RuntimeError(
            f"ace_recovery_periodic_save_blocked:{decision.coordinator_outcome}"
        )
    contract = build_periodic_save_activation_contract(decision, launcher_session=launcher_session)
    if not contract.get("allowed"):
        raise RuntimeError("ace_recovery_periodic_save_contract_not_satisfied")
