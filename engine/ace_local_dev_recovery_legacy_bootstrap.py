"""
One-time legacy LOCAL install recovery bootstrap assessment (read-only unless explicitly applied).

For installs that used LOCAL before recovery ledger/checkpoints existed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_recovery_checkpoint_validate import (
    list_verified_checkpoint_records,
    recovery_checkpoints_root,
)
from engine.ace_local_dev_shutdown_gate import (
    observe_active_local_dev_operations,
    read_managed_launcher_session,
    resolve_local_dev_state_dir,
)

RECOVERY_LEDGER_FILENAME = "recovery-session-ledger.json"


def recovery_ledger_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / RECOVERY_LEDGER_FILENAME


def read_recovery_ledger(ace_root: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    path = recovery_ledger_path(ace_root)
    if not path.is_file():
        return None, False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, True
    return (data if isinstance(data, dict) else None), False


def read_managed_registry(ace_root: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    path = resolve_local_dev_state_dir(ace_root) / "managed-processes.json"
    if not path.is_file():
        return None, False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, True
    return (data if isinstance(data, dict) else None), False

CLASSIFICATION_LEGACY = "LEGACY_INSTALL_RECOVERY_BOOTSTRAP"

BOOTSTRAP_STATE_FILENAME = "recovery-legacy-bootstrap-state.json"
BOOTSTRAP_LOCK_FILENAME = "recovery-legacy-bootstrap.lock"
RESTORE_JOURNAL_FILENAME = "recovery-auto-restore-journal.json"


def read_restore_journal(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = resolve_local_dev_state_dir(ace_root) / RESTORE_JOURNAL_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

BOOTSTRAP_STATUS_IDLE = "IDLE"
BOOTSTRAP_STATUS_IN_PROGRESS = "IN_PROGRESS"
BOOTSTRAP_STATUS_FAILED = "FAILED"
BOOTSTRAP_STATUS_COMPLETED = "COMPLETED"


def bootstrap_state_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / BOOTSTRAP_STATE_FILENAME


def bootstrap_lock_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / BOOTSTRAP_LOCK_FILENAME


def read_bootstrap_state(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = bootstrap_state_path(ace_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _bootstrap_lock_held(ace_root: Path) -> Tuple[bool, Optional[str]]:
    path = bootstrap_lock_path(ace_root)
    if not path.is_file():
        return False, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return True, "bootstrap_lock_unreadable"
    pid = int(payload.get("pid") or 0)
    if pid and _pid_alive(pid):
        return True, "bootstrap_lock_live"
    return False, "bootstrap_lock_stale"


def _any_checkpoint_dirs(ace_root: Path) -> bool:
    root = recovery_checkpoints_root(ace_root)
    if not root.is_dir():
        return False
    for entry in root.iterdir():
        if entry.is_dir():
            return True
    return False


def _read_json_file(path: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not path.is_file():
        return None, False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, True
    return (data if isinstance(data, dict) else None), False


def _observation_blocks_bootstrap(obs: Dict[str, Any]) -> Optional[str]:
    kind = str(obs.get("kind") or "")
    status = str(obs.get("status") or "").upper()
    notes = " ".join(str(n) for n in (obs.get("notes") or []))
    if status == "PARTIAL":
        return f"critical_partial:{kind}"
    if obs.get("cancelCritical"):
        return f"cancel_critical:{kind}"
    if obs.get("publishExternalUnknown") or obs.get("publishExternalRunning"):
        return "publish_external_unfinished"
    if kind in ("CANCEL", "PUBLISH_EXECUTE", "PUBLISH_PREPARE") and status == "ACTIVE":
        if kind == "CANCEL" and obs.get("cancelCritical"):
            return "cancel_active_critical"
        if kind == "PUBLISH_EXECUTE" and (
            obs.get("publishExternalUnknown")
            or obs.get("publishExternalRunning")
            or str(obs.get("currentStep") or "")
            in ("git_push_backend", "github_pages", "registry_update", "render_web_deploy")
        ):
            return "publish_execute_active"
    if "cancel_journal_partial" in notes or "publish_partial" in notes:
        return "partial_journal"
    if "publish_unknown:" in notes:
        return "publish_unknown"
    return None


@dataclass
class LegacyBootstrapAssessment:
    classification: str
    bootstrap_allowed: bool
    block_reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    managed_launcher_session: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classification": self.classification,
            "bootstrapAllowed": self.bootstrap_allowed,
            "blockReasons": list(self.block_reasons),
            "checks": dict(self.checks),
            "managedLauncherSession": self.managed_launcher_session,
        }


def assess_legacy_install_recovery_bootstrap(
    ace_root: Path,
    *,
    sessions_root: Optional[Path] = None,
    execute_busy: bool = False,
    backend_reachable: bool = True,
) -> LegacyBootstrapAssessment:
    """
    Read-only eligibility for one-time legacy recovery bootstrap.
    """
    ace_root = Path(ace_root)
    block: List[str] = []
    checks: Dict[str, Any] = {}

    ledger, ledger_corrupt = read_recovery_ledger(ace_root)
    ledger_path = recovery_ledger_path(ace_root)
    ledger_exists = ledger_path.is_file()

    registry, registry_corrupt = read_managed_registry(ace_root)
    managed_session = read_managed_launcher_session(ace_root) if not registry_corrupt else None

    checks["ledgerFileExists"] = ledger_exists
    checks["ledgerCorrupt"] = ledger_corrupt
    checks["registryCorrupt"] = registry_corrupt
    checks["managedLauncherSession"] = managed_session

    verified = list_verified_checkpoint_records(ace_root)
    checks["verifiedCheckpointCount"] = len(verified)
    checkpoint_dirs = _any_checkpoint_dirs(ace_root)
    checks["checkpointDirPresent"] = checkpoint_dirs

    restore_journal = read_restore_journal(ace_root)
    checks["restoreJournalPresent"] = restore_journal is not None

    bootstrap_state = read_bootstrap_state(ace_root)
    checks["bootstrapStateStatus"] = (
        str(bootstrap_state.get("status") or "") if bootstrap_state else None
    )

    lock_held, lock_note = _bootstrap_lock_held(ace_root)
    checks["bootstrapLockHeld"] = lock_held
    if lock_held and lock_note:
        block.append(lock_note)

    open_cycle_path = ace_root / "_backups" / "sessions" / "active-dev-cycle.json"
    prod_reg_path = ace_root / "_registry" / "production-registry.json"
    open_cycle, open_corrupt = _read_json_file(open_cycle_path)
    prod_reg, prod_corrupt = _read_json_file(prod_reg_path)
    checks["openCycleReadable"] = open_cycle_path.is_file() and not open_corrupt
    checks["productionRegistryReadable"] = prod_reg_path.is_file() and not prod_corrupt
    if open_cycle_path.is_file() and open_corrupt:
        block.append("open_cycle_corrupt")
    if prod_reg_path.is_file() and prod_corrupt:
        block.append("production_registry_corrupt")

    if registry_corrupt:
        block.append("managed_registry_corrupt")
    if ledger_corrupt:
        block.append("recovery_ledger_corrupt")

    if not managed_session:
        return LegacyBootstrapAssessment(
            classification="",
            bootstrap_allowed=False,
            block_reasons=["no_managed_launcher_session"],
            checks=checks,
        )

    is_legacy_shape = (
        not ledger_exists
        and not ledger_corrupt
        and len(verified) == 0
        and restore_journal is None
    )
    checks["legacyInstallShape"] = is_legacy_shape

    if ledger_exists and ledger and not ledger_corrupt:
        if ledger.get("openIncident"):
            block.append("recovery_incident_in_ledger")
        bootstrap_meta = ledger.get("legacyRecoveryBootstrap")
        if isinstance(bootstrap_meta, dict):
            st = str(bootstrap_meta.get("status") or "").upper()
            checks["ledgerBootstrapStatus"] = st
            if st == "COMPLETED":
                block.append("legacy_bootstrap_already_completed")
            if ledger.get("lastVerifiedCheckpoint"):
                block.append("recovery_already_initialized_with_checkpoint")
        else:
            if ledger.get("lastVerifiedCheckpoint"):
                block.append("recovery_already_initialized_with_checkpoint")
            elif not isinstance(bootstrap_meta, dict):
                block.append("recovery_ledger_without_bootstrap_or_checkpoint")

    if checkpoint_dirs:
        block.append("recovery_checkpoint_artifacts_present")
    if restore_journal is not None:
        block.append("recovery_restore_journal_present")

    if bootstrap_state and str(bootstrap_state.get("status") or "").upper() == BOOTSTRAP_STATUS_IN_PROGRESS:
        if not lock_held:
            checks["bootstrapStateStaleInProgress"] = True
        else:
            block.append("bootstrap_already_in_progress")

    scan = observe_active_local_dev_operations(
        ace_root=ace_root,
        sessions_root=sessions_root,
        execute_busy=execute_busy,
        backend_reachable=backend_reachable,
    )
    observations = list(scan.get("observations") or [])
    checks["observationCount"] = len(observations)
    for obs in observations:
        reason = _observation_blocks_bootstrap(obs)
        if reason:
            block.append(reason)

    classification = CLASSIFICATION_LEGACY if is_legacy_shape else ""
    bootstrap_allowed = is_legacy_shape and len(block) == 0

    if is_legacy_shape and block:
        checks["legacyBootstrapBlocked"] = True

    return LegacyBootstrapAssessment(
        classification=classification,
        bootstrap_allowed=bootstrap_allowed,
        block_reasons=block,
        checks=checks,
        managed_launcher_session=managed_session,
    )
