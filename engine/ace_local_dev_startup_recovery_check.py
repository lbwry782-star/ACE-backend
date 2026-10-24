"""
LOCAL startup recovery check — read-only evaluation of the previous launcher session.

Must run before creating a new launcherSession or re-initializing the recovery ledger.
Not wired to Start-AceLocal in this phase.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_cycle_store import resolve_sessions_root
from engine.ace_local_dev_termination_classification import (
    ProcessInspector,
    TERMINATION_NONE,
    TERMINATION_UNKNOWN,
    classify_local_termination,
    read_termination_evidence_file,
    termination_classification_to_dict,
)
from engine.ace_local_dev_recovery_legacy_bootstrap import (
    CLASSIFICATION_LEGACY,
    assess_legacy_install_recovery_bootstrap,
)
from engine.ace_local_dev_shutdown_gate import (
    GATE_RUNNING,
    GATE_SAFE_TO_STOP,
    GATE_SHUTDOWN_BLOCKED,
    GATE_SHUTDOWN_REQUESTED,
    GATE_WAITING_FOR_OPERATION,
    observe_active_local_dev_operations,
    read_managed_launcher_session,
    read_shutdown_gate_state,
    resolve_local_dev_state_dir,
    shutdown_gate_state_path,
)

logger = logging.getLogger(__name__)

RECOVERY_LEDGER_FILENAME = "recovery-session-ledger.json"
RECOVERY_LEDGER_SCHEMA_VERSION = 1
LAST_VERIFIED_SHUTDOWN_SNAPSHOT = "last-verified-shutdown-snapshot.json"

STARTUP_FIRST_START = "FIRST_START"
STARTUP_NORMAL_START = "NORMAL_START"
STARTUP_RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
STARTUP_RECOVERY_REVIEW = "RECOVERY_REVIEW"
STARTUP_CRITICAL_RECOVERY = "CRITICAL_RECOVERY"
STARTUP_LEGACY_INSTALL_RECOVERY_BOOTSTRAP = "LEGACY_INSTALL_RECOVERY_BOOTSTRAP"

_SHUTDOWN_IN_FLIGHT = frozenset(
    {
        GATE_SHUTDOWN_REQUESTED,
        GATE_WAITING_FOR_OPERATION,
        GATE_SHUTDOWN_BLOCKED,
    }
)


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


def read_last_verified_shutdown_snapshot_pointer(ace_root: Path) -> Tuple[Optional[Dict[str, Any]], bool]:
    path = (
        resolve_local_dev_state_dir(ace_root)
        / "shutdown-snapshots"
        / LAST_VERIFIED_SHUTDOWN_SNAPSHOT
    )
    if not path.is_file():
        return None, False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, True
    return (data if isinstance(data, dict) else None), False


def _parse_iso_utc(text: Optional[str]) -> Optional[datetime]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


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


def _collect_managed_live_pids(registry: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not registry:
        return []
    components = registry.get("components")
    if not isinstance(components, dict):
        return []
    live: List[Dict[str, Any]] = []
    for name, row in components.items():
        if not isinstance(row, dict):
            continue
        pid = row.get("pid") or row.get("rootPid")
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        if _pid_alive(pid_int):
            live.append({"component": str(name), "pid": pid_int})
    return live


def _clean_shutdown_verified(ledger: Optional[Dict[str, Any]], session: str) -> bool:
    if not ledger or not session:
        return False
    cs = ledger.get("cleanShutdown")
    if not isinstance(cs, dict):
        return False
    if not cs.get("completed"):
        return False
    bound = str(cs.get("launcherSession") or "").strip()
    return bound == session


def _ledger_valid(ledger: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if not ledger:
        return False, "ledger_missing"
    try:
        if int(ledger.get("schemaVersion", -1)) != RECOVERY_LEDGER_SCHEMA_VERSION:
            return False, "schema_unsupported"
    except (TypeError, ValueError):
        return False, "schema_invalid"
    if not str(ledger.get("launcherSession") or "").strip():
        return False, "launcher_session_missing"
    return True, "ok"


def _has_meaningful_launch_history(
    ledger: Optional[Dict[str, Any]],
    registry: Optional[Dict[str, Any]],
    gate: Optional[Dict[str, Any]],
) -> bool:
    if ledger and _ledger_valid(ledger)[0]:
        return True
    if registry and str(registry.get("launcherSession") or "").strip():
        return True
    if gate and str(gate.get("launcherSession") or "").strip():
        return True
    return False


def _observation_is_critical(obs: Dict[str, Any]) -> bool:
    kind = str(obs.get("kind") or "")
    status = str(obs.get("status") or "").upper()
    notes = " ".join(str(n) for n in (obs.get("notes") or []))

    if status == "PARTIAL":
        return True
    if obs.get("cancelCritical"):
        return True
    if obs.get("publishExternalUnknown") or obs.get("publishExternalRunning"):
        return True
    if kind == "CANCEL" and status == "ACTIVE" and obs.get("cancelCritical"):
        return True
    if kind in ("CANCEL", "PUBLISH_EXECUTE", "PUBLISH_PREPARE") and status == "ACTIVE":
        if kind == "CANCEL" and obs.get("cancelCritical"):
            return True
        if kind == "PUBLISH_EXECUTE" and (
            obs.get("publishExternalUnknown") or obs.get("publishExternalRunning")
        ):
            return True
        if kind == "PUBLISH_EXECUTE" and str(obs.get("currentStep") or "") in (
            "git_push_backend",
            "github_pages",
            "registry_update",
        ):
            return True
    if "cancel_journal_partial" in notes or "publish_partial" in notes:
        return True
    if "publish_unknown:" in notes:
        return True
    if kind == "PUBLISH_EXECUTE" and status == "ACTIVE":
        step = str(obs.get("currentStep") or "")
        if step in ("git_push_backend", "github_pages", "registry_update", "render_web_deploy"):
            return True
    return False


def _observation_is_active_or_unknown(obs: Dict[str, Any]) -> bool:
    status = str(obs.get("status") or "").upper()
    if status in ("ACTIVE", "UNKNOWN", "PARTIAL"):
        return True
    if obs.get("publishExternalUnknown") or obs.get("publishExternalRunning"):
        return True
    return False


def _filter_stale_progress_observations(
    observations: List[Dict[str, Any]],
    *,
    sessions_root: Path,
    clean_shutdown_at: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not clean_shutdown_at:
        return observations, []
    kept: List[Dict[str, Any]] = []
    stale_notes: List[str] = []
    for obs in observations:
        if not _observation_is_active_or_unknown(obs):
            continue
        op_id = str(obs.get("operationId") or "")
        sid = str(obs.get("sessionId") or "")
        if not op_id or not sid:
            kept.append(obs)
            continue
        session_dir = sessions_root / sid
        if not session_dir.is_dir():
            kept.append(obs)
            continue
        prefix = None
        kind = str(obs.get("kind") or "")
        if kind == "CANCEL":
            prefix = "cancel-execute-progress-"
        elif kind == "PUBLISH_PREPARE":
            prefix = "publish-prepare-progress-"
        elif kind == "PUBLISH_EXECUTE":
            prefix = "publish-execute-progress-"
        if not prefix:
            kept.append(obs)
            continue
        path = session_dir / f"{prefix}{op_id}.json"
        if not path.is_file():
            kept.append(obs)
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < clean_shutdown_at:
            stale_notes.append(f"stale_progress_ignored:{kind}:{op_id}")
            continue
        kept.append(obs)
    inactive = [o for o in observations if not _observation_is_active_or_unknown(o)]
    return inactive + kept, stale_notes


@dataclass
class StartupRecoveryDecision:
    outcome: str
    dev_operations_allowed: bool
    previous_launcher_session: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    termination_type: str = TERMINATION_UNKNOWN
    confidence: str = "LOW"
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    component_findings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def recovery_status(self) -> str:
        return self.outcome


def _attach_termination(
    decision: StartupRecoveryDecision,
    *,
    launcher_session: str,
    session_started_at: Optional[str],
    clean_shutdown_verified: bool,
    registry: Optional[Dict[str, Any]],
    first_start: bool,
    termination_evidence: Optional[Dict[str, Any]],
    inspector: Optional[ProcessInspector],
) -> StartupRecoveryDecision:
    classification = classify_local_termination(
        launcher_session=launcher_session,
        session_started_at=session_started_at,
        clean_shutdown_verified=clean_shutdown_verified,
        registry=registry,
        termination_evidence=termination_evidence,
        first_start=first_start,
        inspector=inspector,
    )
    decision.termination_type = classification.termination_type
    decision.confidence = classification.confidence
    decision.evidence = list(classification.evidence)
    decision.component_findings = list(classification.component_findings)
    decision.checks = {
        **decision.checks,
        "terminationClassification": termination_classification_to_dict(classification),
    }
    return decision


def _finish_decision(
    outcome: str,
    *,
    dev_operations_allowed: bool,
    previous_launcher_session: Optional[str],
    reasons: List[str],
    checks: Dict[str, Any],
    termination_ctx: Dict[str, Any],
) -> StartupRecoveryDecision:
    decision = StartupRecoveryDecision(
        outcome=outcome,
        dev_operations_allowed=dev_operations_allowed,
        previous_launcher_session=previous_launcher_session,
        reasons=reasons,
        checks=checks,
    )
    return _attach_termination(decision, **termination_ctx)


def evaluate_previous_local_startup(
    ace_root: Path,
    *,
    sessions_root: Optional[Path] = None,
    inspect_live_processes: bool = True,
    backend_reachable: bool = True,
    stage_timeout_seconds: Optional[float] = None,
    termination_evidence_override: Optional[Dict[str, Any]] = None,
    process_inspector: Optional[ProcessInspector] = None,
) -> StartupRecoveryDecision:
    """
    Read-only classification of how the previous LOCAL session ended.
    Does not mutate ace_root state, git, registry, or processes.
    """
    ace_root = Path(ace_root)
    root = sessions_root if sessions_root is not None else resolve_sessions_root()

    local_dev_dir = resolve_local_dev_state_dir(ace_root)
    ledger, ledger_corrupt = read_recovery_ledger(ace_root)
    registry, registry_corrupt = read_managed_registry(ace_root)
    gate = read_shutdown_gate_state(ace_root)
    snapshot_ptr, snapshot_ptr_corrupt = read_last_verified_shutdown_snapshot_pointer(ace_root)
    term_file, term_file_corrupt = read_termination_evidence_file(local_dev_dir)
    termination_evidence = (
        termination_evidence_override
        if termination_evidence_override is not None
        else term_file
    )

    reasons: List[str] = []
    checks: Dict[str, Any] = {
        "ledgerCorrupt": ledger_corrupt,
        "registryCorrupt": registry_corrupt,
        "snapshotPointerCorrupt": snapshot_ptr_corrupt,
        "gatePresent": gate is not None,
        "snapshotPointerPresent": snapshot_ptr is not None,
        "terminationEvidenceCorrupt": term_file_corrupt,
        "terminationEvidenceFromFile": termination_evidence_override is None and term_file is not None,
    }

    if ledger_corrupt:
        reasons.append("recovery_ledger_corrupt")
    if registry_corrupt:
        reasons.append("managed_registry_corrupt")
    if snapshot_ptr_corrupt:
        reasons.append("shutdown_snapshot_pointer_corrupt")

    ledger_ok, ledger_reason = _ledger_valid(ledger)
    previous_session = ""
    if ledger_ok and ledger:
        previous_session = str(ledger.get("launcherSession") or "").strip()
    elif registry and not registry_corrupt:
        previous_session = str(registry.get("launcherSession") or "").strip()
    elif gate:
        previous_session = str(gate.get("launcherSession") or "").strip()

    checks["previousLauncherSession"] = previous_session or None

    session_started_at = None
    if ledger_ok and ledger:
        session_started_at = str(ledger.get("sessionStartedAt") or "") or None

    def term_ctx(*, clean: bool, first_start: bool = False) -> Dict[str, Any]:
        return {
            "launcher_session": previous_session or "",
            "session_started_at": session_started_at,
            "clean_shutdown_verified": clean,
            "registry": registry if not registry_corrupt else None,
            "first_start": first_start,
            "termination_evidence": termination_evidence,
            "inspector": process_inspector,
        }

    ledger_file_exists = recovery_ledger_path(ace_root).is_file()

    if ledger_corrupt and ledger_file_exists:
        return _finish_decision(
            STARTUP_RECOVERY_REVIEW,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=False),
        )

    if not _has_meaningful_launch_history(ledger if not ledger_corrupt else None, registry, gate):
        return _finish_decision(
            STARTUP_FIRST_START,
            dev_operations_allowed=True,
            previous_launcher_session=None,
            reasons=["no_prior_launch_history"],
            checks=checks,
            termination_ctx=term_ctx(clean=False, first_start=True),
        )

    if not ledger_file_exists and not ledger_corrupt:
        legacy = assess_legacy_install_recovery_bootstrap(
            ace_root,
            sessions_root=root,
            backend_reachable=backend_reachable,
        )
        checks["legacyBootstrap"] = legacy.to_dict()
        if legacy.classification == CLASSIFICATION_LEGACY:
            if legacy.bootstrap_allowed:
                reasons.append("legacy_install_recovery_bootstrap_allowed")
                return _finish_decision(
                    STARTUP_LEGACY_INSTALL_RECOVERY_BOOTSTRAP,
                    dev_operations_allowed=True,
                    previous_launcher_session=previous_session or None,
                    reasons=reasons,
                    checks=checks,
                    termination_ctx=term_ctx(clean=False),
                )
            reasons.extend([f"legacy_bootstrap_blocked:{r}" for r in legacy.block_reasons])
            return _finish_decision(
                STARTUP_RECOVERY_REVIEW,
                dev_operations_allowed=False,
                previous_launcher_session=previous_session or None,
                reasons=reasons,
                checks=checks,
                termination_ctx=term_ctx(clean=False),
            )

    managed_session = read_managed_launcher_session(ace_root) if not registry_corrupt else None
    if (
        previous_session
        and managed_session
        and managed_session != previous_session
        and ledger_ok
    ):
        reasons.append("launcher_session_mismatch_ledger_vs_managed")

    gate_status = str((gate or {}).get("gateStatus") or GATE_RUNNING).upper()
    gate_session = str((gate or {}).get("launcherSession") or "").strip()
    if gate and gate_status in _SHUTDOWN_IN_FLIGHT:
        reasons.append(f"shutdown_gate_incomplete:{gate_status}")
        if gate_session and previous_session and gate_session != previous_session:
            reasons.append("shutdown_gate_session_mismatch")

    open_incident = None
    if ledger_ok and ledger:
        open_incident = ledger.get("openIncident")
        if open_incident:
            reasons.append("open_incident_present")

    clean_verified = _clean_shutdown_verified(ledger, previous_session) if ledger_ok else False
    checks["cleanShutdownVerified"] = clean_verified

    cs_at = None
    if ledger_ok and ledger:
        cs = ledger.get("cleanShutdown")
        if isinstance(cs, dict):
            cs_at = _parse_iso_utc(str(cs.get("completedAt") or ""))

    scan = observe_active_local_dev_operations(
        ace_root=ace_root,
        sessions_root=root,
        execute_busy=False,
        backend_reachable=backend_reachable,
        stage_timeout_seconds=stage_timeout_seconds,
    )
    observations = list(scan.get("observations") or [])
    checks["rawObservationCount"] = len(observations)

    filtered_obs, stale_notes = _filter_stale_progress_observations(
        observations,
        sessions_root=root,
        clean_shutdown_at=cs_at if clean_verified else None,
    )
    checks["staleProgressNotes"] = stale_notes
    checks["relevantObservationCount"] = len(filtered_obs)

    live_pids: List[Dict[str, Any]] = []
    if inspect_live_processes and registry and not registry_corrupt:
        live_pids = _collect_managed_live_pids(registry)
    checks["liveManagedProcesses"] = live_pids

    critical_obs = [o for o in filtered_obs if _observation_is_critical(o)]
    active_unknown_obs = [
        o for o in filtered_obs if _observation_is_active_or_unknown(o) and o not in critical_obs
    ]

    if critical_obs:
        for o in critical_obs:
            reasons.append(f"critical_operation:{o.get('kind')}:{o.get('status')}")
        return _finish_decision(
            STARTUP_CRITICAL_RECOVERY,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks={**checks, "criticalObservations": critical_obs},
            termination_ctx=term_ctx(clean=clean_verified),
        )

    if ledger_ok and ledger and open_incident:
        return _finish_decision(
            STARTUP_RECOVERY_REQUIRED,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=False),
        )

    if active_unknown_obs:
        for o in active_unknown_obs:
            reasons.append(f"active_or_unknown:{o.get('kind')}:{o.get('status')}")
        outcome = STARTUP_RECOVERY_REVIEW
        if any(str(o.get("status") or "").upper() == "UNKNOWN" for o in active_unknown_obs):
            reasons.append("unknown_operation_state")
        return _finish_decision(
            outcome,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks={**checks, "activeObservations": active_unknown_obs},
            termination_ctx=term_ctx(clean=clean_verified),
        )

    if gate and gate_status in _SHUTDOWN_IN_FLIGHT and not clean_verified:
        return _finish_decision(
            STARTUP_RECOVERY_REVIEW,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=False),
        )

    if live_pids and not clean_verified:
        reasons.append("managed_processes_still_alive")
        return _finish_decision(
            STARTUP_RECOVERY_REQUIRED,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=False),
        )

    if live_pids and clean_verified:
        reasons.append("live_processes_after_verified_shutdown")
        return _finish_decision(
            STARTUP_RECOVERY_REVIEW,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=True),
        )

    if "launcher_session_mismatch_ledger_vs_managed" in reasons:
        return _finish_decision(
            STARTUP_RECOVERY_REVIEW,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=clean_verified),
        )

    if not clean_verified:
        reasons.append("clean_shutdown_not_verified")
        return _finish_decision(
            STARTUP_RECOVERY_REQUIRED,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=False),
        )

    if registry_corrupt or snapshot_ptr_corrupt:
        return _finish_decision(
            STARTUP_RECOVERY_REVIEW,
            dev_operations_allowed=False,
            previous_launcher_session=previous_session or None,
            reasons=reasons,
            checks=checks,
            termination_ctx=term_ctx(clean=clean_verified),
        )

    reasons.append("previous_session_clean_shutdown_verified")
    return _finish_decision(
        STARTUP_NORMAL_START,
        dev_operations_allowed=True,
        previous_launcher_session=previous_session or None,
        reasons=reasons,
        checks=checks,
        termination_ctx=term_ctx(clean=True),
    )


def startup_recovery_decision_to_dict(decision: StartupRecoveryDecision) -> Dict[str, Any]:
    return {
        "outcome": decision.outcome,
        "recoveryStatus": decision.recovery_status,
        "terminationType": decision.termination_type,
        "confidence": decision.confidence,
        "evidence": list(decision.evidence),
        "componentFindings": list(decision.component_findings),
        "devOperationsAllowed": decision.dev_operations_allowed,
        "previousLauncherSession": decision.previous_launcher_session,
        "reasons": list(decision.reasons),
        "checks": decision.checks,
    }


def assert_dev_operations_allowed(decision: StartupRecoveryDecision) -> None:
    if not decision.dev_operations_allowed:
        code = decision.outcome
        raise RuntimeError(f"ace_local_startup_recovery_blocked:{code}")


def local_dev_startup_gate_readonly_paths(ace_root: Path) -> List[Path]:
    """Paths touched by read-only startup evaluation (for test hashing)."""
    ace_root = Path(ace_root)
    paths = [
        recovery_ledger_path(ace_root),
        resolve_local_dev_state_dir(ace_root) / "managed-processes.json",
        shutdown_gate_state_path(ace_root),
        resolve_local_dev_state_dir(ace_root)
        / "shutdown-snapshots"
        / LAST_VERIFIED_SHUTDOWN_SNAPSHOT,
        resolve_local_dev_state_dir(ace_root) / "termination-evidence.json",
    ]
    return [p for p in paths if p.is_file()]
