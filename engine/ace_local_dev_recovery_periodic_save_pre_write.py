"""
Pre-checkpoint gate for recovery periodic save (phase 5.7) — coordinator + live guard.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_recovery_periodic_save_guard import assess_recovery_periodic_save_checkpoint
from engine.ace_local_dev_shutdown_gate import resolve_local_dev_state_dir

GUARD_OVERRIDE_FILENAME = "recovery-periodic-save-guard.json"
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_START_ALLOWED,
)
from engine.ace_local_dev_recovery_startup_gate import (
    LATEST_REPORT_FILENAME,
    startup_decision_reports_dir,
)
from engine.ace_local_dev_startup_recovery_check import read_recovery_ledger

ACTIVATION_SNAPSHOT_FILENAME = "recovery-periodic-save-activation-snapshot.json"
REASSESS_MAX_AGE_SECONDS = 5
ALLOWED_COORDINATOR_OUTCOMES = frozenset(
    {COORDINATOR_START_ALLOWED, COORDINATOR_AUTO_RECOVERY_ALLOWED}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def activation_snapshot_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / ACTIVATION_SNAPSHOT_FILENAME


def read_activation_snapshot(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = activation_snapshot_path(ace_root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def read_latest_startup_gate_report(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = startup_decision_reports_dir(ace_root) / LATEST_REPORT_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def evaluate_coordinator_periodic_save_gate(
    ace_root: Path,
    launcher_session: str,
) -> Dict[str, Any]:
    session = (launcher_session or "").strip()
    reasons: List[str] = []
    source = "none"
    outcome = ""
    allow_periodic = False

    report = read_latest_startup_gate_report(ace_root)
    if report:
        source = "startup_gate_report"
        decision = report.get("coordinatorDecision") or {}
        outcome = str(decision.get("coordinatorOutcome") or "")
        allow_periodic = bool(decision.get("allowPeriodicSaveService"))
        contract = decision.get("periodicSaveContract") or {}
        contract_session = str(contract.get("launcherSession") or "").strip()
        if outcome not in ALLOWED_COORDINATOR_OUTCOMES:
            reasons.append(f"coordinator_outcome:{outcome or 'empty'}")
        if not allow_periodic:
            reasons.append("periodic_save_not_allowed")
        if contract_session and contract_session != session:
            reasons.append("coordinator_launcher_session_mismatch")
    else:
        snap = read_activation_snapshot(ace_root)
        if snap:
            source = "activation_snapshot"
            outcome = str(snap.get("coordinatorOutcome") or "")
            allow_periodic = bool(snap.get("allowPeriodicSaveService"))
            snap_session = str(snap.get("launcherSession") or "")
            if outcome not in ALLOWED_COORDINATOR_OUTCOMES:
                reasons.append(f"coordinator_outcome:{outcome or 'empty'}")
            if not allow_periodic:
                reasons.append("periodic_save_not_allowed")
            if snap_session and session and snap_session != session:
                reasons.append("snapshot_launcher_session_mismatch")
        else:
            reasons.append("coordinator_evidence_missing")

    if not session:
        reasons.append("launcher_session_empty")

    return {
        "allowed": not reasons,
        "reasons": reasons,
        "source": source,
        "coordinatorOutcome": outcome,
        "allowPeriodicSaveService": allow_periodic,
    }


def evaluate_ledger_launcher_session_match(
    ace_root: Path,
    launcher_session: str,
) -> Dict[str, Any]:
    session = (launcher_session or "").strip()
    ledger, corrupt = read_recovery_ledger(ace_root)
    if corrupt:
        return {"allowed": False, "reasons": ["ledger_corrupt"], "ledgerSession": ""}
    if not ledger:
        return {"allowed": False, "reasons": ["ledger_missing"], "ledgerSession": ""}
    ledger_session = str(ledger.get("launcherSession") or "").strip()
    if not session or not ledger_session or ledger_session != session:
        return {
            "allowed": False,
            "reasons": ["ledger_launcher_session_mismatch"],
            "ledgerSession": ledger_session,
        }
    return {"allowed": True, "reasons": [], "ledgerSession": ledger_session}


def _read_guard_file_override(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = resolve_local_dev_state_dir(ace_root) / GUARD_OVERRIDE_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("blockCheckpoint"):
        return {
            "allowed": False,
            "reason": str(data.get("reason") or "guard_file_block"),
            "source": "guard_file",
        }
    if data.get("allowCheckpoint"):
        return {
            "allowed": True,
            "reason": "",
            "source": "guard_file",
        }
    return None


def evaluate_pre_checkpoint_write_gate(
    ace_root: Path,
    launcher_session: str,
    *,
    live_assessment_only: bool = False,
    prior_assessment_at_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined gate before checkpoint write. When prior_assessment_at_iso is set and older than
    REASSESS_MAX_AGE_SECONDS, live assessment is always refreshed.
    """
    ace_root = Path(ace_root)
    reasons: List[str] = []
    coordinator: Dict[str, Any] = {}
    ledger: Dict[str, Any] = {}
    live: Dict[str, Any] = {}

    if not live_assessment_only:
        ledger = evaluate_ledger_launcher_session_match(ace_root, launcher_session)
        if not ledger.get("allowed"):
            reasons.extend(list(ledger.get("reasons") or []))
        coordinator = evaluate_coordinator_periodic_save_gate(ace_root, launcher_session)
        if not coordinator.get("allowed"):
            reasons.extend(list(coordinator.get("reasons") or []))

    need_fresh_live = True
    if prior_assessment_at_iso:
        try:
            prior = datetime.fromisoformat(prior_assessment_at_iso.replace("Z", "+00:00"))
            if prior.tzinfo is None:
                prior = prior.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - prior.astimezone(timezone.utc)).total_seconds()
            if age <= REASSESS_MAX_AGE_SECONDS:
                need_fresh_live = False
        except (TypeError, ValueError):
            need_fresh_live = True

    if need_fresh_live or live_assessment_only:
        guard_live = _read_guard_file_override(ace_root)
        if guard_live is not None:
            live = guard_live
        else:
            live = assess_recovery_periodic_save_checkpoint(ace_root=ace_root)
        if not live.get("allowed"):
            reasons.append(str(live.get("reason") or "live_assessment_blocked"))
    else:
        live = {"allowed": True, "reason": "", "source": "prior_assessment_still_fresh"}

    return {
        "allowed": not reasons,
        "reasons": reasons,
        "assessedAt": _utc_now_iso(),
        "ledgerGate": ledger,
        "coordinatorGate": coordinator,
        "liveAssessment": live,
        "reassessMaxAgeSeconds": REASSESS_MAX_AGE_SECONDS,
    }


def _cli_main(argv: Optional[List[str]] = None) -> int:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    ace_text = ""
    launcher = ""
    live_only = False
    prior_at = ""
    for idx, token in enumerate(args):
        if token == "--ace-root" and idx + 1 < len(args):
            ace_text = args[idx + 1]
        if token == "--launcher-session" and idx + 1 < len(args):
            launcher = args[idx + 1]
        if token == "--live-only":
            live_only = True
        if token == "--prior-assessment-at" and idx + 1 < len(args):
            prior_at = args[idx + 1]
    if not ace_text:
        sys.stderr.write("missing --ace-root\n")
        return 1
    payload = evaluate_pre_checkpoint_write_gate(
        Path(ace_text),
        launcher,
        live_assessment_only=live_only,
        prior_assessment_at_iso=prior_at or None,
    )
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return 0 if payload.get("allowed") else 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
