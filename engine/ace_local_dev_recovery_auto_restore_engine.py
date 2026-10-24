"""
Safe automatic recovery engine (phase 5.2 — not wired to Start-AceLocal).

Runs only when coordinator allows; writes journal before mutating workspace files.
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_recovery_auto_restore_plan import (
    ACTION_BLOCK_UNCERTAIN,
    ACTION_DELETE_IF_SAFE,
    ACTION_PRESERVE_BOTH,
    ACTION_PRESERVE_WORKSPACE_FILE,
    ACTION_RESTORE_FROM_CHECKPOINT,
    ACTION_SKIP_UNCHANGED,
    AutoRestorePlan,
    RestorePlanEntry,
    build_auto_restore_plan,
    filter_plan_for_isolated_operations,
    _resolve_checkpoint_backup,
    _resolve_workspace_path,
)
from engine.ace_local_dev_recovery_checkpoint_chain import EffectiveFileState
from engine.ace_local_dev_recovery_startup_coordinator import (
    COORDINATOR_AUTO_RECOVERY_ALLOWED,
    COORDINATOR_RECOVERY_UNSAFE,
    COORDINATOR_START_ALLOWED,
    COORDINATOR_START_BLOCKED,
    COORDINATOR_START_WITH_ISOLATED_OPERATIONS,
    RecoveryStartupCoordinatorDecision,
)
from engine.ace_local_dev_recovery_env_dpapi import restore_missing_env_from_checkpoint
from engine.ace_local_dev_recovery_restore_completeness import (
    snapshot_pre_restore_state,
    verify_restore_completeness,
)
from engine.ace_local_dev_shutdown_gate import (
    _write_json_atomic,
    resolve_local_dev_state_dir,
)

logger = logging.getLogger(__name__)

JOURNAL_FILENAME = "recovery-auto-restore-journal.json"
JOURNAL_STATUS_IN_PROGRESS = "IN_PROGRESS"
JOURNAL_STATUS_COMPLETED = "COMPLETED"
JOURNAL_STATUS_FAILED = "FAILED"
JOURNAL_STATUS_INTERRUPTED = "INTERRUPTED"
JOURNAL_STATUS_SKIPPED = "SKIPPED"

ENTRY_PENDING = "pending"
ENTRY_DONE = "done"
ENTRY_SKIPPED = "skipped"
ENTRY_CONFLICT = "conflict"
ENTRY_FAILED = "failed"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def journal_path(ace_root: Path) -> Path:
    return resolve_local_dev_state_dir(ace_root) / JOURNAL_FILENAME


def read_restore_journal(ace_root: Path) -> Optional[Dict[str, Any]]:
    path = journal_path(ace_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_journal(ace_root: Path, payload: Dict[str, Any]) -> None:
    payload["updatedAt"] = _utc_now_iso()
    _write_json_atomic(journal_path(ace_root), payload)


def _journal_entries_from_plan(plan: AutoRestorePlan) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in plan.entries:
        rows.append(
            {
                "area": entry.area,
                "relPath": entry.rel_path,
                "plannedAction": entry.action,
                "status": ENTRY_PENDING,
                "reason": entry.reason,
                "sourceCheckpointId": entry.source_checkpoint_id,
            }
        )
    return rows


@dataclass
class AutoRecoveryEngineResult:
    ran: bool
    skipped: bool
    skip_reason: str = ""
    coordinator_outcome: str = ""
    journal_status: str = ""
    journal_id: str = ""
    applied: List[str] = field(default_factory=list)
    preserved: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    safety_backup_dir: Optional[str] = None
    plan: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    env_restored: List[str] = field(default_factory=list)
    env_failed: List[str] = field(default_factory=list)
    partial_recovery: bool = False
    completeness: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": self.ran,
            "skipped": self.skipped,
            "skipReason": self.skip_reason,
            "coordinatorOutcome": self.coordinator_outcome,
            "journalStatus": self.journal_status,
            "journalId": self.journal_id,
            "applied": list(self.applied),
            "preserved": list(self.preserved),
            "deleted": list(self.deleted),
            "conflicts": list(self.conflicts),
            "safetyBackupDir": self.safety_backup_dir,
            "plan": self.plan,
            "notes": list(self.notes),
            "envRestored": list(self.env_restored),
            "envFailed": list(self.env_failed),
            "partialRecovery": self.partial_recovery,
            "completeness": self.completeness,
        }


def _should_run_recovery(decision: RecoveryStartupCoordinatorDecision) -> tuple[bool, str, bool]:
    outcome = decision.coordinator_outcome
    if outcome == COORDINATOR_START_ALLOWED:
        return False, "start_allowed_no_recovery", False
    if outcome == COORDINATOR_START_BLOCKED:
        return False, "start_blocked", False
    if outcome == COORDINATOR_RECOVERY_UNSAFE:
        return False, "recovery_unsafe", False
    if outcome == COORDINATOR_AUTO_RECOVERY_ALLOWED:
        return True, "", False
    if outcome == COORDINATOR_START_WITH_ISOLATED_OPERATIONS:
        return True, "", True
    return False, f"outcome_not_supported:{outcome}", False


def _resolve_checkpoint_id(decision: RecoveryStartupCoordinatorDecision) -> Optional[str]:
    assessment = decision.checkpoint_assessment or {}
    cid = str(assessment.get("recommendedCheckpointId") or "").strip()
    if cid:
        return cid
    lvc = decision.last_verified_checkpoint or {}
    return str(lvc.get("checkpointId") or "").strip() or None


def _backup_file(path: Path, backup_dir: Path, area: str, rel: str) -> None:
    dest = backup_dir / area / rel.replace("\\", "_").replace("/", "_")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        shutil.copy2(path, dest)


def _apply_entry(
    entry: RestorePlanEntry,
    plan: AutoRestorePlan,
    *,
    ace_root: Path,
    backend_root: Path,
    frontend_root: Path,
    backup_dir: Path,
    conflicts_dir: Path,
    decrypt_and_write: Optional[Any] = None,
) -> tuple[str, str]:
    action = entry.action
    if action == ACTION_SKIP_UNCHANGED:
        return ENTRY_SKIPPED, "unchanged"

    target = _resolve_workspace_path(entry.area, entry.rel_path, backend_root, frontend_root)
    if action in (ACTION_PRESERVE_BOTH, ACTION_PRESERVE_WORKSPACE_FILE, ACTION_BLOCK_UNCERTAIN):
        if target and target.is_file():
            conflict_copy = conflicts_dir / entry.area / entry.rel_path.replace("\\", "_")
            conflict_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, conflict_copy)
        return ENTRY_CONFLICT, entry.reason

    if action == ACTION_DELETE_IF_SAFE:
        if target and target.is_file():
            _backup_file(target, backup_dir, entry.area, entry.rel_path)
            target.unlink()
            return ENTRY_DONE, "deleted"
        return ENTRY_SKIPPED, "already_absent"

    if action == ACTION_RESTORE_FROM_CHECKPOINT and entry.area.upper() == "ENV":
        link_root = _find_source_root(plan, entry.source_checkpoint_id)
        cp_root = link_root or Path(plan.checkpoint_root)
        state = EffectiveFileState(
            area=entry.area,
            rel_path=entry.rel_path,
            change_type="env-dpapi",
            sha256=entry.checkpoint_sha256 or "",
            source_checkpoint_id=entry.source_checkpoint_id,
            source_checkpoint_root=cp_root,
        )
        dpapi_src = _resolve_checkpoint_backup(state)
        if dpapi_src is None or not dpapi_src.is_file():
            return ENTRY_FAILED, "env_dpapi_missing"
        meta: Dict[str, Any] = {}
        meta_path = cp_root / "checkpoint-metadata.json"
        if meta_path.is_file():
            try:
                import json as _json

                loaded = _json.loads(meta_path.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (OSError, ValueError):
                pass
        attempt = restore_missing_env_from_checkpoint(
            ace_root=ace_root,
            backend_root=backend_root,
            frontend_root=frontend_root,
            logical_name=entry.rel_path,
            dpapi_path=dpapi_src,
            expected_blob_sha256=entry.checkpoint_sha256 or "",
            checkpoint_root=cp_root,
            checkpoint_metadata=meta,
            decrypt_and_write=decrypt_and_write,
        )
        if attempt.ok:
            return ENTRY_DONE, "env_restored"
        return ENTRY_FAILED, attempt.detail

    if action == ACTION_RESTORE_FROM_CHECKPOINT:
        link_root = _find_source_root(plan, entry.source_checkpoint_id)
        state = EffectiveFileState(
            area=entry.area,
            rel_path=entry.rel_path,
            change_type="modified",
            sha256=entry.checkpoint_sha256 or "",
            source_checkpoint_id=entry.source_checkpoint_id,
            source_checkpoint_root=link_root or Path(plan.checkpoint_root),
        )
        backup_path = _resolve_checkpoint_backup(state)
        if backup_path is None or not backup_path.is_file() or target is None:
            return ENTRY_FAILED, "backup_missing"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            _backup_file(target, backup_dir, entry.area, entry.rel_path)
        tmp = target.with_suffix(target.suffix + f".ace-restore-{uuid.uuid4().hex[:8]}.tmp")
        shutil.copy2(backup_path, tmp)
        tmp.replace(target)
        return ENTRY_DONE, "restored"

    return ENTRY_FAILED, "unknown_action"


def _find_source_root(plan: AutoRestorePlan, checkpoint_id: str) -> Optional[Path]:
    for link in plan.chain_resolution.get("chain") or []:
        if isinstance(link, dict) and str(link.get("checkpointId") or "") == checkpoint_id:
            root = str(link.get("checkpointRoot") or "")
            if root:
                return Path(root)
    return Path(plan.checkpoint_root) if plan.checkpoint_root else None


def run_conditional_auto_recovery(
    ace_root: Path,
    decision: RecoveryStartupCoordinatorDecision,
    *,
    backend_root: Path,
    frontend_root: Path,
    resume_interrupted: bool = True,
    simulate_crash_after: int = 0,
    decrypt_and_write: Optional[Any] = None,
) -> AutoRecoveryEngineResult:
    """
    Execute safe auto-recovery for temp/isolated workspaces only when coordinator permits.
    `simulate_crash_after` applies N file mutations then leaves journal IN_PROGRESS (tests).
    """
    ace_root = Path(ace_root)
    should_run, skip_reason, isolated_partial = _should_run_recovery(decision)
    result = AutoRecoveryEngineResult(
        ran=False,
        skipped=not should_run,
        skip_reason=skip_reason,
        coordinator_outcome=decision.coordinator_outcome,
    )
    if not should_run:
        existing = read_restore_journal(ace_root)
        if existing and str(existing.get("status") or "") == JOURNAL_STATUS_IN_PROGRESS:
            result.notes.append("prior_journal_left_in_progress")
        result.journal_status = ""
        return result

    git_before, protected_before = snapshot_pre_restore_state(
        ace_root, backend_root, frontend_root
    )

    checkpoint_id = _resolve_checkpoint_id(decision)
    if not checkpoint_id:
        result.skipped = True
        result.skip_reason = "no_checkpoint_id"
        return result

    plan = build_auto_restore_plan(
        ace_root=ace_root,
        checkpoint_id=checkpoint_id,
        backend_root=backend_root,
        frontend_root=frontend_root,
    )
    if isolated_partial:
        plan = filter_plan_for_isolated_operations(plan)

    result.plan = plan.to_dict()
    if not plan.checkpoint_valid:
        result.skipped = True
        result.skip_reason = "invalid_checkpoint_chain"
        return result

    journal = read_restore_journal(ace_root)
    journal_entries = _journal_entries_from_plan(plan)
    journal_id = uuid.uuid4().hex
    resume = False

    if (
        resume_interrupted
        and journal
        and str(journal.get("status") or "") == JOURNAL_STATUS_IN_PROGRESS
        and str(journal.get("targetCheckpointId") or "") == plan.checkpoint_id
    ):
        resume = True
        journal_id = str(journal.get("journalId") or journal_id)
        journal_entries = list(journal.get("entries") or journal_entries)

    safety_parent = resolve_local_dev_state_dir(ace_root) / "recovery-restore-safety"
    safety_parent.mkdir(parents=True, exist_ok=True)
    backup_dir = safety_parent / f"backup-{journal_id}"
    conflicts_dir = safety_parent / f"conflicts-{journal_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    conflicts_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        _write_journal(
            ace_root,
            {
                "journalId": journal_id,
                "status": JOURNAL_STATUS_IN_PROGRESS,
                "coordinatorOutcome": decision.coordinator_outcome,
                "targetCheckpointId": plan.checkpoint_id,
                "chainIds": list(plan.chain_ids),
                "safetyBackupDir": str(backup_dir),
                "conflictsDir": str(conflicts_dir),
                "startedAt": _utc_now_iso(),
                "finishedAt": None,
                "entries": journal_entries,
                "previousLauncherSession": decision.previous_launcher_session,
                "evidencePreservation": dict(decision.evidence_preservation),
            },
        )

    applied: List[str] = []
    preserved: List[str] = []
    deleted: List[str] = []
    conflicts: List[str] = []
    mutations = 0

    entry_by_key = {(e.area, e.rel_path): e for e in plan.entries}

    for idx, row in enumerate(journal_entries):
        if str(row.get("status") or "") == ENTRY_DONE:
            continue
        key = (str(row.get("area") or ""), str(row.get("relPath") or ""))
        plan_entry = entry_by_key.get(key)
        if plan_entry is None:
            row["status"] = ENTRY_SKIPPED
            continue

        status, detail = _apply_entry(
            plan_entry,
            plan,
            ace_root=ace_root,
            backend_root=backend_root,
            frontend_root=frontend_root,
            backup_dir=backup_dir,
            conflicts_dir=conflicts_dir,
            decrypt_and_write=decrypt_and_write,
        )
        if plan_entry.area.upper() == "ENV" and isinstance(row, dict):
            row["envRestoreDetail"] = detail[:80] if detail else ""
        row["status"] = status
        row["detail"] = detail
        row["finishedAt"] = _utc_now_iso()

        rel = plan_entry.rel_path
        if status == ENTRY_DONE:
            mutations += 1
            if plan_entry.action == ACTION_DELETE_IF_SAFE:
                deleted.append(rel)
            elif plan_entry.action == ACTION_RESTORE_FROM_CHECKPOINT:
                if plan_entry.area.upper() == "ENV":
                    result.env_restored.append(plan_entry.rel_path)
                else:
                    applied.append(rel)
        elif status == ENTRY_FAILED and plan_entry.area.upper() == "ENV":
            result.env_failed.append(plan_entry.rel_path)
            result.partial_recovery = True
        elif status == ENTRY_CONFLICT:
            conflicts.append(rel)
            preserved.append(rel)
        elif status == ENTRY_SKIPPED:
            preserved.append(rel)
        else:
            preserved.append(rel)

        _write_journal(
            ace_root,
            {
                "journalId": journal_id,
                "status": JOURNAL_STATUS_IN_PROGRESS,
                "coordinatorOutcome": decision.coordinator_outcome,
                "targetCheckpointId": plan.checkpoint_id,
                "chainIds": list(plan.chain_ids),
                "safetyBackupDir": str(backup_dir),
                "conflictsDir": str(conflicts_dir),
                "startedAt": journal.get("startedAt") if resume and journal else _utc_now_iso(),
                "finishedAt": None,
                "entries": journal_entries,
                "previousLauncherSession": decision.previous_launcher_session,
                "evidencePreservation": dict(decision.evidence_preservation),
            },
        )

        if simulate_crash_after > 0 and mutations >= simulate_crash_after:
            result.ran = True
            result.journal_status = JOURNAL_STATUS_IN_PROGRESS
            result.journal_id = journal_id
            result.applied = applied
            result.preserved = preserved
            result.deleted = deleted
            result.conflicts = conflicts
            result.safety_backup_dir = str(backup_dir)
            result.notes.append("simulated_crash_mid_restore")
            return result

    final_status = JOURNAL_STATUS_COMPLETED
    if conflicts and not applied and not deleted:
        final_status = JOURNAL_STATUS_COMPLETED
    _write_journal(
        ace_root,
        {
            "journalId": journal_id,
            "status": final_status,
            "coordinatorOutcome": decision.coordinator_outcome,
            "targetCheckpointId": plan.checkpoint_id,
            "chainIds": list(plan.chain_ids),
            "safetyBackupDir": str(backup_dir),
            "conflictsDir": str(conflicts_dir),
            "startedAt": journal.get("startedAt") if resume and journal else _utc_now_iso(),
            "finishedAt": _utc_now_iso(),
            "entries": journal_entries,
            "previousLauncherSession": decision.previous_launcher_session,
            "evidencePreservation": dict(decision.evidence_preservation),
            "summary": {
                "appliedCount": len(applied),
                "deletedCount": len(deleted),
                "conflictCount": len(conflicts),
            },
        },
    )

    result.ran = True
    result.skipped = False
    result.journal_status = final_status
    result.journal_id = journal_id
    result.applied = applied
    result.preserved = preserved
    result.deleted = deleted
    result.conflicts = conflicts
    result.safety_backup_dir = str(backup_dir)

    completeness = verify_restore_completeness(
        ace_root=ace_root,
        backend_root=backend_root,
        frontend_root=frontend_root,
        git_before=git_before,
        protected_before=protected_before,
        plan=plan,
        engine_result=result.to_dict(),
    )
    result.completeness = completeness.to_dict()
    return result


def identify_interrupted_restore(ace_root: Path) -> Dict[str, Any]:
    journal = read_restore_journal(ace_root)
    if not journal:
        return {"interrupted": False}
    status = str(journal.get("status") or "")
    entries = list(journal.get("entries") or [])
    done = [e for e in entries if str(e.get("status") or "") == ENTRY_DONE]
    pending = [e for e in entries if str(e.get("status") or "") == ENTRY_PENDING]
    return {
        "interrupted": status == JOURNAL_STATUS_IN_PROGRESS,
        "journalId": journal.get("journalId"),
        "targetCheckpointId": journal.get("targetCheckpointId"),
        "completedEntries": done,
        "pendingEntries": pending,
        "status": status,
    }
