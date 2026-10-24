"""Automatic file recovery plan — Full/Light chain merge (no blind restore)."""
from __future__ import annotations

import csv
import hashlib
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.ace_local_dev_recovery_checkpoint_chain import (
    EffectiveFileState,
    merge_chain_effective_state,
    resolve_restorable_checkpoint_chain,
)

ACTION_SKIP_UNCHANGED = "SKIP_UNCHANGED"
ACTION_RESTORE_FROM_CHECKPOINT = "RESTORE_FROM_CHECKPOINT"
ACTION_PRESERVE_BOTH = "PRESERVE_BOTH"
ACTION_PRESERVE_WORKSPACE_FILE = "PRESERVE_WORKSPACE_FILE"
ACTION_DELETE_IF_SAFE = "DELETE_IF_SAFE"
ACTION_BLOCK_UNCERTAIN = "BLOCK_UNCERTAIN"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_workspace_path(area: str, rel: str, backend_root: Path, frontend_root: Path) -> Optional[Path]:
    rel_norm = rel.replace("/", "\\")
    if area == "BACKEND":
        return backend_root / rel_norm
    if area == "FRONTEND":
        return frontend_root / rel_norm
    if area == "ENV":
        if rel_norm == "backend.env.local":
            return backend_root / ".env.local"
        if rel_norm == "frontend.env.development.local":
            return frontend_root / ".env.development.local"
        return backend_root / rel_norm
    return None


def _resolve_checkpoint_backup(state: EffectiveFileState) -> Optional[Path]:
    rel_norm = state.rel_path.replace("/", "\\")
    root = state.source_checkpoint_root
    area = state.area.upper()
    change = state.change_type.lower()
    if area in ("BACKEND", "FRONTEND"):
        if change == "deleted":
            return root / area / "deleted-from-HEAD" / rel_norm
        return root / area / "mirror" / rel_norm
    if area == "ENV" and change == "env-dpapi":
        return root / "ENV" / f"{state.rel_path}.dpapi"
    return None


@dataclass
class RestorePlanEntry:
    area: str
    rel_path: str
    action: str
    reason: str
    workspace_sha256: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    source_checkpoint_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area": self.area,
            "relPath": self.rel_path,
            "action": self.action,
            "reason": self.reason,
            "workspaceSha256": self.workspace_sha256,
            "checkpointSha256": (
                "redacted"
                if self.area.upper() == "ENV" and self.checkpoint_sha256
                else self.checkpoint_sha256
            ),
            "dpapiBlobSha256Prefix": (
                (self.checkpoint_sha256 or "")[:12]
                if self.area.upper() == "ENV" and self.checkpoint_sha256
                else ""
            ),
            "sourceCheckpointId": self.source_checkpoint_id,
        }


@dataclass
class AutoRestorePlan:
    checkpoint_id: str
    checkpoint_root: str
    checkpoint_valid: bool
    chain_ids: List[str] = field(default_factory=list)
    entries: List[RestorePlanEntry] = field(default_factory=list)
    safe_auto_apply: bool = False
    has_conflicts: bool = False
    safety_backup_dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    chain_resolution: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpointId": self.checkpoint_id,
            "checkpointRoot": self.checkpoint_root,
            "checkpointValid": self.checkpoint_valid,
            "chainIds": list(self.chain_ids),
            "entries": [e.to_dict() for e in self.entries],
            "safeAutoApply": self.safe_auto_apply,
            "hasConflicts": self.has_conflicts,
            "safetyBackupDir": self.safety_backup_dir,
            "notes": list(self.notes),
            "chainResolution": self.chain_resolution,
        }


def _plan_entry_from_state(
    state: EffectiveFileState,
    *,
    backend_root: Path,
    frontend_root: Path,
) -> RestorePlanEntry:
    backup_path = _resolve_checkpoint_backup(state)
    workspace_path = _resolve_workspace_path(state.area, state.rel_path, backend_root, frontend_root)
    entry = RestorePlanEntry(
        area=state.area,
        rel_path=state.rel_path,
        action=ACTION_BLOCK_UNCERTAIN,
        reason="unclassified",
        source_checkpoint_id=state.source_checkpoint_id,
    )

    if state.area.upper() == "ENV" and state.change_type == "env-dpapi":
        if workspace_path is not None and workspace_path.is_file():
            entry.action = ACTION_PRESERVE_BOTH
            entry.reason = "env_present_no_auto_plaintext_restore"
            return entry
        if backup_path is None or not backup_path.is_file():
            entry.action = ACTION_BLOCK_UNCERTAIN
            entry.reason = "env_dpapi_missing_in_checkpoint"
            return entry
        entry.action = ACTION_RESTORE_FROM_CHECKPOINT
        entry.reason = "env_missing_dpapi_restore"
        if state.sha256:
            entry.checkpoint_sha256 = state.sha256
        return entry

    if state.change_type == "deleted":
        if workspace_path is None or not workspace_path.is_file():
            entry.action = ACTION_SKIP_UNCHANGED
            entry.reason = "already_deleted_or_absent"
            return entry
        entry.workspace_sha256 = _sha256_file(workspace_path)
        if state.sha256 and entry.workspace_sha256 == state.sha256:
            entry.action = ACTION_DELETE_IF_SAFE
            entry.reason = "matches_pre_delete_checkpoint_content"
        else:
            entry.action = ACTION_PRESERVE_WORKSPACE_FILE
            entry.reason = "delete_not_proven_safe"
        return entry

    if backup_path is None or not backup_path.is_file():
        entry.action = ACTION_BLOCK_UNCERTAIN
        entry.reason = "checkpoint_backup_missing"
        return entry

    entry.checkpoint_sha256 = state.sha256 or _sha256_file(backup_path)
    if workspace_path is None:
        entry.action = ACTION_BLOCK_UNCERTAIN
        entry.reason = "workspace_path_unknown"
        return entry

    if not workspace_path.is_file():
        entry.action = ACTION_RESTORE_FROM_CHECKPOINT
        entry.reason = "workspace_file_missing"
        return entry

    ws_hash = _sha256_file(workspace_path)
    entry.workspace_sha256 = ws_hash
    if ws_hash == entry.checkpoint_sha256:
        entry.action = ACTION_SKIP_UNCHANGED
        entry.reason = "already_matches_chain_state"
    else:
        entry.action = ACTION_PRESERVE_BOTH
        entry.reason = "workspace_differs_no_overwrite"
    return entry


def build_auto_restore_plan(
    *,
    ace_root: Path,
    checkpoint_id: str,
    backend_root: Path,
    frontend_root: Path,
) -> AutoRestorePlan:
    resolution = resolve_restorable_checkpoint_chain(
        ace_root,
        preferred_checkpoint_id=checkpoint_id,
    )
    tip_id = checkpoint_id
    tip_root = ""
    if resolution.ok and resolution.chain:
        tip = resolution.chain[-1]
        tip_id = tip.checkpoint_id
        tip_root = str(tip.checkpoint_root)

    plan = AutoRestorePlan(
        checkpoint_id=tip_id,
        checkpoint_root=tip_root,
        checkpoint_valid=resolution.ok,
        chain_resolution=resolution.to_dict(),
    )
    if not resolution.ok:
        plan.notes.append("chain_not_restorable")
        plan.has_conflicts = True
        plan.safe_auto_apply = False
        return plan

    plan.chain_ids = [link.checkpoint_id for link in resolution.chain]
    effective = merge_chain_effective_state(resolution.chain)
    for state in effective.values():
        entry = _plan_entry_from_state(state, backend_root=backend_root, frontend_root=frontend_root)
        plan.entries.append(entry)
        if entry.action in (ACTION_PRESERVE_BOTH, ACTION_PRESERVE_WORKSPACE_FILE, ACTION_BLOCK_UNCERTAIN):
            plan.has_conflicts = True

    safe_actions = (
        ACTION_SKIP_UNCHANGED,
        ACTION_RESTORE_FROM_CHECKPOINT,
        ACTION_DELETE_IF_SAFE,
    )
    plan.safe_auto_apply = (
        plan.checkpoint_valid
        and not plan.has_conflicts
        and all(e.action in safe_actions for e in plan.entries)
    )
    return plan


def build_auto_restore_plan_legacy_single_checkpoint(
    *,
    checkpoint_root: Path,
    backend_root: Path,
    frontend_root: Path,
) -> AutoRestorePlan:
    """Backward-compatible wrapper when only a checkpoint root directory is known."""
    checkpoint_root = Path(checkpoint_root)
    meta_path = checkpoint_root / "checkpoint-metadata.json"
    cid = checkpoint_root.name
    ace_root = checkpoint_root.parent.parent.parent.parent
    if meta_path.is_file():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            cid = str(meta.get("checkpointId") or cid)
            ace_text = str(meta.get("aceRoot") or "")
            if ace_text:
                ace_root = Path(ace_text)
        except (OSError, json.JSONDecodeError):
            pass
    return build_auto_restore_plan(
        ace_root=ace_root,
        checkpoint_id=cid,
        backend_root=backend_root,
        frontend_root=frontend_root,
    )


def filter_plan_for_isolated_operations(plan: AutoRestorePlan) -> AutoRestorePlan:
    """Partial restore: only missing-file recovery; never overwrite or delete."""
    filtered = AutoRestorePlan(
        checkpoint_id=plan.checkpoint_id,
        checkpoint_root=plan.checkpoint_root,
        checkpoint_valid=plan.checkpoint_valid,
        chain_ids=list(plan.chain_ids),
        chain_resolution=dict(plan.chain_resolution),
        notes=list(plan.notes) + ["isolated_operations_partial_restore"],
    )
    for entry in plan.entries:
        if entry.action == ACTION_RESTORE_FROM_CHECKPOINT:
            filtered.entries.append(entry)
        elif entry.action == ACTION_SKIP_UNCHANGED:
            filtered.entries.append(entry)
        else:
            clone = RestorePlanEntry(
                area=entry.area,
                rel_path=entry.rel_path,
                action=ACTION_PRESERVE_BOTH,
                reason=f"isolated_skip:{entry.reason}",
                workspace_sha256=entry.workspace_sha256,
                checkpoint_sha256=entry.checkpoint_sha256,
                source_checkpoint_id=entry.source_checkpoint_id,
            )
            filtered.entries.append(clone)
            filtered.has_conflicts = True
    safe_actions = (ACTION_SKIP_UNCHANGED, ACTION_RESTORE_FROM_CHECKPOINT)
    filtered.safe_auto_apply = filtered.checkpoint_valid and all(
        e.action in safe_actions for e in filtered.entries
    )
    return filtered


def apply_auto_restore_plan_in_test_sandbox(
    plan: AutoRestorePlan,
    *,
    backend_root: Path,
    frontend_root: Path,
    safety_backup_parent: Path,
    entry_filter: Optional[Callable[[RestorePlanEntry], bool]] = None,
) -> Dict[str, Any]:
    if not plan.checkpoint_valid:
        raise RuntimeError("ace_auto_restore_invalid_checkpoint")
    backup_dir = safety_backup_parent / f"pre-restore-{uuid.uuid4().hex[:12]}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    plan.safety_backup_dir = str(backup_dir)

    applied: List[str] = []
    preserved: List[str] = []
    deleted: List[str] = []

    for entry in plan.entries:
        if entry_filter and not entry_filter(entry):
            preserved.append(entry.rel_path)
            continue
        if entry.action == ACTION_SKIP_UNCHANGED:
            continue
        if entry.action in (ACTION_PRESERVE_BOTH, ACTION_PRESERVE_WORKSPACE_FILE, ACTION_BLOCK_UNCERTAIN):
            preserved.append(entry.rel_path)
            continue
        if entry.action == ACTION_DELETE_IF_SAFE:
            target = _resolve_workspace_path(entry.area, entry.rel_path, backend_root, frontend_root)
            if target and target.is_file():
                rel_backup = backup_dir / entry.area / entry.rel_path.replace("/", "_")
                rel_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, rel_backup)
                target.unlink()
                deleted.append(entry.rel_path)
            continue
        if entry.action != ACTION_RESTORE_FROM_CHECKPOINT:
            preserved.append(entry.rel_path)
            continue

        state = EffectiveFileState(
            area=entry.area,
            rel_path=entry.rel_path,
            change_type="modified",
            sha256=entry.checkpoint_sha256 or "",
            source_checkpoint_id=entry.source_checkpoint_id,
            source_checkpoint_root=Path(plan.checkpoint_root),
        )
        for link_root in _chain_roots_from_plan(plan):
            candidate = link_root / "checkpoint-metadata.json"
            if not candidate.is_file():
                continue
            try:
                import json

                meta = json.loads(candidate.read_text(encoding="utf-8-sig"))
                if str(meta.get("checkpointId") or "") == entry.source_checkpoint_id:
                    state = EffectiveFileState(
                        area=entry.area,
                        rel_path=entry.rel_path,
                        change_type="modified",
                        sha256=entry.checkpoint_sha256 or "",
                        source_checkpoint_id=entry.source_checkpoint_id,
                        source_checkpoint_root=link_root,
                    )
                    break
            except (OSError, json.JSONDecodeError):
                continue

        backup_path = _resolve_checkpoint_backup(state)
        target = _resolve_workspace_path(entry.area, entry.rel_path, backend_root, frontend_root)
        if backup_path is None or target is None or not backup_path.is_file():
            preserved.append(entry.rel_path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        rel_backup = backup_dir / entry.area / entry.rel_path.replace("/", "_")
        rel_backup.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            shutil.copy2(target, rel_backup)
        shutil.copy2(backup_path, target)
        applied.append(entry.rel_path)

    return {
        "applied": applied,
        "preserved": preserved,
        "deleted": deleted,
        "safetyBackupDir": plan.safety_backup_dir,
        "appliedAt": datetime.now(timezone.utc).isoformat(),
    }


def _chain_roots_from_plan(plan: AutoRestorePlan) -> List[Path]:
    roots: List[Path] = []
    for link in plan.chain_resolution.get("chain") or []:
        if isinstance(link, dict) and link.get("checkpointRoot"):
            roots.append(Path(str(link["checkpointRoot"])))
    if plan.checkpoint_root:
        roots.append(Path(plan.checkpoint_root))
    return roots
