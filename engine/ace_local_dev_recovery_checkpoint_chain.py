"""Verified Full/Light checkpoint chain resolution and merged effective state."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_recovery_checkpoint_validate import (
    list_verified_checkpoint_records,
    recovery_checkpoints_root,
    validate_recovery_checkpoint_root,
    _read_metadata,
)

CHECKPOINT_TYPE_FULL = "full"
CHECKPOINT_TYPE_LIGHT = "light"


@dataclass
class CheckpointChainLink:
    checkpoint_id: str
    checkpoint_root: Path
    checkpoint_type: str
    base_checkpoint_id: Optional[str]
    verified_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpointId": self.checkpoint_id,
            "checkpointRoot": str(self.checkpoint_root),
            "checkpointType": self.checkpoint_type,
            "baseCheckpointId": self.base_checkpoint_id,
            "verifiedAt": self.verified_at,
        }


@dataclass
class CheckpointChainResolution:
    ok: bool
    chain: List[CheckpointChainLink] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fallback_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "chain": [link.to_dict() for link in self.chain],
            "errors": list(self.errors),
            "fallbackFrom": self.fallback_from,
            "chainIds": [link.checkpoint_id for link in self.chain],
        }


@dataclass
class EffectiveFileState:
    area: str
    rel_path: str
    change_type: str
    sha256: str
    source_checkpoint_id: str
    source_checkpoint_root: Path

    @property
    def key(self) -> Tuple[str, str]:
        return (self.area.upper(), self.rel_path.replace("/", "\\"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area": self.area,
            "relPath": self.rel_path,
            "changeType": self.change_type,
            "sha256": self.sha256,
            "sourceCheckpointId": self.source_checkpoint_id,
        }


def _verified_index(ace_root: Path) -> Dict[str, Dict[str, Any]]:
    return {str(r["checkpointId"]): r for r in list_verified_checkpoint_records(ace_root)}


def build_verified_checkpoint_chain(
    ace_root: Path,
    checkpoint_id: str,
) -> CheckpointChainResolution:
    ace_root = Path(ace_root)
    index = _verified_index(ace_root)
    if checkpoint_id not in index:
        return CheckpointChainResolution(ok=False, errors=["CHECKPOINT_NOT_VERIFIED"])

    chain: List[CheckpointChainLink] = []
    cur = checkpoint_id
    guard = 0
    while cur and guard < 64:
        guard += 1
        row = index.get(cur)
        if not row:
            return CheckpointChainResolution(
                ok=False,
                chain=chain,
                errors=[f"CHAIN_BROKEN_MISSING:{cur}"],
            )
        root = Path(str(row["checkpointRoot"]))
        validation = validate_recovery_checkpoint_root(root)
        if not validation.ok:
            return CheckpointChainResolution(
                ok=False,
                chain=chain,
                errors=[f"CHAIN_LINK_INVALID:{cur}"] + list(validation.errors),
            )
        meta = _read_metadata(root) or {}
        link = CheckpointChainLink(
            checkpoint_id=cur,
            checkpoint_root=root,
            checkpoint_type=str(row.get("checkpointType") or meta.get("checkpointType") or "").lower(),
            base_checkpoint_id=str(row.get("baseCheckpointId") or meta.get("baseCheckpointId") or "") or None,
            verified_at=str(row.get("verifiedAt") or ""),
        )
        chain.insert(0, link)
        cur = link.base_checkpoint_id or ""

    if not chain:
        return CheckpointChainResolution(ok=False, errors=["CHAIN_EMPTY"])
    if chain[0].checkpoint_type != CHECKPOINT_TYPE_FULL:
        return CheckpointChainResolution(
            ok=False,
            chain=chain,
            errors=["CHAIN_ROOT_NOT_FULL"],
        )
    return CheckpointChainResolution(ok=True, chain=chain)


def resolve_restorable_checkpoint_chain(
    ace_root: Path,
    *,
    preferred_checkpoint_id: Optional[str] = None,
) -> CheckpointChainResolution:
    ace_root = Path(ace_root)
    verified = list_verified_checkpoint_records(ace_root)
    candidates: List[str] = []
    if preferred_checkpoint_id:
        candidates.append(preferred_checkpoint_id)
    for row in reversed(verified):
        cid = str(row.get("checkpointId") or "")
        if cid and cid not in candidates:
            candidates.append(cid)

    tried = preferred_checkpoint_id
    for cid in candidates:
        resolution = build_verified_checkpoint_chain(ace_root, cid)
        if resolution.ok:
            if tried and cid != tried:
                resolution.fallback_from = tried
            return resolution
    return CheckpointChainResolution(
        ok=False,
        errors=["NO_RESTORABLE_CHAIN"],
        fallback_from=preferred_checkpoint_id,
    )


def _read_manifest_rows(checkpoint_root: Path) -> List[Dict[str, str]]:
    path = checkpoint_root / "MANIFEST.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_chain_effective_state(chain: List[CheckpointChainLink]) -> Dict[Tuple[str, str], EffectiveFileState]:
    state: Dict[Tuple[str, str], EffectiveFileState] = {}
    for link in chain:
        rows = _read_manifest_rows(link.checkpoint_root)
        if link.checkpoint_type == CHECKPOINT_TYPE_FULL:
            state = {}
        for row in rows:
            area = str(row.get("Area") or "").upper()
            rel = str(row.get("RelPath") or "")
            change = str(row.get("ChangeType") or "modified").lower()
            sha = str(row.get("BackupSha256") or "").strip().lower()
            if not area or not rel:
                continue
            if change in ("env-missing",):
                continue
            key = (area, rel.replace("/", "\\"))
            if change == "deleted":
                prev = state.get(key)
                prev_sha = prev.sha256 if prev else sha
                state[key] = EffectiveFileState(
                    area=area,
                    rel_path=rel,
                    change_type="deleted",
                    sha256=prev_sha,
                    source_checkpoint_id=link.checkpoint_id,
                    source_checkpoint_root=link.checkpoint_root,
                )
                continue
            state[key] = EffectiveFileState(
                area=area,
                rel_path=rel,
                change_type=change,
                sha256=sha,
                source_checkpoint_id=link.checkpoint_id,
                source_checkpoint_root=link.checkpoint_root,
            )
    return state
