"""Temp checkpoint chain fixtures for recovery engine tests."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from engine.ace_local_dev_recovery_checkpoint_validate import recovery_checkpoints_root

FileSpec = Tuple[str, str, str]  # area, relpath, content or "" for delete marker


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_meta(
    root: Path,
    *,
    checkpoint_id: str,
    checkpoint_type: str,
    base_id: str = "",
    chain_root_id: str = "",
) -> None:
    verified_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "schemaVersion": 1,
        "checkpointId": checkpoint_id,
        "checkpointType": checkpoint_type,
        "baseCheckpointId": base_id or None,
        "chainRootId": chain_root_id or checkpoint_id,
        "status": "VERIFIED",
        "verifiedAt": verified_at,
        "finishedAt": verified_at,
    }
    (root / "checkpoint-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for label in ("BACKEND", "FRONTEND"):
        git_dir = root / label / "git"
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "HEAD.txt").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git_dir / "branch.txt").write_text("main\n", encoding="utf-8")


def _apply_env_dpapi(root: Path, logical_name: str, blob: bytes) -> None:
    env_dir = root / "ENV"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / f"{logical_name}.dpapi").write_bytes(blob)


def _apply_files_to_checkpoint(root: Path, files: Iterable[FileSpec], env_blobs: Optional[Dict[str, bytes]] = None) -> None:
    rows = ["Area,RelPath,ChangeType,BackupSha256"]
    env_blobs = env_blobs or {}
    for logical, blob in env_blobs.items():
        _apply_env_dpapi(root, logical, blob)
        digest = hashlib.sha256(blob).hexdigest()
        rows.append(f"ENV,{logical},env-dpapi,{digest}")
    for area, rel, content in files:
        area = area.upper()
        rel_norm = rel.replace("/", "\\")
        if content == "__DELETED__":
            deleted = root / area / "deleted-from-HEAD" / rel_norm
            deleted.parent.mkdir(parents=True, exist_ok=True)
            deleted.write_text("marker\n", encoding="utf-8")
            rows.append(f"{area},{rel},deleted,")
            continue
        mirror = root / area / "mirror" / rel_norm
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_bytes(content.encode("utf-8"))
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rows.append(f"{area},{rel},modified,{digest}")
    (root / "MANIFEST.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_full_checkpoint(
    ace_root: Path,
    checkpoint_id: str,
    files: Iterable[FileSpec],
    *,
    env_blobs: Optional[Dict[str, bytes]] = None,
) -> Path:
    root = recovery_checkpoints_root(ace_root) / checkpoint_id
    root.mkdir(parents=True, exist_ok=True)
    _write_meta(root, checkpoint_id=checkpoint_id, checkpoint_type="full")
    _apply_files_to_checkpoint(root, files, env_blobs=env_blobs)
    return root


def write_light_checkpoint(
    ace_root: Path,
    checkpoint_id: str,
    base_id: str,
    chain_root_id: str,
    delta_files: Iterable[FileSpec],
) -> Path:
    root = recovery_checkpoints_root(ace_root) / checkpoint_id
    root.mkdir(parents=True, exist_ok=True)
    _write_meta(
        root,
        checkpoint_id=checkpoint_id,
        checkpoint_type="light",
        base_id=base_id,
        chain_root_id=chain_root_id,
    )
    _apply_files_to_checkpoint(root, delta_files)
    return root


def merged_file_hash(full_files: Dict[str, str], deltas: Iterable[FileSpec]) -> Dict[str, str]:
    """Key rel path -> expected content after chain (BACKEND only helper)."""
    state = dict(full_files)
    for area, rel, content in deltas:
        if area.upper() != "BACKEND":
            continue
        if content == "__DELETED__":
            state.pop(rel, None)
        else:
            state[rel] = content
    return state
