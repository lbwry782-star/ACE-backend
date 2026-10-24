"""Read-only recovery checkpoint validation (no restore, no delete)."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR_NAME = "recovery-checkpoints"
METADATA_FILENAME = "checkpoint-metadata.json"
MANIFEST_FILENAME = "MANIFEST.csv"


def recovery_checkpoints_root(ace_root: Path) -> Path:
    return ace_root / "_backups" / "local-dev" / CHECKPOINTS_DIR_NAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metadata(checkpoint_root: Path) -> Optional[Dict[str, Any]]:
    path = checkpoint_root / METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def list_verified_checkpoint_records(ace_root: Path) -> List[Dict[str, Any]]:
    root = recovery_checkpoints_root(ace_root)
    if not root.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        meta = _read_metadata(entry)
        if not meta:
            continue
        if str(meta.get("status") or "").upper() != "VERIFIED":
            continue
        rows.append(
            {
                "checkpointId": str(meta.get("checkpointId") or entry.name),
                "checkpointRoot": str(entry),
                "verifiedAt": str(meta.get("verifiedAt") or ""),
                "checkpointType": str(meta.get("checkpointType") or "").lower(),
                "baseCheckpointId": str(meta.get("baseCheckpointId") or "") or None,
                "chainRootId": str(meta.get("chainRootId") or "") or None,
            }
        )
    rows.sort(key=lambda r: str(r.get("verifiedAt") or ""))
    return rows


@dataclass
class CheckpointValidationResult:
    checkpoint_id: str
    checkpoint_root: str
    ok: bool
    status: str
    errors: List[str] = field(default_factory=list)


def validate_recovery_checkpoint_root(checkpoint_root: Path) -> CheckpointValidationResult:
    checkpoint_root = Path(checkpoint_root)
    meta = _read_metadata(checkpoint_root)
    cid = str((meta or {}).get("checkpointId") or checkpoint_root.name)
    if not meta:
        return CheckpointValidationResult(
            checkpoint_id=cid,
            checkpoint_root=str(checkpoint_root),
            ok=False,
            status="MISSING_METADATA",
            errors=["CHECKPOINT_METADATA_MISSING"],
        )
    status = str(meta.get("status") or "").upper()
    errors: List[str] = []
    if status != "VERIFIED":
        errors.append(f"CHECKPOINT_STATUS_{status or 'UNKNOWN'}")
    manifest_path = checkpoint_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        errors.append("MANIFEST_MISSING")
    else:
        try:
            with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                errors.append("MANIFEST_EMPTY")
            for row in rows:
                area = str(row.get("Area") or "")
                rel = str(row.get("RelPath") or "")
                expected = str(row.get("BackupSha256") or "").strip().lower()
                if not area or not rel:
                    continue
                target: Optional[Path] = None
                change = str(row.get("ChangeType") or "")
                if area in ("BACKEND", "FRONTEND"):
                    if change == "deleted":
                        target = checkpoint_root / area / "deleted-from-HEAD" / rel.replace("/", "\\")
                    else:
                        target = checkpoint_root / area / "mirror" / rel.replace("/", "\\")
                elif area == "ENV" and change == "env-dpapi":
                    target = checkpoint_root / "ENV" / f"{rel}.dpapi"
                if target is None or not target.is_file():
                    if change not in ("env-missing",):
                        errors.append(f"MANIFEST_FILE_MISSING:{area}:{rel}")
                    continue
                if expected and expected != "redacted":
                    actual = _sha256_file(target)
                    if actual != expected:
                        errors.append(f"MANIFEST_HASH_MISMATCH:{area}:{rel}")
        except (OSError, csv.Error) as exc:
            errors.append(f"MANIFEST_READ_FAILED:{exc}")
    for label in ("BACKEND", "FRONTEND"):
        head_file = checkpoint_root / label / "git" / "HEAD.txt"
        if not head_file.is_file():
            errors.append(f"MISSING_{label}_HEAD")
    ok = len(errors) == 0 and status == "VERIFIED"
    return CheckpointValidationResult(
        checkpoint_id=cid,
        checkpoint_root=str(checkpoint_root),
        ok=ok,
        status=status if status else "UNKNOWN",
        errors=errors,
    )


@dataclass
class RecoveryCheckpointAssessment:
    verified_count: int
    last_ledger_checkpoint_id: Optional[str]
    recommended_checkpoint_id: Optional[str]
    last_validation: Optional[CheckpointValidationResult]
    recommended_validation: Optional[CheckpointValidationResult]
    prior_verified_available: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verifiedCount": self.verified_count,
            "lastLedgerCheckpointId": self.last_ledger_checkpoint_id,
            "recommendedCheckpointId": self.recommended_checkpoint_id,
            "lastValidationOk": bool(self.last_validation.ok) if self.last_validation else None,
            "recommendedValidationOk": bool(self.recommended_validation.ok)
            if self.recommended_validation
            else None,
            "priorVerifiedAvailable": self.prior_verified_available,
            "notes": list(self.notes),
            "lastValidationErrors": list(self.last_validation.errors) if self.last_validation else [],
            "recommendedValidationErrors": list(self.recommended_validation.errors)
            if self.recommended_validation
            else [],
        }


def assess_recovery_checkpoints(
    ace_root: Path,
    *,
    ledger: Optional[Dict[str, Any]] = None,
) -> RecoveryCheckpointAssessment:
    from engine.ace_local_dev_startup_recovery_check import read_recovery_ledger

    ace_root = Path(ace_root)
    if ledger is None:
        ledger, _corrupt = read_recovery_ledger(ace_root)
    verified = list_verified_checkpoint_records(ace_root)
    last_id: Optional[str] = None
    last_path: Optional[Path] = None
    if ledger and isinstance(ledger.get("lastVerifiedCheckpoint"), dict):
        lvc = ledger["lastVerifiedCheckpoint"]
        last_id = str(lvc.get("checkpointId") or "") or None
        cp_path = str(lvc.get("checkpointPath") or "").strip()
        if cp_path:
            last_path = Path(cp_path)
        elif last_id:
            last_path = recovery_checkpoints_root(ace_root) / last_id

    notes: List[str] = []
    last_val: Optional[CheckpointValidationResult] = None
    recommended_id: Optional[str] = None
    recommended_val: Optional[CheckpointValidationResult] = None

    if last_path and last_path.is_dir():
        last_val = validate_recovery_checkpoint_root(last_path)
        if last_val.ok:
            recommended_id = last_val.checkpoint_id
        else:
            notes.append("last_ledger_checkpoint_not_restorable")
            for row in reversed(verified):
                cid = str(row.get("checkpointId") or "")
                if cid and cid != last_id:
                    root = Path(str(row.get("checkpointRoot") or ""))
                    if root.is_dir():
                        candidate = validate_recovery_checkpoint_root(root)
                        if candidate.ok:
                            recommended_id = candidate.checkpoint_id
                            recommended_val = candidate
                            notes.append("fallback_to_prior_verified_checkpoint")
                            break
    elif last_id:
        notes.append("last_ledger_checkpoint_path_missing")

    prior_available = len(verified) > 1 or (
        len(verified) == 1 and (not last_id or str(verified[0].get("checkpointId")) != last_id)
    )

    return RecoveryCheckpointAssessment(
        verified_count=len(verified),
        last_ledger_checkpoint_id=last_id,
        recommended_checkpoint_id=recommended_id,
        last_validation=last_val,
        recommended_validation=recommended_val,
        prior_verified_available=prior_available,
        notes=notes,
    )
