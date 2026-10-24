"""
ENV recovery via existing Windows DPAPI (Read-AceLocalShutdownEnvPlain / Protect-ContentForCurrentUser).

No plaintext in logs, journals, or reports. Encrypted blob SHA256 only for integrity.
"""
from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_LOGICAL_PATHS = frozenset(
    {
        "backend.env.local",
        "frontend.env.development.local",
    }
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def resolve_env_target_path(
    logical_name: str,
    *,
    backend_root: Path,
    frontend_root: Path,
) -> Optional[Path]:
    name = (logical_name or "").strip().lower()
    if name == "backend.env.local":
        return backend_root / ".env.local"
    if name == "frontend.env.development.local":
        return frontend_root / ".env.development.local"
    return None


def validate_env_dpapi_blob(
    dpapi_path: Path,
    *,
    expected_blob_sha256: str = "",
) -> Dict[str, Any]:
    dpapi_path = Path(dpapi_path)
    if not dpapi_path.is_file():
        return {"ok": False, "errors": ["DPAPI_FILE_MISSING"], "blobSha256": ""}
    try:
        blob = dpapi_path.read_bytes()
    except OSError as exc:
        return {"ok": False, "errors": [f"DPAPI_READ_FAILED:{exc}"], "blobSha256": ""}
    if len(blob) < 16:
        return {"ok": False, "errors": ["DPAPI_BLOB_TOO_SHORT"], "blobSha256": _sha256_bytes(blob)}
    digest = _sha256_bytes(blob)
    errors: List[str] = []
    expected = (expected_blob_sha256 or "").strip().lower()
    if expected and digest != expected:
        errors.append("DPAPI_BLOB_SHA256_MISMATCH")
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "blobSha256": digest,
        "sizeBytes": len(blob),
    }


def validate_env_restore_context(
    *,
    ace_root: Path,
    backend_root: Path,
    frontend_root: Path,
    logical_name: str,
    checkpoint_root: Path,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ace_root = Path(ace_root).resolve()
    backend_root = Path(backend_root).resolve()
    frontend_root = Path(frontend_root).resolve()
    if logical_name not in ENV_LOGICAL_PATHS:
        reasons.append("ENV_LOGICAL_NAME_UNKNOWN")
    try:
        backend_root.relative_to(ace_root)
        frontend_root.relative_to(ace_root)
    except ValueError:
        reasons.append("WORKSPACE_OUTSIDE_ACE_ROOT")
    meta = checkpoint_metadata or {}
    meta_ace = str(meta.get("aceRoot") or "").strip()
    if meta_ace:
        try:
            if Path(meta_ace).resolve() != ace_root:
                reasons.append("CHECKPOINT_ACE_ROOT_MISMATCH")
        except OSError:
            reasons.append("CHECKPOINT_ACE_ROOT_INVALID")
    cp_root = Path(checkpoint_root).resolve()
    if not cp_root.is_dir():
        reasons.append("CHECKPOINT_ROOT_MISSING")
    return len(reasons) == 0, reasons


def _recovery_env_ps1_path() -> Path:
    backend_root = Path(__file__).resolve().parent.parent
    return backend_root / "scripts" / "AceLocalRecoveryEnvRestoreLib.ps1"


def restore_env_from_dpapi_powershell(
    *,
    dpapi_path: Path,
    target_path: Path,
) -> Tuple[bool, str]:
    if platform.system().lower() != "windows":
        return False, "env_restore_requires_windows"
    script = _recovery_env_ps1_path()
    if not script.is_file():
        return False, "env_restore_script_missing"
    dpapi_path = Path(dpapi_path)
    target_path = Path(target_path)
    if target_path.is_file():
        return False, "env_target_exists"
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-DpapiPath",
        str(dpapi_path.resolve()),
        "-TargetPath",
        str(target_path.resolve()),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ACE_RECOVERY_ENV_RESTORE_INVOKE_FAILED code=%s", type(exc).__name__)
        return False, "env_restore_invoke_failed"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace").strip()
        safe = err.split("\n")[0][:120] if err else "env_restore_failed"
        logger.warning("ACE_RECOVERY_ENV_RESTORE_FAILED rc=%s hint=%s", proc.returncode, safe)
        return False, "env_dpapi_decrypt_or_write_failed"
    if not target_path.is_file():
        return False, "env_restore_verify_missing"
    return True, "env_restored"


@dataclass
class EnvRestoreAttempt:
    logical_name: str
    ok: bool
    detail: str
    blob_sha256: str = ""
    target_path: str = ""

    def journal_row(self) -> Dict[str, Any]:
        return {
            "logicalName": self.logical_name,
            "ok": self.ok,
            "detail": self.detail,
            "blobSha256Prefix": (self.blob_sha256 or "")[:12],
            "targetPath": self.target_path,
        }


def restore_missing_env_from_checkpoint(
    *,
    ace_root: Path,
    backend_root: Path,
    frontend_root: Path,
    logical_name: str,
    dpapi_path: Path,
    expected_blob_sha256: str,
    checkpoint_root: Path,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
    decrypt_and_write: Optional[Callable[[Path, Path], Tuple[bool, str]]] = None,
) -> EnvRestoreAttempt:
    target = resolve_env_target_path(logical_name, backend_root=backend_root, frontend_root=frontend_root)
    if target is None:
        return EnvRestoreAttempt(logical_name, False, "invalid_target_mapping")
    if target.is_file():
        return EnvRestoreAttempt(logical_name, False, "env_exists_no_overwrite", target_path=str(target))

    ctx_ok, ctx_errors = validate_env_restore_context(
        ace_root=ace_root,
        backend_root=backend_root,
        frontend_root=frontend_root,
        logical_name=logical_name,
        checkpoint_root=checkpoint_root,
        checkpoint_metadata=checkpoint_metadata,
    )
    if not ctx_ok:
        return EnvRestoreAttempt(
            logical_name,
            False,
            "context_invalid:" + ",".join(ctx_errors[:3]),
            target_path=str(target),
        )

    validation = validate_env_dpapi_blob(dpapi_path, expected_blob_sha256=expected_blob_sha256)
    if not validation.get("ok"):
        return EnvRestoreAttempt(
            logical_name,
            False,
            "dpapi_integrity_failed",
            blob_sha256=str(validation.get("blobSha256") or ""),
            target_path=str(target),
        )

    writer = decrypt_and_write or restore_env_from_dpapi_powershell
    ok, detail = writer(dpapi_path=dpapi_path, target_path=target)
    return EnvRestoreAttempt(
        logical_name,
        ok,
        detail,
        blob_sha256=str(validation.get("blobSha256") or ""),
        target_path=str(target),
    )
