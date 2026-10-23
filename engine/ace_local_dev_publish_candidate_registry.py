"""
Local publish candidate registry — separate from production-registry.json.

Stored under ACE _registry; not deployed to production.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "publish-candidate-registry.json"
SCHEMA_VERSION = 1
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_STATUS_ACTIVE = "ACTIVE"


def resolve_ace_root() -> Path:
    override = (os.environ.get("ACE_ORI_ACE_ROOT") or "").strip()
    if override:
        return Path(override)
    backend_root = Path(__file__).resolve().parent.parent
    return backend_root.parent


def publish_candidate_registry_path() -> Path:
    reg_override = (os.environ.get("ACE_PUBLISH_CANDIDATE_REGISTRY_PATH") or "").strip()
    if reg_override:
        return Path(reg_override)
    return resolve_ace_root() / "_registry" / REGISTRY_FILENAME


def publish_candidate_registry_sidecar_path() -> Path:
    return Path(str(publish_candidate_registry_path()) + ".sha256")


def normalize_full_sha(value: str) -> Optional[str]:
    token = (value or "").strip().lower()
    if not _FULL_SHA_RE.match(token):
        return None
    return token


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_registry_atomic(payload: Dict[str, Any]) -> None:
    path = publish_candidate_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    sidecar = publish_candidate_registry_sidecar_path()
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sidecar.write_text(f"{digest}  {REGISTRY_FILENAME}\n", encoding="ascii")


def load_publish_candidate_registry() -> Optional[Dict[str, Any]]:
    return _read_json(publish_candidate_registry_path())


def production_registry_path() -> Path:
    override = (os.environ.get("ACE_ORI_PS_REGISTRY_PATH") or "").strip()
    if override:
        return Path(override)
    return resolve_ace_root() / "_registry" / "production-registry.json"


def load_production_backend_baseline_sha() -> Optional[str]:
    data = _read_json(production_registry_path())
    if not data:
        return None
    prod = data.get("production") if isinstance(data.get("production"), dict) else {}
    web = prod.get("backendWeb") if isinstance(prod.get("backendWeb"), dict) else {}
    sha = normalize_full_sha(str(web.get("deployedCommitSha") or ""))
    return sha


def load_registry_wip_builder1_sha() -> Optional[str]:
    data = _read_json(production_registry_path())
    if not data:
        return None
    wip = data.get("wip") if isinstance(data.get("wip"), dict) else {}
    b1 = wip.get("builder1") if isinstance(wip.get("builder1"), dict) else {}
    return normalize_full_sha(str(b1.get("commitSha") or ""))


def git_commit_exists(backend_root: Path, sha: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
            cwd=str(backend_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def git_is_ancestor(backend_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=str(backend_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_backend_root() -> Path:
    override = (os.environ.get("ACE_ORI_PS_BACKEND_ROOT") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent


def build_registry_record(
    *,
    session_id: str,
    candidate_backend_sha: str,
    production_backend_baseline_sha: str,
    candidate_frontend_sha: Optional[str] = None,
    preview_fingerprint: str = "",
) -> Dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sessionId": session_id,
        "candidateBackendSha": candidate_backend_sha,
        "productionBackendBaselineSha": production_backend_baseline_sha,
        "candidateFrontendSha": candidate_frontend_sha or None,
        "registeredAt": _utc_now_iso(),
        "status": _STATUS_ACTIVE,
        "publishUriPreviewFingerprintAtRegistration": preview_fingerprint or None,
        "note": "Local-only; not production deploy state.",
    }


def save_active_publish_candidate(record: Dict[str, Any]) -> None:
    _write_registry_atomic(record)


def evaluate_registered_candidate(
    *,
    record: Dict[str, Any],
    active_session_id: str,
    cycle_status: str,
    current_production_baseline_sha: str,
    backend_root: Optional[Path] = None,
) -> Tuple[str, List[str]]:
    """
    Returns (effective_status, reasons). effective_status ACTIVE only when fully valid.
    """
    reasons: List[str] = []
    reg_session = str(record.get("sessionId") or "").strip()
    candidate = normalize_full_sha(str(record.get("candidateBackendSha") or ""))
    stored_baseline = normalize_full_sha(str(record.get("productionBackendBaselineSha") or ""))
    stored_status = str(record.get("status") or "").strip().upper()

    if stored_status and stored_status != _STATUS_ACTIVE:
        reasons.append(f"REGISTRY_STATUS:{stored_status}")
        return "STALE", reasons

    if not candidate:
        reasons.append("REGISTRY_CANDIDATE_SHA_INVALID")
        return "STALE", reasons

    if reg_session.lower() != (active_session_id or "").strip().lower():
        reasons.append("SESSION_MISMATCH")
        return "STALE", reasons

    if cycle_status != "OPEN":
        reasons.append(f"CYCLE_NOT_OPEN:{cycle_status}")
        return "STALE", reasons

    current_base = normalize_full_sha(current_production_baseline_sha or "")
    if not current_base or stored_baseline != current_base:
        reasons.append("PRODUCTION_BASELINE_CHANGED")
        return "STALE", reasons

    root = backend_root or resolve_backend_root()
    if not git_commit_exists(root, candidate):
        reasons.append("CANDIDATE_COMMIT_MISSING")
        return "STALE", reasons

    if stored_baseline and not git_is_ancestor(root, stored_baseline, candidate):
        reasons.append("CANDIDATE_NOT_DESCENDANT_OF_BASELINE")
        return "STALE", reasons

    return "ACTIVE", reasons
