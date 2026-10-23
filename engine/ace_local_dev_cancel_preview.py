"""
Local cancel-development Preview — invokes PowerShell engine read-only; no mutations.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.ace_local_dev_preview_fingerprint import compute_preview_fingerprint

logger = logging.getLogger(__name__)

_SESSION_ID_PATTERN = re.compile(r"^ori-\d{8}-\d{6}-[0-9a-f]{8}$", re.IGNORECASE)
_SENSITIVE_KEY_RE = re.compile(
    r"(secret|password|token|apikey|api_key|authorization|dpapi|plainsha|credential)",
    re.IGNORECASE,
)


def resolve_configured_dev_session_id() -> Optional[str]:
    raw = (os.environ.get("ACE_ORI_DEV_SESSION_ID") or "").strip()
    if not raw:
        return None
    if not _SESSION_ID_PATTERN.match(raw):
        raise ValueError("ace_ori_dev_session_id_invalid")
    return raw


def _backend_scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def run_cancel_dev_preview_powershell(session_id: str) -> Dict[str, Any]:
    script = _backend_scripts_dir() / "Export-OriCancelDevPreviewJson.ps1"
    if not script.is_file():
        raise FileNotFoundError("ace_cancel_preview_script_missing")

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-SessionId",
        session_id,
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(err or "ace_cancel_preview_powershell_failed")

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ace_cancel_preview_invalid_json") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ace_cancel_preview_invalid_payload")
    return data


def _strip_sensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, val in obj.items():
            if _SENSITIVE_KEY_RE.search(str(key)):
                continue
            out[key] = _strip_sensitive(val)
        return out
    if isinstance(obj, list):
        return [_strip_sensitive(x) for x in obj]
    return obj


def _commit_list(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "shortSha": item.get("shortSha"),
                "subject": item.get("subject"),
                "classification": item.get("classification"),
                "branches": item.get("branches") or [],
                "preservedInWipRef": bool(item.get("preservedInWipRef")),
                "preservedInBackup": bool(item.get("preservedInBackup")),
                "explicitlyIncluded": item.get("explicitlyIncluded"),
            }
        )
    return rows


def build_public_preview_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    plan = raw.get("plan") if isinstance(raw.get("plan"), dict) else {}
    plan = _strip_sensitive(plan)

    cancel_cycle = plan.get("cancelCycle") if isinstance(plan.get("cancelCycle"), dict) else {}
    cycle_status = str(cancel_cycle.get("status") or "UNKNOWN")
    session_id = str(plan.get("sessionId") or "")

    git = plan.get("git") if isinstance(plan.get("git"), dict) else {}
    be_git = git.get("backend") if isinstance(git.get("backend"), dict) else {}
    fe_git = git.get("frontend") if isinstance(git.get("frontend"), dict) else {}

    paths_be = list(be_git.get("PathsToRevert") or [])
    paths_fe = list(fe_git.get("PathsToRevert") or [])
    commits_be = _commit_list((plan.get("commitsToCancel") or {}).get("backend"))
    commits_fe = _commit_list((plan.get("commitsToCancel") or {}).get("frontend"))

    pre_warnings: List[Dict[str, Any]] = []
    graph = plan.get("commitGraph") if isinstance(plan.get("commitGraph"), dict) else {}
    for label in ("backend", "frontend"):
        block = graph.get(label) if isinstance(graph.get(label), dict) else {}
        for pre in block.get("PreBaselineCommits") or []:
            if not isinstance(pre, dict):
                continue
            pre_warnings.append(
                {
                    "repo": label.upper(),
                    "shortSha": pre.get("shortSha"),
                    "subject": pre.get("subject"),
                    "explicitlyIncluded": bool(pre.get("explicitlyIncluded")),
                    "requiresExplicitScope": not bool(pre.get("explicitlyIncluded")),
                }
            )

    blockers = [str(b) for b in (plan.get("blockers") or [])]
    can_execute = bool(plan.get("canExecute"))
    has_work = bool(
        paths_be
        or paths_fe
        or commits_be
        or commits_fe
        or (be_git.get("CurrentDirtyCount") or 0) > 0
        or (fe_git.get("CurrentDirtyCount") or 0) > 0
    )

    infra_paths = {
        "backend": list((plan.get("infraAdvancementPaths") or {}).get("backend") or []),
        "frontend": list((plan.get("infraAdvancementPaths") or {}).get("frontend") or []),
    }
    infra_commits = {
        "backend": _commit_list((plan.get("infraCommitsPreserved") or {}).get("backend")),
        "frontend": _commit_list((plan.get("infraCommitsPreserved") or {}).get("frontend")),
    }

    env_summary = {}
    env_block = plan.get("env") if isinstance(plan.get("env"), dict) else {}
    for side in ("backend", "frontend"):
        side_env = env_block.get(side) if isinstance(env_block.get(side), dict) else {}
        env_summary[side] = {
            "addedKeys": list(side_env.get("AddedKeys") or []),
            "removedKeys": list(side_env.get("RemovedKeys") or []),
            "changedKeys": list(side_env.get("ChangedKeys") or []),
            "preservedKeys": list(side_env.get("PreservedKeys") or []),
            "canApply": bool(side_env.get("CanApply", True)),
        }

    preview_fp = compute_preview_fingerprint(raw)

    return {
        "ok": True,
        "previewMode": "READ_ONLY",
        "previewFingerprint": preview_fp,
        "executeEndpointAvailable": True,
        "activeCycle": {
            "sessionId": session_id,
            "status": cycle_status,
            "hasWorkToCancel": has_work,
            "canExecuteInEngine": can_execute,
            "executeAvailableInApi": True,
        },
        "blockers": blockers,
        "registryInfraDrift": list(plan.get("registryInfraDrift") or []),
        "commitsToCancel": {"backend": commits_be, "frontend": commits_fe},
        "pathsToRevert": {"backend": paths_be, "frontend": paths_fe},
        "infraPreserved": {"commits": infra_commits, "paths": infra_paths},
        "envChanges": env_summary,
        "preBaselineWarnings": pre_warnings,
        "backupExpectations": {
            "fullRepoBackupBeforeReset": True,
            "includesAllRefsBundle": True,
            "includesEnvDpapiBackup": True,
            "includesWorkingTreeMirror": True,
            "recoveryGuideFile": "RECOVERY.md",
            "note": "Preview only; no backup created by this API call.",
        },
        "infraTargets": plan.get("infraTargets"),
        "willCancelSummary": plan.get("willCancel"),
    }


def no_active_cycle_response(*, reason: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "previewMode": "READ_ONLY",
        "executeEndpointAvailable": False,
        "activeCycle": {
            "sessionId": None,
            "status": "NONE",
            "hasWorkToCancel": False,
            "canExecuteInEngine": False,
            "executeAvailableInApi": False,
            "message": reason,
        },
        "blockers": [],
        "commitsToCancel": {"backend": [], "frontend": []},
        "pathsToRevert": {"backend": [], "frontend": []},
        "infraPreserved": {"commits": {"backend": [], "frontend": []}, "paths": {"backend": [], "frontend": []}},
        "preBaselineWarnings": [],
        "backupExpectations": {
            "fullRepoBackupBeforeReset": True,
            "includesAllRefsBundle": True,
            "includesEnvDpapiBackup": True,
            "includesWorkingTreeMirror": True,
            "recoveryGuideFile": "RECOVERY.md",
            "note": "Configure ACE_ORI_DEV_SESSION_ID for an active READY baseline session.",
        },
    }


def preview_error_response(*, error: str, message: str, blockers: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "message": message,
        "previewMode": "READ_ONLY",
        "executeEndpointAvailable": False,
        "blockers": blockers or [],
    }
