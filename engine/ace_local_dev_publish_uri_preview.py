"""
Publish Uri Preview — read-only; invokes PowerShell plan engine. No push or deploy.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_powershell import run_powershell_script_file
from engine.ace_local_dev_publish_uri_fingerprint import compute_publish_uri_preview_fingerprint

logger = logging.getLogger(__name__)

_SENSITIVE_KEY_RE = re.compile(
    r"(secret|password|token|apikey|api_key|authorization|dpapi|plainsha|credential)",
    re.IGNORECASE,
)


def _backend_scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def run_publish_uri_preview_powershell(session_id: str) -> Dict[str, Any]:
    script = _backend_scripts_dir() / "Export-OriPublishUriPreviewJson.ps1"
    if not script.is_file():
        raise FileNotFoundError("ace_publish_uri_preview_script_missing")

    returncode, stdout, stderr = run_powershell_script_file(
        script,
        ["-SessionId", session_id],
        timeout=120,
        env=os.environ.copy(),
    )
    if returncode != 0:
        err = (stderr or stdout or "").strip()
        raise RuntimeError(err or "ace_publish_uri_preview_powershell_failed")

    text = stdout.strip()
    if not text:
        raise RuntimeError("ace_publish_uri_preview_powershell_empty_stdout")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ace_publish_uri_preview_invalid_json") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ace_publish_uri_preview_invalid_payload")
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


def _commit_rows(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        paths = item.get("paths") or []
        path_names = []
        if isinstance(paths, list):
            for p in paths:
                if isinstance(p, dict) and p.get("path"):
                    path_names.append(str(p["path"]))
                elif isinstance(p, str):
                    path_names.append(p)
        rows.append(
            {
                "shortSha": item.get("shortSha"),
                "subject": item.get("subject"),
                "classification": item.get("classification"),
                "branches": item.get("branches") or [],
                "paths": path_names,
                "securityFlags": list(item.get("securityFlags") or []),
            }
        )
    return rows


def build_public_publish_uri_preview_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    plan = raw.get("plan") if isinstance(raw.get("plan"), dict) else {}
    plan = _strip_sensitive(plan)

    production = plan.get("production") if isinstance(plan.get("production"), dict) else {}
    proposed = plan.get("proposedChanges") if isinstance(plan.get("proposedChanges"), dict) else {}
    be_prop = proposed.get("backend") if isinstance(proposed.get("backend"), dict) else {}
    fe_prop = proposed.get("frontend") if isinstance(proposed.get("frontend"), dict) else {}
    commits = plan.get("commits") if isinstance(plan.get("commits"), dict) else {}
    cand = commits.get("productCandidate") if isinstance(commits.get("productCandidate"), dict) else {}
    excl = commits.get("excludedLocal") if isinstance(commits.get("excludedLocal"), dict) else {}
    readiness = plan.get("readiness") if isinstance(plan.get("readiness"), dict) else {}
    worker = plan.get("workerImpact") if isinstance(plan.get("workerImpact"), dict) else {}
    cancel_cycle = plan.get("cancelCycle") if isinstance(plan.get("cancelCycle"), dict) else {}

    blockers = [str(b) for b in (plan.get("blockers") or [])]
    mixed = _commit_rows(plan.get("mixedCommits"))

    fingerprint = compute_publish_uri_preview_fingerprint(raw)

    return {
        "ok": True,
        "previewMode": "READ_ONLY",
        "previewKind": "WIP_DEFAULT",
        "publishUriPreviewFingerprint": fingerprint,
        "executePublishEndpointAvailable": False,
        "sessionId": str(plan.get("sessionId") or ""),
        "production": {
            "backendWeb": production.get("backendWeb"),
            "backendWorker": production.get("backendWorker"),
            "backendBranch": production.get("backendBranch"),
            "frontend": production.get("frontend"),
            "frontendBranch": production.get("frontendBranch"),
        },
        "proposedChanges": {
            "backend": {
                "deployTargetBranch": be_prop.get("deployTargetBranch"),
                "sourceRef": be_prop.get("sourceRef"),
                "proposedHeadSha": be_prop.get("proposedHeadSha"),
                "commits": _commit_rows(be_prop.get("commits")),
                "files": list(be_prop.get("files") or []),
            },
            "frontend": {
                "deployTargetBranch": fe_prop.get("deployTargetBranch"),
                "sourceRef": fe_prop.get("sourceRef"),
                "proposedHeadSha": fe_prop.get("proposedHeadSha"),
                "commits": _commit_rows(fe_prop.get("commits")),
                "files": list(fe_prop.get("files") or []),
            },
        },
        "workerImpact": {
            "deployWithBackendWeb": bool(worker.get("deployWithBackendWeb")),
            "proposedCommitSha": worker.get("proposedCommitSha"),
            "note": worker.get("note"),
        },
        "commits": {
            "productCandidate": {
                "backend": _commit_rows(cand.get("backend")),
                "frontend": _commit_rows(cand.get("frontend")),
            },
            "excludedLocal": {
                "backend": _commit_rows(excl.get("backend")),
                "frontend": _commit_rows(excl.get("frontend")),
            },
            "mixedOrBlocked": mixed,
        },
        "blockers": blockers,
        "registryInfraDrift": list(plan.get("registryInfraDrift") or []),
        "registryWipDrift": list(plan.get("registryWipDrift") or []),
        "readiness": {
            "hasProductChanges": bool(readiness.get("hasProductChanges")),
            "canPrepareLocalPublish": bool(readiness.get("canPrepareLocalPublish")),
            "verifiedSafeForLiveProduction": bool(readiness.get("verifiedSafeForLiveProduction")),
            "notes": list(readiness.get("notes") or []),
        },
        "activeCycle": {
            "status": str(cancel_cycle.get("status") or "UNKNOWN"),
        },
        "note": "Preview only; no push, deploy, or production registry update.",
    }


def publish_uri_no_active_cycle_response(*, reason: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "previewMode": "READ_ONLY",
        "executePublishEndpointAvailable": False,
        "sessionId": None,
        "blockers": [],
        "readiness": {
            "hasProductChanges": False,
            "canPrepareLocalPublish": False,
            "verifiedSafeForLiveProduction": False,
        },
        "activeCycle": {"status": "NONE", "message": reason},
    }


def publish_uri_preview_error_response(
    *, error: str, message: str, blockers: Optional[List[str]] = None
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "message": message,
        "previewMode": "READ_ONLY",
        "executePublishEndpointAvailable": False,
        "blockers": blockers or [],
    }
