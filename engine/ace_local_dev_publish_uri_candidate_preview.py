"""
Publish Uri Candidate Preview — explicit backend SHA from production baseline; read-only.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.ace_local_dev_powershell import run_powershell_script_file
from engine.ace_local_dev_publish_uri_fingerprint import (
    compute_publish_uri_candidate_preview_fingerprint,
)
from engine.ace_local_dev_publish_uri_preview import (
    _commit_rows,
    _strip_sensitive,
    publish_uri_no_active_cycle_response,
    publish_uri_preview_error_response,
)

logger = logging.getLogger(__name__)

_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def normalize_candidate_backend_sha(value: str) -> Optional[str]:
    token = (value or "").strip().lower()
    if not _FULL_SHA_RE.match(token):
        return None
    return token


def run_publish_uri_candidate_preview_powershell(
    session_id: str, candidate_backend_sha: str
) -> Dict[str, Any]:
    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "Export-OriPublishUriCandidatePreviewJson.ps1"
    )
    if not script.is_file():
        raise FileNotFoundError("ace_publish_uri_candidate_preview_script_missing")

    returncode, stdout, stderr = run_powershell_script_file(
        script,
        ["-SessionId", session_id, "-CandidateBackendSha", candidate_backend_sha],
        timeout=120,
        env=os.environ.copy(),
    )
    if returncode != 0:
        err = (stderr or stdout or "").strip()
        raise RuntimeError(err or "ace_publish_uri_candidate_preview_powershell_failed")

    text = stdout.strip()
    if not text:
        raise RuntimeError("ace_publish_uri_candidate_preview_powershell_empty_stdout")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ace_publish_uri_candidate_preview_invalid_json") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ace_publish_uri_candidate_preview_invalid_payload")
    return data


def build_public_publish_uri_candidate_preview_response(raw: Dict[str, Any]) -> Dict[str, Any]:
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

    candidate_blockers = [str(b) for b in (plan.get("candidateBlockers") or plan.get("blockers") or [])]
    warnings = [str(w) for w in (plan.get("warnings") or [])]
    mixed = _commit_rows(plan.get("mixedCommits"))

    fingerprint = compute_publish_uri_candidate_preview_fingerprint(raw)

    return {
        "ok": True,
        "previewMode": "READ_ONLY",
        "previewKind": "CANDIDATE_BACKEND",
        "publishUriPreviewFingerprint": fingerprint,
        "candidateBackendSha": str(plan.get("candidateBackendSha") or be_prop.get("proposedHeadSha") or ""),
        "executePublishEndpointAvailable": False,
        "executePublishRouteAvailable": True,
        "executeProgressRouteAvailable": True,
        "prepareSimulateEndpointAvailable": True,
        "candidateDryRunAvailable": True,
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
        "candidateBlockers": candidate_blockers,
        "blockers": candidate_blockers,
        "warnings": warnings,
        "registryInfraDrift": list(plan.get("registryInfraDrift") or []),
        "registryWipDrift": list(plan.get("registryWipDrift") or []),
        "readiness": {
            "hasProductChanges": bool(readiness.get("hasProductChanges")),
            "hasBackendProductChanges": bool(readiness.get("hasBackendProductChanges")),
            "hasFrontendProductChanges": bool(readiness.get("hasFrontendProductChanges")),
            "candidateVerifiedSafeForPublish": bool(
                readiness.get("candidateVerifiedSafeForPublish")
            ),
            "verifiedSafeForLiveProduction": bool(
                readiness.get("verifiedSafeForLiveProduction")
            ),
            "canPrepareLocalPublish": bool(readiness.get("canPrepareLocalPublish")),
            "candidateDryRunAvailable": bool(readiness.get("candidateDryRunAvailable", True)),
            "candidatePrepareSimulateAvailable": bool(readiness.get("canPrepareLocalPublish")),
            "notes": list(readiness.get("notes") or []),
        },
        "activeCycle": {
            "status": str(cancel_cycle.get("status") or "UNKNOWN"),
        },
        "note": "Candidate preview only; no push, deploy, or production registry update.",
    }


def publish_uri_candidate_invalid_sha_response() -> Dict[str, Any]:
    return publish_uri_preview_error_response(
        error="ace_publish_uri_candidate_sha_invalid",
        message="candidateBackendSha must be exactly 40 lowercase hexadecimal characters.",
    )


__all__ = [
    "build_public_publish_uri_candidate_preview_response",
    "normalize_candidate_backend_sha",
    "publish_uri_candidate_invalid_sha_response",
    "publish_uri_no_active_cycle_response",
    "publish_uri_preview_error_response",
    "run_publish_uri_candidate_preview_powershell",
]
