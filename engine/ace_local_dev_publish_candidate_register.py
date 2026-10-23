"""Register a publish candidate after full Candidate Preview validation."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from engine.ace_local_dev_cycle_store import resolve_active_dev_session_id_for_api
from engine.ace_local_dev_publish_candidate_registry import (
    build_registry_record,
    git_commit_exists,
    git_is_ancestor,
    load_production_backend_baseline_sha,
    load_registry_wip_builder1_sha,
    normalize_full_sha,
    resolve_backend_root,
    save_active_publish_candidate,
)
from engine.ace_local_dev_publish_uri_candidate_preview import (
    build_public_publish_uri_candidate_preview_response,
    run_publish_uri_candidate_preview_powershell,
)
from engine.ace_local_dev_publish_uri_fingerprint import (
    compute_publish_uri_candidate_preview_fingerprint,
)

logger = logging.getLogger(__name__)

CONFIRM_REGISTER_TOKEN = "REGISTER"


class PublishCandidateRegisterRejected(Exception):
    def __init__(self, code: str, *, message: Optional[str] = None, blockers: Optional[list] = None):
        super().__init__(code)
        self.code = code
        self.message = message or code
        self.blockers = blockers or []


def register_publish_candidate(
    *,
    candidate_backend_sha: str,
    confirm_register: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    if (confirm_register or "").strip() != CONFIRM_REGISTER_TOKEN:
        raise PublishCandidateRegisterRejected("ace_publish_candidate_register_confirm_required")

    candidate = normalize_full_sha(candidate_backend_sha)
    if not candidate:
        raise PublishCandidateRegisterRejected("ace_publish_uri_candidate_sha_invalid")

    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        raise PublishCandidateRegisterRejected("ace_active_dev_cycle_invalid")
    active_sid = resolution.session_id
    if not active_sid or resolution.cycle_status == "NONE":
        raise PublishCandidateRegisterRejected("ace_dev_cycle_none")
    if resolution.cycle_status != "OPEN" or "CYCLE_ALREADY_CANCELLED" in resolution.blockers:
        raise PublishCandidateRegisterRejected("ace_dev_cycle_not_open")

    if session_id and session_id.strip().lower() != active_sid.lower():
        raise PublishCandidateRegisterRejected("ace_publish_candidate_session_mismatch")

    baseline = load_production_backend_baseline_sha()
    if not baseline:
        raise PublishCandidateRegisterRejected("ace_production_baseline_unavailable")

    backend_root = resolve_backend_root()
    if not git_commit_exists(backend_root, candidate):
        raise PublishCandidateRegisterRejected("ace_publish_uri_candidate_sha_missing")
    if candidate == baseline:
        raise PublishCandidateRegisterRejected("ace_publish_candidate_equals_production_baseline")
    if not git_is_ancestor(backend_root, baseline, candidate):
        raise PublishCandidateRegisterRejected("ace_publish_candidate_not_descendant_of_production")

    wip_sha = load_registry_wip_builder1_sha()
    if wip_sha and candidate == wip_sha:
        raise PublishCandidateRegisterRejected("ace_publish_candidate_is_wip_head")

    raw = run_publish_uri_candidate_preview_powershell(active_sid, candidate)
    if not raw.get("ok"):
        raise PublishCandidateRegisterRejected("ace_publish_uri_candidate_preview_failed")

    preview = build_public_publish_uri_candidate_preview_response(raw)
    blockers = list(preview.get("candidateBlockers") or preview.get("blockers") or [])
    if blockers:
        raise PublishCandidateRegisterRejected(
            "ace_publish_candidate_register_blocked",
            blockers=blockers,
        )

    readiness = preview.get("readiness") if isinstance(preview.get("readiness"), dict) else {}
    if not readiness.get("candidateVerifiedSafeForPublish"):
        raise PublishCandidateRegisterRejected("ace_publish_candidate_not_verified_safe")

    fingerprint = compute_publish_uri_candidate_preview_fingerprint(raw)
    record = build_registry_record(
        session_id=active_sid,
        candidate_backend_sha=candidate,
        production_backend_baseline_sha=baseline,
        candidate_frontend_sha=None,
        preview_fingerprint=fingerprint,
    )
    save_active_publish_candidate(record)
    from engine.ace_local_dev_publish_candidate_registry import publish_candidate_registry_path

    return {
        "ok": True,
        "mode": "REGISTER_PUBLISH_CANDIDATE",
        "sessionId": active_sid,
        "candidateBackendSha": candidate,
        "productionBackendBaselineSha": baseline,
        "registeredAt": record["registeredAt"],
        "status": record["status"],
        "publishUriPreviewFingerprint": fingerprint,
        "registryPath": str(publish_candidate_registry_path()),
        "note": "Local registry only; no deploy.",
    }


def discover_registered_publish_candidate() -> Dict[str, Any]:
    from engine.ace_local_dev_publish_candidate_registry import (
        evaluate_registered_candidate,
        load_publish_candidate_registry,
        load_production_backend_baseline_sha,
        publish_candidate_registry_path,
    )

    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        return {
            "ok": True,
            "hasRegisteredCandidate": False,
            "status": "UNAVAILABLE",
            "reason": "ace_active_dev_cycle_invalid",
            "message": "Active dev cycle pointer is missing or invalid.",
        }

    active_sid = resolution.session_id
    if not active_sid or resolution.cycle_status == "NONE":
        return {
            "ok": True,
            "hasRegisteredCandidate": False,
            "status": "NO_ACTIVE_CYCLE",
            "reason": "ace_dev_cycle_none",
            "message": "No active dev cycle.",
        }

    record = load_publish_candidate_registry()
    if not record:
        return {
            "ok": True,
            "hasRegisteredCandidate": False,
            "status": "NONE",
            "sessionId": active_sid,
            "message": "No publish candidate registered for local discovery.",
        }

    baseline = load_production_backend_baseline_sha() or ""
    effective, reasons = evaluate_registered_candidate(
        record=record,
        active_session_id=active_sid,
        cycle_status=resolution.cycle_status,
        current_production_baseline_sha=baseline,
    )

    if effective != "ACTIVE":
        return {
            "ok": True,
            "hasRegisteredCandidate": False,
            "status": "STALE",
            "sessionId": active_sid,
            "storedSessionId": str(record.get("sessionId") or ""),
            "staleReasons": reasons,
            "registeredAt": record.get("registeredAt"),
            "message": "Registered publish candidate is stale or belongs to another cycle.",
        }

    candidate = normalize_full_sha(str(record.get("candidateBackendSha") or "")) or ""
    return {
        "ok": True,
        "hasRegisteredCandidate": True,
        "status": "ACTIVE",
        "sessionId": active_sid,
        "candidateBackendSha": candidate,
        "productionBackendBaselineSha": str(record.get("productionBackendBaselineSha") or ""),
        "candidateFrontendSha": record.get("candidateFrontendSha"),
        "registeredAt": record.get("registeredAt"),
        "registryPath": str(publish_candidate_registry_path()),
        "message": "Registered publish candidate is valid for the active dev cycle.",
    }
