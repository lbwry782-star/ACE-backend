"""Shared publish candidate prechecks (preview + fingerprint + product scope)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from engine.ace_local_dev_publish_candidate_registry import (
    evaluate_registered_candidate,
    load_production_backend_baseline_sha,
    load_publish_candidate_registry,
    normalize_full_sha,
)
from engine.ace_local_dev_cycle_store import resolve_active_dev_session_id_for_api
from engine.ace_local_dev_publish_uri_candidate_preview import (
    build_public_publish_uri_candidate_preview_response,
    run_publish_uri_candidate_preview_powershell,
)
from engine.ace_local_dev_publish_uri_fingerprint import (
    compute_publish_uri_candidate_preview_fingerprint,
)
from engine.ace_local_dev_publish_uri_candidate_prepare import (
    FORBIDDEN_MIXED_COMMIT_SHA,
    _validate_no_forbidden_commits,
    _validate_worker_coupling,
    build_explicit_execution_plan,
)


class PublishPrecheckRejected(Exception):
    def __init__(
        self,
        code: str,
        *,
        message: str | None = None,
        blockers: List[str] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code
        self.blockers = blockers or []


@dataclass
class PublishPrecheckResult:
    session_id: str
    candidate_backend_sha: str
    raw_preview: Dict[str, Any]
    preview_public: Dict[str, Any]
    plan: Dict[str, Any]
    readiness: Dict[str, Any]
    fingerprint: str
    execution_plan: Dict[str, Any]


def validate_fingerprint_token(value: str) -> str:
    fp = (value or "").strip().lower()
    if len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp):
        raise PublishPrecheckRejected("ace_publish_uri_candidate_fingerprint_invalid")
    return fp


def resolve_active_registered_candidate() -> Tuple[str, str, Dict[str, Any]]:
    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        raise PublishPrecheckRejected("ace_active_dev_cycle_invalid")
    session_id = resolution.session_id
    if not session_id or resolution.cycle_status == "NONE":
        raise PublishPrecheckRejected("ace_dev_cycle_none")
    if resolution.cycle_status != "OPEN" or "CYCLE_ALREADY_CANCELLED" in resolution.blockers:
        raise PublishPrecheckRejected("ace_dev_cycle_not_open")

    record = load_publish_candidate_registry()
    if not record:
        raise PublishPrecheckRejected("ace_publish_candidate_not_registered")

    baseline = load_production_backend_baseline_sha() or ""
    effective, reasons = evaluate_registered_candidate(
        record=record,
        active_session_id=session_id,
        cycle_status=resolution.cycle_status,
        current_production_baseline_sha=baseline,
    )
    if effective != "ACTIVE":
        raise PublishPrecheckRejected(
            "ace_publish_candidate_registry_stale",
            blockers=reasons,
        )

    candidate = normalize_full_sha(str(record.get("candidateBackendSha") or ""))
    if not candidate:
        raise PublishPrecheckRejected("ace_publish_uri_candidate_sha_invalid")
    return session_id, candidate, record


def run_publish_candidate_prechecks(
    *,
    expected_fingerprint: str,
) -> PublishPrecheckResult:
    expected_fp = validate_fingerprint_token(expected_fingerprint)
    session_id, candidate_sha, _record = resolve_active_registered_candidate()

    raw = run_publish_uri_candidate_preview_powershell(session_id, candidate_sha)
    if not raw.get("ok"):
        raise PublishPrecheckRejected("ace_publish_uri_candidate_preview_failed")

    plan = raw.get("plan") if isinstance(raw.get("plan"), dict) else {}
    if normalize_full_sha(str(plan.get("candidateBackendSha") or "")) != candidate_sha:
        raise PublishPrecheckRejected("ace_publish_prepare_candidate_mismatch")

    fresh_fp = compute_publish_uri_candidate_preview_fingerprint(raw)
    if fresh_fp.lower() != expected_fp:
        raise PublishPrecheckRejected("ace_publish_uri_candidate_fingerprint_stale")

    preview = build_public_publish_uri_candidate_preview_response(raw)
    blockers = list(preview.get("candidateBlockers") or preview.get("blockers") or [])
    readiness = preview.get("readiness") if isinstance(preview.get("readiness"), dict) else {}

    extra_blockers = _validate_no_forbidden_commits(plan)
    if extra_blockers:
        blockers = sorted(set(blockers + extra_blockers))
    if blockers:
        raise PublishPrecheckRejected("ace_publish_prepare_blocked", blockers=blockers)

    if not readiness.get("candidateVerifiedSafeForPublish"):
        raise PublishPrecheckRejected("ace_publish_candidate_not_verified_safe")

    if not readiness.get("canPrepareLocalPublish"):
        drift = list(preview.get("registryInfraDrift") or []) + list(
            preview.get("registryWipDrift") or []
        )
        raise PublishPrecheckRejected(
            "ace_publish_prepare_not_ready",
            blockers=drift or ["canPrepareLocalPublish=false"],
        )

    worker_blockers = _validate_worker_coupling(plan)
    if worker_blockers:
        raise PublishPrecheckRejected("ace_publish_prepare_worker_blocked", blockers=worker_blockers)

    execution_plan = build_explicit_execution_plan(plan=plan, raw_preview=raw)
    return PublishPrecheckResult(
        session_id=session_id,
        candidate_backend_sha=candidate_sha,
        raw_preview=raw,
        preview_public=preview,
        plan=plan,
        readiness=readiness,
        fingerprint=fresh_fp,
        execution_plan=execution_plan,
    )


__all__ = [
    "FORBIDDEN_MIXED_COMMIT_SHA",
    "PublishPrecheckRejected",
    "PublishPrecheckResult",
    "resolve_active_registered_candidate",
    "run_publish_candidate_prechecks",
    "validate_fingerprint_token",
]
