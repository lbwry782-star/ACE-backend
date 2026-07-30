"""
Builder2 final-output diagnostics — read-only media completion vs resume classification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from engine.builder2_closure_render import classify_url_route_family
from engine.builder2_final_video_verification import extract_builder2_final_video_token
from engine.builder2_winner_persistence import (
    collect_winner_media_continuation_missing_fields,
    compute_winner_development_plan_fingerprint,
    is_winner_media_continuation_ready,
)

_MEDIA_DIAGNOSTIC_COMPLETED = "completed"
_MEDIA_DIAGNOSTIC_READY_TO_RESUME = "ready_to_resume"
_MEDIA_DIAGNOSTIC_INCOMPLETE_NOT_READY = "incomplete_not_ready"

_SIGNED_QUERY_MARKERS = (
    "signature",
    "sig",
    "token",
    "access_token",
    "credential",
    "expires",
    "x-amz-",
    "se=",
    "sp=",
    "sv=",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _media_bucket(state: Dict[str, Any]) -> Dict[str, Any]:
    media = state.get("mediaResume")
    return media if isinstance(media, dict) else {}


def _first_url(*values: Any) -> str:
    for value in values:
        token = _clean(value)
        if token:
            return token
    return ""


def assess_winner_plan_state_store_agreement(state: Dict[str, Any]) -> bool:
    plan = state.get("winnerDevelopmentPlan")
    if not isinstance(plan, dict) or not plan:
        return False
    stored_fp = _clean(state.get("winnerDevelopmentPlanFingerprint"))
    if not stored_fp:
        return True
    return stored_fp == compute_winner_development_plan_fingerprint(plan)


def closure_video_present(state: Dict[str, Any]) -> bool:
    media = _media_bucket(state)
    return bool(_first_url(media.get("finalVideoWithClosureUrl")))


def durable_final_url_present(state: Dict[str, Any]) -> bool:
    media = _media_bucket(state)
    return bool(_first_url(media.get("finalPublicUrl")))


def is_builder2_media_diagnostically_completed(state: Dict[str, Any]) -> bool:
    return (
        closure_video_present(state)
        and durable_final_url_present(state)
        and assess_winner_plan_state_store_agreement(state)
    )


def classify_builder2_media_diagnostic_phase(state: Dict[str, Any]) -> str:
    if is_builder2_media_diagnostically_completed(state):
        return _MEDIA_DIAGNOSTIC_COMPLETED
    if is_winner_media_continuation_ready(state):
        return _MEDIA_DIAGNOSTIC_READY_TO_RESUME
    return _MEDIA_DIAGNOSTIC_INCOMPLETE_NOT_READY


def collect_media_resume_contract_missing_fields(state: Dict[str, Any]) -> List[str]:
    if is_builder2_media_diagnostically_completed(state):
        return []
    return list(collect_winner_media_continuation_missing_fields(state))


def _url_has_signed_query_credentials(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.password:
        return True
    query = (parsed.query or "").lower()
    if not query:
        return False
    return any(marker in query for marker in _SIGNED_QUERY_MARKERS)


def _is_private_storage_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    private_markers = (
        "amazonaws.com",
        "cloudfront.net",
        "runway",
        "blob.core.windows.net",
        "storage.googleapis.com",
    )
    return any(marker in host for marker in private_markers)


def resolve_safe_durable_final_public_output(state: Dict[str, Any]) -> Dict[str, Any]:
    media = _media_bucket(state)
    candidates = [
        _clean(media.get("finalPublicUrl")),
        _clean(media.get("finalVideoWithClosureUrl")),
    ]
    report: Dict[str, Any] = {
        "durableFinalPublicUrl": None,
        "durableFinalPublicPath": None,
        "finalVideoToken": None,
        "finalUrlPubliclyVerifiable": False,
    }

    def _resolve_from_url(stored_url: str) -> Dict[str, Any]:
        empty = {
            "durableFinalPublicUrl": None,
            "durableFinalPublicPath": None,
            "finalVideoToken": None,
            "finalUrlPubliclyVerifiable": False,
        }
        if not stored_url:
            return empty

        route_family = classify_url_route_family(stored_url)
        parsed = urlparse(stored_url)
        token = extract_builder2_final_video_token(stored_url)

        if route_family == "api/builder2-final-video" and token:
            path = f"/api/builder2-final-video/{token}"
            resolved = dict(empty)
            resolved["finalVideoToken"] = token
            resolved["durableFinalPublicPath"] = path
            if parsed.scheme and parsed.netloc:
                resolved["durableFinalPublicUrl"] = f"{parsed.scheme}://{parsed.netloc}{path}"
            else:
                resolved["durableFinalPublicUrl"] = path
            resolved["finalUrlPubliclyVerifiable"] = True
            return resolved

        if route_family == "api/video-headline":
            path = parsed.path or ""
            headline_token = path.rsplit("/", 1)[-1].split(".", 1)[0] if path else ""
            if not path:
                return empty
            resolved = dict(empty)
            resolved["durableFinalPublicPath"] = path
            if headline_token:
                resolved["finalVideoToken"] = headline_token
            if parsed.scheme and parsed.netloc and not _url_has_signed_query_credentials(stored_url):
                resolved["durableFinalPublicUrl"] = f"{parsed.scheme}://{parsed.netloc}{path}"
            resolved["finalUrlPubliclyVerifiable"] = bool(headline_token)
            return resolved

        if _url_has_signed_query_credentials(stored_url) or _is_private_storage_host(stored_url):
            return empty

        if parsed.scheme in {"http", "https"} and parsed.path:
            resolved = dict(empty)
            resolved["durableFinalPublicPath"] = parsed.path
            resolved["durableFinalPublicUrl"] = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            resolved["finalUrlPubliclyVerifiable"] = True
            return resolved
        return empty

    for stored_url in candidates:
        if not stored_url:
            continue
        route_family = classify_url_route_family(stored_url)
        if route_family in {"api/builder2-final-video", "api/video-headline"}:
            return _resolve_from_url(stored_url)

    for stored_url in candidates:
        if not stored_url:
            continue
        resolved = _resolve_from_url(stored_url)
        if resolved.get("durableFinalPublicUrl") or resolved.get("durableFinalPublicPath"):
            return resolved
    return report


def build_builder2_media_diagnostic_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    phase = classify_builder2_media_diagnostic_phase(state)
    media_completed = phase == _MEDIA_DIAGNOSTIC_COMPLETED
    missing = collect_media_resume_contract_missing_fields(state)
    media_resume_needed = phase == _MEDIA_DIAGNOSTIC_READY_TO_RESUME
    safe_output = resolve_safe_durable_final_public_output(state)
    final_output_available = media_completed and bool(
        safe_output.get("durableFinalPublicUrl") or safe_output.get("durableFinalPublicPath")
    )
    media_resume_ready = media_resume_needed and not missing
    blocked_reason: Optional[str] = None
    if media_completed:
        blocked_reason = "media_already_completed"
    elif not media_resume_needed:
        blocked_reason = "media_prerequisites_incomplete"
    elif missing:
        blocked_reason = f"missing:{','.join(missing)}"

    return {
        "mediaCompleted": media_completed,
        "mediaResumeNeeded": media_resume_needed,
        "mediaDiagnosticPhase": phase,
        "mediaResumeReady": media_resume_ready,
        "mediaResumeBlockedReason": blocked_reason,
        "mediaResumeMissingFields": missing,
        "finalOutputAvailable": final_output_available,
        "closureVideoPresent": closure_video_present(state),
        "durableFinalUrlPresent": durable_final_url_present(state),
        "stateStoreAgreement": assess_winner_plan_state_store_agreement(state),
        **safe_output,
    }


def inspect_builder2_final_output(state: Dict[str, Any]) -> Dict[str, Any]:
    diagnostic = build_builder2_media_diagnostic_fields(state)
    return {
        "jobId": _clean(state.get("jobId")),
        "tournamentId": _clean(state.get("tournamentId")),
        "finalOutputAvailable": diagnostic["finalOutputAvailable"],
        "mediaCompleted": diagnostic["mediaCompleted"],
        "mediaResumeNeeded": diagnostic["mediaResumeNeeded"],
        "mediaDiagnosticPhase": diagnostic["mediaDiagnosticPhase"],
        "closureVideoPresent": diagnostic["closureVideoPresent"],
        "durableFinalUrlPresent": diagnostic["durableFinalUrlPresent"],
        "durableFinalPublicUrl": diagnostic.get("durableFinalPublicUrl"),
        "durableFinalPublicPath": diagnostic.get("durableFinalPublicPath"),
        "finalVideoToken": diagnostic.get("finalVideoToken"),
        "finalUrlPubliclyVerifiable": diagnostic.get("finalUrlPubliclyVerifiable"),
        "stateStoreAgreement": diagnostic["stateStoreAgreement"],
        "stateMutated": False,
        "paidCalls": 0,
    }
