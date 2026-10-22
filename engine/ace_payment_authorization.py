"""
ACE payment authorization guards for Builder1/Builder2 routes.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from flask import Request, jsonify

from engine.ace_payment_checkout_store import (
    CHECKOUT_STATE_BOUND,
    CHECKOUT_STATE_PAID,
    bind_checkout_to_resource,
    checkout_allows_builder,
    checkout_bound_to_resource,
    get_checkout_record,
    verify_checkout_browser_token,
)
from engine.ace_security_config import (
    assert_protected_route_security_config,
    assert_security_redis_available,
    is_security_enabled,
    is_security_explicitly_disabled,
)

logger = logging.getLogger(__name__)

CHECKOUT_ID_HEADER = "X-ACE-Checkout-Id"


def _extract_bearer_token(request: Request) -> str:
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def extract_checkout_credentials(request: Request) -> Tuple[str, str]:
    checkout_id = (request.headers.get(CHECKOUT_ID_HEADER) or "").strip()
    if not checkout_id and request.is_json:
        body = request.get_json(silent=True)
        if isinstance(body, dict):
            checkout_id = (body.get("checkoutId") or "").strip()
    browser_token = _extract_bearer_token(request)
    return checkout_id, browser_token


def security_disabled_response() -> Optional[Tuple[Any, int]]:
    return None


def require_security_redis_or_fail() -> Optional[Tuple[Any, int]]:
    if is_security_explicitly_disabled():
        return None
    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err
    if not is_security_enabled():
        return None
    return assert_security_redis_available()


def auth_reject(error_code: str, *, checkout_id: str = "", status: int = 403) -> Tuple[Any, int]:
    logger.warning(
        "ACE_SECURITY_AUTH_REJECTED reason=%s checkoutId=%s",
        error_code,
        checkout_id,
    )
    return jsonify({"ok": False, "error": error_code}), status


def authorize_builder_create(
    *,
    request: Request,
    builder: str,
    client_quantity: Any,
    resource_id: str,
    resource_type: str,
    initial_create: bool = False,
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[Tuple[Any, int]]]:
    """
    For initial Builder create routes when security is enabled.

    Returns (purchased_quantity, checkout_id, idem_bind_code, error_response).
    When security disabled, returns (None, None, None, None) — caller uses legacy quantity parsing.

    initial_create=True rejects a second initial-create attempt once checkout is already bound.
    """
    if is_security_explicitly_disabled():
        return None, None, None, None

    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return None, None, None, config_err

    if not is_security_enabled():
        return None, None, None, None

    redis_err = require_security_redis_or_fail()
    if redis_err is not None:
        return None, None, None, redis_err

    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return None, None, None, auth_reject("ace_checkout_credentials_required", checkout_id=checkout_id)

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return None, None, None, auth_reject(token_err or "browser_token_invalid", checkout_id=checkout_id)

    if not checkout_allows_builder(record, builder):
        return None, None, None, auth_reject("ace_checkout_builder_mismatch", checkout_id=checkout_id)

    if record.state == CHECKOUT_STATE_BOUND:
        if checkout_bound_to_resource(record, resource_id):
            if initial_create:
                return None, None, None, auth_reject(
                    "ace_checkout_initial_already_started",
                    checkout_id=checkout_id,
                )
            return record.purchased_quantity, checkout_id, "idempotent", None
        return None, None, None, auth_reject("ace_checkout_already_bound", checkout_id=checkout_id)

    if record.state != CHECKOUT_STATE_PAID:
        return None, None, None, auth_reject("ace_checkout_not_paid", checkout_id=checkout_id)

    if client_quantity is not None:
        try:
            if int(client_quantity) != record.purchased_quantity:
                return None, None, None, auth_reject("ace_purchased_quantity_mismatch", checkout_id=checkout_id)
        except (TypeError, ValueError):
            return None, None, None, auth_reject("ace_purchased_quantity_mismatch", checkout_id=checkout_id)

    ok, bind_code = bind_checkout_to_resource(
        checkout_id=checkout_id,
        resource_id=resource_id,
        resource_type=resource_type,
    )
    if not ok and bind_code != "idempotent":
        return None, None, None, auth_reject(bind_code, checkout_id=checkout_id)

    return record.purchased_quantity, checkout_id, bind_code, None


def peek_builder_checkout_entitlement(
    *,
    request: Request,
    builder: str,
    client_quantity: Any,
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[Tuple[Any, int]]]:
    """
    Read checkout entitlement without binding. Returns (quantity, bound_resource_id, checkout_id, error).
    """
    if is_security_explicitly_disabled():
        return None, None, None, None

    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return None, None, None, config_err

    if not is_security_enabled():
        return None, None, None, None

    redis_err = require_security_redis_or_fail()
    if redis_err is not None:
        return None, None, None, redis_err

    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return None, None, None, auth_reject("ace_checkout_credentials_required", checkout_id=checkout_id)

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return None, None, None, auth_reject(token_err or "browser_token_invalid", checkout_id=checkout_id)

    if not checkout_allows_builder(record, builder):
        return None, None, None, auth_reject("ace_checkout_builder_mismatch", checkout_id=checkout_id)

    if client_quantity is not None:
        try:
            if int(client_quantity) != record.purchased_quantity:
                return None, None, None, auth_reject("ace_purchased_quantity_mismatch", checkout_id=checkout_id)
        except (TypeError, ValueError):
            return None, None, None, auth_reject("ace_purchased_quantity_mismatch", checkout_id=checkout_id)

    bound_resource_id = ""
    if record.state == CHECKOUT_STATE_BOUND:
        bound_resource_id = str(record.bound_resource_id or "")
    elif record.state != CHECKOUT_STATE_PAID:
        return None, None, None, auth_reject("ace_checkout_not_paid", checkout_id=checkout_id)

    return record.purchased_quantity, bound_resource_id or None, checkout_id, None


def authorize_builder_resource_access(
    *,
    request: Request,
    builder: str,
    resource_id: str,
) -> Optional[Tuple[Any, int]]:
    """Protect continue/poll/cancel/download routes tied to a bound resource."""
    if is_security_explicitly_disabled():
        return None

    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err

    if not is_security_enabled():
        return None

    redis_err = require_security_redis_or_fail()
    if redis_err is not None:
        return redis_err

    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return auth_reject("ace_checkout_credentials_required", checkout_id=checkout_id)

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return auth_reject(token_err or "browser_token_invalid", checkout_id=checkout_id)

    if not checkout_allows_builder(record, builder):
        return auth_reject("ace_checkout_builder_mismatch", checkout_id=checkout_id)

    if record.state != CHECKOUT_STATE_BOUND:
        return auth_reject("ace_checkout_not_bound", checkout_id=checkout_id)

    if not checkout_bound_to_resource(record, resource_id):
        return auth_reject("ace_checkout_resource_mismatch", checkout_id=checkout_id)

    return None


def authorize_builder1_campaign_access(request: Request, campaign_id: str) -> Optional[Tuple[Any, int]]:
    return authorize_builder_resource_access(
        request=request,
        builder="builder1",
        resource_id=campaign_id,
    )


def authorize_builder2_allowance_access(request: Request, allowance_id: str) -> Optional[Tuple[Any, int]]:
    return authorize_builder_resource_access(
        request=request,
        builder="builder2",
        resource_id=allowance_id,
    )


def authorize_builder1_artifact_access(request: Request, token: str) -> Optional[Tuple[Any, int]]:
    if is_security_explicitly_disabled():
        return None

    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err

    if not is_security_enabled():
        return None

    redis_err = require_security_redis_or_fail()
    if redis_err is not None:
        return redis_err

    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return auth_reject("ace_checkout_credentials_required", checkout_id=checkout_id)

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return auth_reject(token_err or "browser_token_invalid", checkout_id=checkout_id)

    if record.builder != "builder1" or record.state != CHECKOUT_STATE_BOUND:
        return auth_reject("ace_checkout_not_bound", checkout_id=checkout_id)

    campaign_id = str(record.bound_resource_id or "")
    if not campaign_id:
        return auth_reject("ace_checkout_campaign_missing", checkout_id=checkout_id)

    from engine.builder1_image_artifact_store import artifact_token_for_ad

    token_clean = (token or "").strip()
    for ad_index in range(1, 5):
        for plan_revision in range(0, 12):
            if artifact_token_for_ad(
                campaign_id=campaign_id,
                ad_index=ad_index,
                plan_revision=plan_revision,
            ) == token_clean:
                return None
    return auth_reject("ace_artifact_token_mismatch", checkout_id=checkout_id)


def resolve_builder2_allowance_for_job(job: dict) -> str:
    return str(job.get("videoAllowanceId") or job.get("video_allowance_id") or "").strip()


def _job_urls_contain_token(job: dict, token: str) -> bool:
    token_clean = (token or "").strip()
    if not token_clean:
        return False
    url_field_names = (
        "videoUrl",
        "video_url",
        "finalVideoUrl",
        "final_video_url",
        "finalPublicUrl",
        "final_public_url",
        "headlineArtifactUrl",
        "musicArtifactUrl",
        "closureVideoUrl",
    )
    for key in url_field_names:
        if token_clean in str(job.get(key) or ""):
            return True
    for value in job.values():
        text = str(value or "")
        if token_clean not in text:
            continue
        if any(
            marker in text
            for marker in (
                "/video-headline/",
                "/builder2-final-video/",
                "/builder2-music-artifact/",
            )
        ):
            return True
    return False


def _allowance_artifact_token_matches(allowance_id: str, token: str) -> bool:
    from engine.builder2_video_allowance_store import get_video_allowance
    from engine.video_jobs_redis import video_job_get_raw

    allowance = get_video_allowance(allowance_id)
    if not allowance:
        return False
    job_ids: list[str] = []
    for entry in allowance.get("videos") or []:
        if not isinstance(entry, dict):
            continue
        jid = str(entry.get("jobId") or "").strip()
        if jid:
            job_ids.append(jid)
    for jid in job_ids:
        job = video_job_get_raw(jid) or {}
        if _job_urls_contain_token(job, token):
            return True
    return False


def authorize_builder2_artifact_access(request: Request, token: str) -> Optional[Tuple[Any, int]]:
    if is_security_explicitly_disabled():
        return None

    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err

    if not is_security_enabled():
        return None

    redis_err = require_security_redis_or_fail()
    if redis_err is not None:
        return redis_err

    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return auth_reject("ace_checkout_credentials_required", checkout_id=checkout_id)

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return auth_reject(token_err or "browser_token_invalid", checkout_id=checkout_id)

    if record.builder != "builder2" or record.state != CHECKOUT_STATE_BOUND:
        return auth_reject("ace_checkout_not_bound", checkout_id=checkout_id)

    allowance_id = str(record.bound_resource_id or "")
    if not allowance_id:
        return auth_reject("ace_checkout_allowance_missing", checkout_id=checkout_id)

    if _allowance_artifact_token_matches(allowance_id, token):
        return None
    return auth_reject("ace_artifact_token_mismatch", checkout_id=checkout_id)


def _normalize_video_url_for_match(url: str) -> str:
    from urllib.parse import unquote, urlparse

    cleaned = unquote((url or "").strip())
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/")
    if not scheme or not netloc:
        return cleaned.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def _job_urls_contain_video_url(job: dict, video_url: str) -> bool:
    target = _normalize_video_url_for_match(video_url)
    if not target:
        return False
    url_field_names = (
        "videoUrl",
        "video_url",
        "finalVideoUrl",
        "final_video_url",
        "finalPublicUrl",
        "final_public_url",
        "headlineArtifactUrl",
        "musicArtifactUrl",
        "closureVideoUrl",
    )
    for key in url_field_names:
        raw = str(job.get(key) or "").strip()
        if not raw:
            continue
        if _normalize_video_url_for_match(raw) == target:
            return True
        if target in raw or raw in video_url:
            return True
    return _job_urls_contain_token(job, _extract_url_token(video_url))


def _extract_url_token(video_url: str) -> str:
    import re

    for pattern in (
        r"/api/video-headline/([a-f0-9]{32})(?:[/?#]|$)",
        r"/api/builder2-final-video/([a-f0-9]{32})(?:[/?#]|$)",
        r"/api/builder2-music-artifact/([a-f0-9]{32})(?:[/?#]|$)",
    ):
        match = re.search(pattern, video_url or "", re.IGNORECASE)
        if match:
            return (match.group(1) or "").strip()
    return ""


def _allowance_video_url_matches(allowance_id: str, video_url: str) -> bool:
    from engine.builder2_video_allowance_store import get_video_allowance
    from engine.video_jobs_redis import video_job_get_raw

    allowance = get_video_allowance(allowance_id)
    if not allowance:
        return False
    token = _extract_url_token(video_url)
    if token and _allowance_artifact_token_matches(allowance_id, token):
        return True
    job_ids: list[str] = []
    for entry in allowance.get("videos") or []:
        if not isinstance(entry, dict):
            continue
        jid = str(entry.get("jobId") or "").strip()
        if jid:
            job_ids.append(jid)
    for jid in job_ids:
        job = video_job_get_raw(jid) or {}
        if _job_urls_contain_video_url(job, video_url):
            return True
    return False


def authorize_builder2_video_url_for_download(
    request: Request,
    video_url: str,
) -> Optional[Tuple[Any, int]]:
    """
    Authorize GET /api/download-video-zip for local and remote video URLs when security is ON.
    Fails closed when the URL cannot be tied to the bound Builder2 allowance.
    """
    if is_security_explicitly_disabled():
        return None

    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err

    if not is_security_enabled():
        return None

    redis_err = require_security_redis_or_fail()
    if redis_err is not None:
        return redis_err

    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return auth_reject("ace_checkout_credentials_required", checkout_id=checkout_id)

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return auth_reject(token_err or "browser_token_invalid", checkout_id=checkout_id)

    if record.builder != "builder2" or record.state != CHECKOUT_STATE_BOUND:
        return auth_reject("ace_checkout_not_bound", checkout_id=checkout_id)

    allowance_id = str(record.bound_resource_id or "")
    if not allowance_id:
        return auth_reject("ace_checkout_allowance_missing", checkout_id=checkout_id)

    if _allowance_video_url_matches(allowance_id, video_url):
        return None
    return auth_reject("ace_video_url_not_authorized", checkout_id=checkout_id)


def authorize_builder2_job_access(request: Request, job: dict) -> Optional[Tuple[Any, int]]:
    allowance_id = resolve_builder2_allowance_for_job(job)
    if allowance_id:
        return authorize_builder2_allowance_access(request, allowance_id)
    checkout_id, browser_token = extract_checkout_credentials(request)
    if not checkout_id and not browser_token:
        return auth_reject("ace_checkout_credentials_required")
    record = get_checkout_record(checkout_id) if checkout_id else None
    if record and record.builder == "builder2" and record.bound_resource_id:
        return authorize_builder_resource_access(
            request=request,
            builder="builder2",
            resource_id=record.bound_resource_id,
        )
    return auth_reject("ace_checkout_not_bound", checkout_id=checkout_id)
