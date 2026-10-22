"""
ACE master security configuration — single switch for payment + Builder authorization.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from flask import Response, jsonify

logger = logging.getLogger(__name__)

SECURITY_STATE_ENABLED = "enabled"
SECURITY_STATE_DISABLED = "disabled"
SECURITY_STATE_INVALID = "invalid"

_SECURITY_ENABLED_VALUES = frozenset({"true", "1", "yes", "on"})
_SECURITY_DISABLED_VALUES = frozenset({"false", "0", "no", "off"})


def resolve_security_config_state() -> str:
    """
    Three-state ACE_SECURITY_ENABLED resolution.

    - explicitly enabled: true, 1, yes, on
    - explicitly disabled: false, 0, no, off
    - invalid: missing, empty, malformed, unknown
    """
    if "ACE_SECURITY_ENABLED" not in os.environ:
        return SECURITY_STATE_INVALID
    raw = os.environ.get("ACE_SECURITY_ENABLED")
    if raw is None:
        return SECURITY_STATE_INVALID
    cleaned = raw.strip().lower()
    if not cleaned:
        return SECURITY_STATE_INVALID
    if cleaned in _SECURITY_ENABLED_VALUES:
        return SECURITY_STATE_ENABLED
    if cleaned in _SECURITY_DISABLED_VALUES:
        return SECURITY_STATE_DISABLED
    return SECURITY_STATE_INVALID


def is_security_enabled() -> bool:
    return resolve_security_config_state() == SECURITY_STATE_ENABLED


def is_security_explicitly_disabled() -> bool:
    if resolve_security_config_state() != SECURITY_STATE_DISABLED:
        return False
    from engine.ace_local_dev_config import is_local_dev_security_bypass_allowed

    return is_local_dev_security_bypass_allowed()


def is_security_config_invalid() -> bool:
    return resolve_security_config_state() == SECURITY_STATE_INVALID


def security_config_error_response(*, include_ok: bool = True) -> Tuple[Response, int]:
    payload = {"error": "ace_security_config_invalid"}
    if include_ok:
        payload = {"ok": False, "error": "ace_security_config_invalid"}
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response, 503


def assert_protected_route_security_config(*, include_ok: bool = True) -> Optional[Tuple[Response, int]]:
    """Invalid/missing config fails closed before paid/provider work."""
    if is_security_config_invalid():
        logger.warning("ACE_SECURITY_CONFIG reason=ace_security_config_invalid")
        return security_config_error_response(include_ok=include_ok)
    if resolve_security_config_state() == SECURITY_STATE_DISABLED:
        from engine.ace_local_dev_config import is_local_dev_security_bypass_allowed

        if not is_local_dev_security_bypass_allowed():
            logger.warning("ACE_SECURITY_CONFIG reason=ace_security_disabled_not_allowed")
            payload = {"error": "ace_security_disabled_not_allowed"}
            if include_ok:
                payload = {"ok": False, "error": "ace_security_disabled_not_allowed"}
            response = jsonify(payload)
            response.headers["Cache-Control"] = "no-store"
            return response, 503
    return None


def security_config_payload() -> dict:
    return {"securityEnabled": is_security_enabled()}


def security_config_response() -> Tuple[Response, int]:
    config_err = assert_protected_route_security_config(include_ok=False)
    if config_err is not None:
        return config_err
    response = jsonify(security_config_payload())
    response.headers["Cache-Control"] = "no-store"
    return response, 200


def assert_security_redis_available() -> Optional[Tuple[Response, int]]:
    """When security is ON, Redis must be reachable before paid/provider work."""
    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err
    if not is_security_enabled():
        return None
    from engine.video_jobs_redis import redis_configured

    if not redis_configured():
        logger.warning("ACE_SECURITY_AUTH_REJECTED reason=security_redis_unconfigured")
        return jsonify({"ok": False, "error": "ace_security_redis_unavailable"}), 503
    try:
        from engine.video_jobs_redis import get_redis

        get_redis().ping()
    except Exception:
        logger.warning("ACE_SECURITY_AUTH_REJECTED reason=security_redis_unreachable")
        return jsonify({"ok": False, "error": "ace_security_redis_unavailable"}), 503
    return None


def log_security_config_on_startup() -> None:
    state = resolve_security_config_state()
    if state == SECURITY_STATE_INVALID:
        logger.warning("ACE_SECURITY_CONFIG state=invalid fail_closed=true")
    elif state == SECURITY_STATE_DISABLED:
        if is_security_explicitly_disabled():
            logger.info("ACE_SECURITY_CONFIG securityEnabled=false security_checks_bypassed=true local_dev=true")
        else:
            logger.warning(
                "ACE_SECURITY_CONFIG state=disabled_not_allowed fail_closed=true "
                "(requires ACE_LOCAL_DEV_ENABLED + loopback bind)"
            )
    else:
        logger.info("ACE_SECURITY_CONFIG securityEnabled=true")
