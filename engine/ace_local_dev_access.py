"""
Local-dev-only HTTP access guards (loopback + ACE_LOCAL_DEV_ENABLED).
"""
from __future__ import annotations

from typing import Optional, Tuple

from flask import Response, jsonify, request

from engine.ace_local_dev_config import (
    is_local_dev_enabled,
    is_loopback_bind_confirmed,
    is_loopback_bind_host,
)

_LOCAL_DEV_CACHE = "no-store"


def _deny(error: str, status: int = 403) -> Tuple[Response, int]:
    response = jsonify({"ok": False, "error": error})
    response.headers["Cache-Control"] = _LOCAL_DEV_CACHE
    return response, status


def assert_local_dev_loopback_api_access() -> Optional[Tuple[Response, int]]:
    """
    Fail closed unless local dev is enabled, process bind is loopback, and client is loopback.
    """
    if not is_local_dev_enabled():
        return _deny("ace_local_dev_not_enabled", 403)
    if not is_loopback_bind_confirmed():
        return _deny("ace_local_dev_loopback_bind_not_confirmed", 403)

    remote = (request.remote_addr or "").strip()
    if not is_loopback_bind_host(remote):
        return _deny("ace_local_dev_loopback_client_required", 403)

    return None


def reject_unexpected_query_params() -> Optional[Tuple[Response, int]]:
    if request.args:
        return _deny("ace_local_dev_unexpected_query_parameters", 400)
    return None
