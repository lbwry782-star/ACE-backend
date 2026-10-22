"""
Local development API — cancel-development Preview only (no Execute).
"""
from __future__ import annotations

import logging

from flask import Blueprint, Response, jsonify

from engine.ace_local_dev_access import (
    assert_local_dev_loopback_api_access,
    reject_unexpected_query_params,
)
from engine.ace_local_dev_cancel_preview import (
    build_public_preview_response,
    no_active_cycle_response,
    preview_error_response,
    resolve_configured_dev_session_id,
    run_cancel_dev_preview_powershell,
)

logger = logging.getLogger(__name__)

ace_local_dev_bp = Blueprint("ace_local_dev", __name__)


@ace_local_dev_bp.route("/api/local-dev/cancel-development/preview", methods=["GET"])
def local_dev_cancel_development_preview():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params()
    if bad_query is not None:
        return bad_query

    try:
        session_id = resolve_configured_dev_session_id()
    except ValueError:
        payload = preview_error_response(
            error="ace_ori_dev_session_id_invalid",
            message="ACE_ORI_DEV_SESSION_ID must match ori-YYYYMMDD-HHMMSS-xxxxxxxx",
        )
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response, 400

    if not session_id:
        payload = no_active_cycle_response(
            reason="No active dev cancel cycle configured (ACE_ORI_DEV_SESSION_ID unset).",
        )
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response, 200

    try:
        raw = run_cancel_dev_preview_powershell(session_id)
    except FileNotFoundError:
        payload = preview_error_response(
            error="ace_cancel_preview_script_missing",
            message="Cancel preview script not found in backend scripts directory.",
        )
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response, 503
    except RuntimeError as exc:
        msg = str(exc)
        logger.warning("LOCAL_DEV_CANCEL_PREVIEW_FAILED sessionId=%s err=%s", session_id, msg)
        if "SESSION_JSON_MISSING" in msg or "SESSION_NOT_READY" in msg:
            payload = no_active_cycle_response(
                reason="Configured session is missing or not READY.",
            )
            status = 200
        elif "PLAN_BLOCKED" in msg or "REGISTRY" in msg.upper():
            payload = preview_error_response(
                error="ace_cancel_preview_blocked",
                message="Preview plan blocked; see blockers.",
                blockers=[msg],
            )
            status = 200
        else:
            payload = preview_error_response(
                error="ace_cancel_preview_failed",
                message=msg[:500],
            )
            status = 503
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response, status

    if not raw.get("ok"):
        payload = preview_error_response(
            error="ace_cancel_preview_failed",
            message=str(raw.get("error") or "unknown"),
        )
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response, 503

    payload = build_public_preview_response(raw)
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response, 200
