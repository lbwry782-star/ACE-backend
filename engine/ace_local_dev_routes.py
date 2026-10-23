"""
Local development API — cancel-development Preview only (no Execute).
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

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
from engine.ace_local_dev_cancel_execute import CancelExecuteRejected, execute_cancel_development
from engine.ace_local_dev_cancel_progress import load_public_progress, validate_operation_id
from engine.ace_local_dev_cycle_store import (
    active_cycle_pointer_path,
    resolve_active_dev_cycle,
    resolve_active_dev_session_id_for_api,
)
from engine.ace_local_dev_request_body import parse_strict_json_object
from engine.ace_local_dev_post_cancel_auto_start import try_resume_post_cancel_auto_start
from engine.ace_local_dev_start_cycle import start_dev_cycle
from engine.ace_local_dev_publish_uri_preview import (
    build_public_publish_uri_preview_response,
    publish_uri_no_active_cycle_response,
    publish_uri_preview_error_response,
    run_publish_uri_preview_powershell,
)

logger = logging.getLogger(__name__)

ace_local_dev_bp = Blueprint("ace_local_dev", __name__)


def _no_store_json(payload, status: int = 200):
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response, status


@ace_local_dev_bp.route("/api/local-dev/dev-cycle/status", methods=["GET"])
def local_dev_dev_cycle_status():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params()
    if bad_query is not None:
        return bad_query

    try_resume_post_cancel_auto_start()
    resolution = resolve_active_dev_cycle()
    return _no_store_json(resolution.to_status_payload(), 200)


@ace_local_dev_bp.route("/api/local-dev/cancel-development/preview", methods=["GET"])
def local_dev_cancel_development_preview():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params()
    if bad_query is not None:
        return bad_query

    if not active_cycle_pointer_path().is_file():
        try:
            resolve_configured_dev_session_id()
        except ValueError:
            payload = preview_error_response(
                error="ace_ori_dev_session_id_invalid",
                message="ACE_ORI_DEV_SESSION_ID must match ori-YYYYMMDD-HHMMSS-xxxxxxxx",
            )
            return _no_store_json(payload, 400)

    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        payload = preview_error_response(
            error="ace_active_dev_cycle_invalid",
            message="Active dev cycle pointer is missing or invalid; fix before preview.",
            blockers=list(resolution.blockers) + list(resolution.pointer_errors),
        )
        return _no_store_json(payload, 200)

    session_id = resolution.session_id
    if not session_id or resolution.cycle_status == "NONE":
        payload = no_active_cycle_response(
            reason="No active dev cycle (set active-dev-cycle.json or ACE_ORI_DEV_SESSION_ID).",
        )
        return _no_store_json(payload, 200)

    if resolution.cycle_status == "CANCELLED" or "CYCLE_ALREADY_CANCELLED" in resolution.blockers:
        payload = no_active_cycle_response(
            reason="Dev cancel cycle is CANCELLED for the configured session.",
        )
        return _no_store_json(payload, 200)

    try:
        raw = run_cancel_dev_preview_powershell(session_id)
    except FileNotFoundError:
        payload = preview_error_response(
            error="ace_cancel_preview_script_missing",
            message="Cancel preview script not found in backend scripts directory.",
        )
        return _no_store_json(payload, 503)
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
        return _no_store_json(payload, status)

    if not raw.get("ok"):
        payload = preview_error_response(
            error="ace_cancel_preview_failed",
            message=str(raw.get("error") or "unknown"),
        )
        return _no_store_json(payload, 503)

    payload = build_public_preview_response(raw)
    return _no_store_json(payload, 200)


@ace_local_dev_bp.route("/api/local-dev/publish-uri/preview", methods=["GET"])
def local_dev_publish_uri_preview():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params()
    if bad_query is not None:
        return bad_query

    resolution = resolve_active_dev_session_id_for_api()
    if resolution.cycle_status == "INVALID":
        payload = publish_uri_preview_error_response(
            error="ace_active_dev_cycle_invalid",
            message="Active dev cycle pointer is missing or invalid; fix before preview.",
            blockers=list(resolution.blockers) + list(resolution.pointer_errors),
        )
        return _no_store_json(payload, 200)

    session_id = resolution.session_id
    if not session_id or resolution.cycle_status == "NONE":
        payload = publish_uri_no_active_cycle_response(
            reason="No active dev cycle (set active-dev-cycle.json or ACE_ORI_DEV_SESSION_ID).",
        )
        return _no_store_json(payload, 200)

    if resolution.cycle_status == "CANCELLED" or "CYCLE_ALREADY_CANCELLED" in resolution.blockers:
        payload = publish_uri_no_active_cycle_response(
            reason="Dev cancel cycle is CANCELLED for the configured session.",
        )
        return _no_store_json(payload, 200)

    try:
        raw = run_publish_uri_preview_powershell(session_id)
    except FileNotFoundError:
        payload = publish_uri_preview_error_response(
            error="ace_publish_uri_preview_script_missing",
            message="Publish Uri preview script not found in backend scripts directory.",
        )
        return _no_store_json(payload, 503)
    except RuntimeError as exc:
        msg = str(exc)
        logger.warning("LOCAL_DEV_PUBLISH_URI_PREVIEW_FAILED sessionId=%s err=%s", session_id, msg)
        if "SESSION_JSON_MISSING" in msg or "SESSION_NOT_READY" in msg:
            payload = publish_uri_no_active_cycle_response(
                reason="Configured session is missing or not READY.",
            )
            status = 200
        elif "REGISTRY" in msg.upper():
            payload = publish_uri_preview_error_response(
                error="ace_publish_uri_preview_blocked",
                message="Preview blocked; see blockers.",
                blockers=[msg],
            )
            status = 200
        else:
            payload = publish_uri_preview_error_response(
                error="ace_publish_uri_preview_failed",
                message=msg[:500],
            )
            status = 503
        return _no_store_json(payload, status)

    if not raw.get("ok"):
        payload = publish_uri_preview_error_response(
            error="ace_publish_uri_preview_failed",
            message=str(raw.get("error") or "unknown"),
        )
        return _no_store_json(payload, 503)

    payload = build_public_publish_uri_preview_response(raw)
    return _no_store_json(payload, 200)


@ace_local_dev_bp.route("/api/local-dev/dev-cycle/start", methods=["POST"])
def local_dev_dev_cycle_start():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params()
    if bad_query is not None:
        return bad_query

    body, body_err = parse_strict_json_object(request, allowed_keys={"confirmStart"})
    if body_err:
        return _no_store_json({"ok": False, "error": body_err}, 400)

    try:
        payload = start_dev_cycle(confirm_start=str(body.get("confirmStart") or ""))
    except ValueError as exc:
        code = str(exc)
        status = 409 if "open_exists" in code else 400
        return _no_store_json({"ok": False, "error": code, "message": code}, status)
    except RuntimeError as exc:
        logger.exception("LOCAL_DEV_START_CYCLE_FAILED")
        return _no_store_json(
            {"ok": False, "error": "ace_dev_cycle_start_failed", "message": str(exc)[:500]},
            503,
        )

    return _no_store_json(payload, 200)


@ace_local_dev_bp.route("/api/local-dev/cancel-development/progress", methods=["GET"])
def local_dev_cancel_development_progress():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params(frozenset({"operationId", "sessionId"}))
    if bad_query is not None:
        return bad_query

    operation_id = (request.args.get("operationId") or "").strip()
    if not operation_id:
        return _no_store_json(
            {"ok": False, "error": "ace_cancel_progress_operation_id_required"},
            400,
        )
    try:
        validate_operation_id(operation_id)
    except ValueError:
        return _no_store_json(
            {
                "ok": False,
                "error": "ace_cancel_progress_operation_id_invalid",
                "message": (
                    "operationId must be exactly 32 lowercase hexadecimal characters "
                    "(UUID without hyphens)."
                ),
            },
            400,
        )
    session_hint = (request.args.get("sessionId") or "").strip() or None
    payload = load_public_progress(operation_id, session_id_hint=session_hint)
    if payload is None:
        return _no_store_json({"ok": False, "error": "ace_cancel_progress_not_found"}, 404)
    return _no_store_json(payload, 200)


@ace_local_dev_bp.route("/api/local-dev/cancel-development/execute", methods=["POST"])
def local_dev_cancel_development_execute():
    denied = assert_local_dev_loopback_api_access()
    if denied is not None:
        return denied
    bad_query = reject_unexpected_query_params()
    if bad_query is not None:
        return bad_query

    body, body_err = parse_strict_json_object(
        request,
        allowed_keys={"confirmExecute", "previewFingerprint", "operationId"},
    )
    if body_err:
        return _no_store_json({"ok": False, "error": body_err}, 400)

    def _with_operation(payload: dict, operation_id: str | None) -> dict:
        if operation_id:
            payload = dict(payload)
            payload["operationId"] = operation_id
        return payload

    try:
        payload = execute_cancel_development(
            confirm_execute=str(body.get("confirmExecute") or ""),
            preview_fingerprint=str(body.get("previewFingerprint") or ""),
            operation_id=str(body.get("operationId") or "") or None,
        )
    except CancelExecuteRejected as exc:
        code = str(exc)
        op = exc.operation_id
        if code.startswith("ace_cancel_execute_blockers:"):
            blockers = code.split(":", 1)[1].split(",") if ":" in code else []
            return _no_store_json(
                _with_operation(
                    {
                        "ok": False,
                        "error": "ace_cancel_execute_blocked",
                        "message": "Execute blocked; refresh preview.",
                        "blockers": blockers,
                    },
                    op,
                ),
                409,
            )
        if code == "ace_cancel_execute_not_allowed":
            return _no_store_json(
                _with_operation(
                    {
                        "ok": False,
                        "error": code,
                        "message": (
                            "Real cancel Execute is blocked. Use isolated TEMP tests or set "
                            "ACE_LOCAL_DEV_ALLOW_REAL_EXECUTE after explicit approval."
                        ),
                    },
                    op,
                ),
                403,
            )
        status = 409 if "in_progress" in code or "stale" in code or "cancelled" in code else 400
        if code == "ace_cancel_progress_operation_id_already_used":
            status = 409
        return _no_store_json(_with_operation({"ok": False, "error": code, "message": code}, op), status)
    except ValueError as exc:
        code = str(exc)
        if code == "ace_cancel_progress_operation_id_invalid":
            return _no_store_json({"ok": False, "error": code, "message": code}, 400)
        status = 409 if "in_progress" in code or "stale" in code or "cancelled" in code else 400
        return _no_store_json({"ok": False, "error": code, "message": code}, status)
    except FileNotFoundError as exc:
        logger.exception("LOCAL_DEV_CANCEL_EXECUTE_MISSING")
        return _no_store_json(
            {"ok": False, "error": "ace_cancel_execute_failed", "message": str(exc)[:500]},
            503,
        )
    except RuntimeError as exc:
        logger.exception("LOCAL_DEV_CANCEL_EXECUTE_RUNTIME")
        return _no_store_json(
            {"ok": False, "error": "ace_cancel_execute_failed", "message": str(exc)[:500]},
            503,
        )

    if not payload.get("ok"):
        return _no_store_json(payload, 503)
    return _no_store_json(payload, 200)
