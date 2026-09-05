"""
Flask routes for temporary iCount protocol diagnostics.

Register via register_icount_diagnostic_routes(app) — remove after discovery.
"""
from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, Response, redirect, request

from engine.icount_diagnostic_harness import (
    build_observation_report,
    ipn_success_response_body,
    log_icount_diagnostic_observation,
    resolve_return_redirect_target,
    validate_diagnostic_nonce,
)

icount_diagnostic_bp = Blueprint(
    "icount_diagnostic",
    __name__,
    url_prefix="/api/diagnostics/icount",
)


def _parse_json_payload() -> Any:
    parsed = request.get_json(silent=True)
    if parsed is not None:
        return parsed
    raw = request.get_data(as_text=True)
    if not raw or not raw.strip():
        return None
    try:
        import json

        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _query_mapping() -> Dict[str, Any]:
    return {str(key): request.args.get(key) for key in request.args.keys()}


def _form_mapping() -> Dict[str, Any]:
    if not request.form:
        return {}
    return {str(key): request.form.get(key) for key in request.form.keys()}


def _header_mapping() -> Dict[str, str]:
    return {str(key): str(value) for key, value in request.headers.items()}


def _invalid_nonce_response() -> tuple[str, int]:
    return "invalid diagnostic nonce", 400


@icount_diagnostic_bp.route("/ipn/<diagnostic_nonce>", methods=["GET", "POST"])
def icount_diagnostic_ipn(diagnostic_nonce: str):
    if not validate_diagnostic_nonce(diagnostic_nonce):
        return _invalid_nonce_response()

    report = build_observation_report(
        channel="IPN",
        diagnostic_nonce=diagnostic_nonce,
        method=request.method,
        content_type=str(request.content_type or ""),
        query_fields=_query_mapping(),
        form_fields=_form_mapping(),
        json_payload=_parse_json_payload(),
        header_map=_header_mapping(),
    )
    log_icount_diagnostic_observation(report)

    body, status, headers = ipn_success_response_body()
    return Response(body, status=status, headers=headers)


@icount_diagnostic_bp.route("/return/<diagnostic_nonce>", methods=["GET", "POST"])
def icount_diagnostic_return(diagnostic_nonce: str):
    if not validate_diagnostic_nonce(diagnostic_nonce):
        return _invalid_nonce_response()

    report = build_observation_report(
        channel="RETURN",
        diagnostic_nonce=diagnostic_nonce,
        method=request.method,
        content_type=str(request.content_type or ""),
        query_fields=_query_mapping(),
        form_fields=_form_mapping(),
        json_payload=_parse_json_payload(),
        header_map=_header_mapping(),
    )
    log_icount_diagnostic_observation(report)

    target = resolve_return_redirect_target()
    return redirect(target, code=302)


def register_icount_diagnostic_routes(app: Any) -> None:
    import logging

    logging.getLogger(__name__).info(
        "ICOUNT_DIAG_ROUTES_REGISTERED prefix=/api/diagnostics/icount "
        "ipn=/api/diagnostics/icount/ipn/<diagnosticNonce> "
        "return=/api/diagnostics/icount/return/<diagnosticNonce>"
    )
    app.register_blueprint(icount_diagnostic_bp)
