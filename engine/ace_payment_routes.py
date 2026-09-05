"""
Production ACE payment routes — checkout, status, iCount IPN.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

from flask import Blueprint, Response, jsonify, request

from engine.ace_payment_authorization import extract_checkout_credentials
from engine.ace_payment_checkout_store import (
    checkout_status_public,
    create_checkout,
    verify_checkout_browser_token,
)
from engine.ace_payment_offers import append_payment_reference_to_url, get_offer, offer_public_dict
from engine.ace_payment_icount_ipn import extract_icount_form_fields, process_icount_ipn
from engine.ace_security_config import (
    assert_protected_route_security_config,
    assert_security_redis_available,
    is_security_enabled,
    is_security_explicitly_disabled,
    security_config_response,
)

logger = logging.getLogger(__name__)

ace_payment_bp = Blueprint("ace_payment", __name__)


@ace_payment_bp.route("/api/security/config", methods=["GET"])
def ace_security_config_route():
    return security_config_response()


@ace_payment_bp.route("/api/payments/checkout", methods=["POST"])
def ace_create_checkout():
    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err
    if is_security_explicitly_disabled():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "ace_security_disabled",
                    "message": "Payment checkout requires ACE_SECURITY_ENABLED=true",
                }
            ),
            503,
        )

    redis_err = assert_security_redis_available()
    if redis_err is not None:
        return redis_err

    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input"}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input"}), 400

    offer_code = str(body.get("offerCode") or "").strip()
    offer = get_offer(offer_code)
    if offer is None:
        return jsonify({"ok": False, "error": "unknown_offer"}), 400

    record, browser_token, payment_reference = create_checkout(offer)
    payment_url = append_payment_reference_to_url(offer.payment_page_url, payment_reference)
    return (
        jsonify(
            {
                "checkoutId": record.checkout_id,
                "browserToken": browser_token,
                "paymentUrl": payment_url,
                "offer": offer_public_dict(offer),
            }
        ),
        200,
    )


@ace_payment_bp.route("/api/payments/checkout/status", methods=["POST"])
def ace_checkout_status():
    config_err = assert_protected_route_security_config()
    if config_err is not None:
        return config_err
    if is_security_explicitly_disabled():
        return jsonify({"ok": False, "error": "ace_security_disabled"}), 503

    redis_err = assert_security_redis_available()
    if redis_err is not None:
        return redis_err

    if not request.is_json:
        return jsonify({"ok": False, "error": "invalid_input"}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "invalid_input"}), 400

    checkout_id = str(body.get("checkoutId") or "").strip()
    _, browser_token = extract_checkout_credentials(request)
    if not checkout_id or not browser_token:
        return jsonify({"ok": False, "error": "ace_checkout_credentials_required"}), 401

    record, token_err = verify_checkout_browser_token(checkout_id, browser_token)
    if token_err or record is None:
        return jsonify({"ok": False, "error": token_err or "browser_token_invalid"}), 401

    return jsonify(checkout_status_public(record)), 200


@ace_payment_bp.route("/api/payments/icount/ipn", methods=["POST"])
def ace_icount_ipn():
    config_err = assert_protected_route_security_config(include_ok=False)
    if config_err is not None:
        return Response("CONFIG_INVALID", status=503, mimetype="text/plain")
    if is_security_explicitly_disabled():
        return Response("DISABLED", status=503, mimetype="text/plain")

    redis_err = assert_security_redis_available()
    if redis_err is not None:
        return Response("UNAVAILABLE", status=503, mimetype="text/plain")

    form_fields = extract_icount_form_fields(request.form or {})
    if not form_fields and request.get_data():
        # iCount may send urlencoded body without Flask form parsing in tests
        try:
            from urllib.parse import parse_qs

            parsed = parse_qs(request.get_data(as_text=True), keep_blank_values=True)
            form_fields = {k: (v[0] if v else "") for k, v in parsed.items()}
        except Exception:
            form_fields = {}

    status, body, _reason = process_icount_ipn(
        form_fields=form_fields,
        headers={k: v for k, v in request.headers.items()},
    )
    return Response(body, status=status, mimetype="text/plain")
