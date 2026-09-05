"""
iCount production IPN validation — server-side only payment proof.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from engine.ace_payment_checkout_store import mark_checkout_paid, resolve_checkout_by_payment_reference
from engine.ace_payment_offers import amounts_equal, get_offer
from engine.ace_payment_tokens import load_icount_ipn_secret_sha256_from_env, verify_icount_ipn_secret

logger = logging.getLogger(__name__)

ICOUNT_SECRET_HEADER = "X-Icount-Secret"
PAYMENT_REFERENCE_FIELD = "ace_checkout_id"


def extract_icount_form_fields(payload: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        out[str(key)] = str(value).strip()
    return out


def verify_icount_ipn_secret_header(headers: Mapping[str, str]) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns (present, verified, reject_code).
    Never logs raw header value.
    """
    expected = load_icount_ipn_secret_sha256_from_env()
    if not expected:
        return False, False, "ipn_secret_not_configured"

    raw = ""
    for key, value in headers.items():
        if str(key).casefold() == ICOUNT_SECRET_HEADER.casefold():
            raw = str(value or "")
            break

    present = bool(raw)
    if not present:
        return False, False, "ipn_secret_missing"

    verified = verify_icount_ipn_secret(raw, expected)
    if not verified:
        return True, False, "ipn_secret_invalid"
    return True, True, None


def validate_icount_ipn_payload(
    form_fields: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate correlation + offer fields. Returns (checkout_id, reject_code).
    """
    fields = extract_icount_form_fields(form_fields)
    payment_reference = fields.get(PAYMENT_REFERENCE_FIELD) or fields.get("custom[ace_checkout_id]", "")
    if not payment_reference:
        return None, "payment_reference_missing"

    checkout_data = resolve_checkout_by_payment_reference(payment_reference)
    if not checkout_data:
        return None, "payment_reference_unknown"

    checkout_id = str(checkout_data.get("checkoutId") or "")
    offer = get_offer(str(checkout_data.get("offerCode") or ""))
    if offer is None:
        return checkout_id, "offer_not_found"

    sum_value = fields.get("sum") or fields.get("total_paid") or ""
    currency_code = (fields.get("currency_code") or "").strip().upper()
    doctype = (fields.get("doctype") or "").strip()
    docnum = (fields.get("docnum") or fields.get("docnum_id") or "").strip()

    if not doctype or not docnum:
        return checkout_id, "document_identity_missing"

    if currency_code != "ILS":
        return checkout_id, "currency_mismatch"

    if not amounts_equal(sum_value, offer.amount):
        total_paid = fields.get("total_paid")
        if total_paid and amounts_equal(total_paid, offer.amount):
            sum_value = total_paid
        else:
            return checkout_id, "amount_mismatch"

    expected_amount = str(checkout_data.get("expectedAmount") or "")
    if expected_amount and not amounts_equal(sum_value, offer.amount):
        return checkout_id, "checkout_amount_mismatch"

    accepted, code = mark_checkout_paid(
        checkout_id=checkout_id,
        doctype=doctype,
        docnum=docnum,
        sum_value=sum_value,
        currency_code=currency_code,
        confirmation_code=fields.get("confirmation_code", ""),
        cp=fields.get("cp", ""),
        total_paid=fields.get("total_paid", ""),
    )
    if not accepted:
        return checkout_id, code

    return checkout_id, None


def process_icount_ipn(
    *,
    form_fields: Mapping[str, Any],
    headers: Mapping[str, str],
) -> Tuple[int, str, str]:
    """
    Full IPN pipeline. Returns (http_status, body, log_reason_code).
    """
    present, verified, secret_code = verify_icount_ipn_secret_header(headers)
    if secret_code:
        logger.warning(
            "ACE_ICOUNT_IPN_REJECTED reason=%s secretPresent=%s secretVerified=%s",
            secret_code,
            "true" if present else "false",
            "true" if verified else "false",
        )
        return 401, "REJECTED", secret_code

    checkout_id, reject_code = validate_icount_ipn_payload(form_fields)
    if reject_code:
        logger.warning(
            "ACE_ICOUNT_IPN_REJECTED reason=%s checkoutId=%s secretPresent=true secretVerified=true",
            reject_code,
            checkout_id or "",
        )
        return 400, "REJECTED", reject_code

    logger.info(
        "ACE_ICOUNT_IPN_ACCEPTED checkoutId=%s secretPresent=true secretVerified=true",
        checkout_id or "",
    )
    return 200, "OK", "accepted"
