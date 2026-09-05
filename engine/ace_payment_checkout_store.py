"""
Redis-backed ACE payment checkout state — multi-user safe, atomic transitions.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from engine.ace_payment_offers import AcePaymentOffer, get_offer, offer_public_dict
from engine.ace_payment_tokens import (
    generate_browser_token,
    generate_payment_reference,
    sha256_hex,
    verify_browser_token,
)
from engine.video_jobs_redis import get_redis, redis_configured

logger = logging.getLogger(__name__)

CHECKOUT_KEY_PREFIX = "ace:security:checkout:"
PAYMENT_REF_KEY_PREFIX = "ace:security:paymentref:"
ICOUNT_DOC_KEY_PREFIX = "ace:security:icountdoc:"

PENDING_TTL_SECONDS = 48 * 3600
PAID_BOUND_TTL_SECONDS = 30 * 24 * 3600

CHECKOUT_STATE_PENDING = "pending"
CHECKOUT_STATE_PAID = "paid"
CHECKOUT_STATE_BOUND = "bound"

_memory_lock = threading.Lock()
_memory_checkouts: Dict[str, Dict[str, Any]] = {}
_memory_payment_refs: Dict[str, str] = {}
_memory_docs: Dict[str, str] = {}
_use_memory_store = False


class AceCheckoutStoreError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AceCheckoutRecord:
    checkout_id: str
    offer_code: str
    builder: str
    purchased_quantity: int
    expected_amount: str
    expected_currency: str
    state: str
    browser_token_hash: str
    payment_reference_hash: str
    created_at: str
    paid_at: str = ""
    bound_resource_id: str = ""
    bound_resource_type: str = ""
    bound_at: str = ""
    doc_doctype: str = ""
    doc_docnum: str = ""
    doc_confirmation_code: str = ""
    doc_cp: str = ""
    doc_sum: str = ""
    doc_currency_code: str = ""


def configure_checkout_memory_store(enabled: bool) -> None:
    global _use_memory_store
    _use_memory_store = bool(enabled)


def clear_memory_checkout_store_for_tests() -> None:
    with _memory_lock:
        _memory_checkouts.clear()
        _memory_payment_refs.clear()
        _memory_docs.clear()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkout_key(checkout_id: str) -> str:
    return f"{CHECKOUT_KEY_PREFIX}{checkout_id}"


def _payment_ref_key(payment_reference_hash: str) -> str:
    return f"{PAYMENT_REF_KEY_PREFIX}{payment_reference_hash}"


def _doc_key(doctype: str, docnum: str) -> str:
    return f"{ICOUNT_DOC_KEY_PREFIX}{(doctype or '').strip()}:{str(docnum or '').strip()}"


def _record_to_dict(record: AceCheckoutRecord) -> Dict[str, Any]:
    return {
        "checkoutId": record.checkout_id,
        "offerCode": record.offer_code,
        "builder": record.builder,
        "purchasedQuantity": record.purchased_quantity,
        "expectedAmount": record.expected_amount,
        "expectedCurrency": record.expected_currency,
        "state": record.state,
        "browserTokenHash": record.browser_token_hash,
        "paymentReferenceHash": record.payment_reference_hash,
        "createdAt": record.created_at,
        "paidAt": record.paid_at,
        "boundResourceId": record.bound_resource_id,
        "boundResourceType": record.bound_resource_type,
        "boundAt": record.bound_at,
        "docDoctype": record.doc_doctype,
        "docDocnum": record.doc_docnum,
        "docConfirmationCode": record.doc_confirmation_code,
        "docCp": record.doc_cp,
        "docSum": record.doc_sum,
        "docCurrencyCode": record.doc_currency_code,
    }


def _dict_to_record(data: Dict[str, Any]) -> AceCheckoutRecord:
    return AceCheckoutRecord(
        checkout_id=str(data.get("checkoutId") or ""),
        offer_code=str(data.get("offerCode") or ""),
        builder=str(data.get("builder") or ""),
        purchased_quantity=int(data.get("purchasedQuantity") or 0),
        expected_amount=str(data.get("expectedAmount") or ""),
        expected_currency=str(data.get("expectedCurrency") or "ILS"),
        state=str(data.get("state") or CHECKOUT_STATE_PENDING),
        browser_token_hash=str(data.get("browserTokenHash") or ""),
        payment_reference_hash=str(data.get("paymentReferenceHash") or ""),
        created_at=str(data.get("createdAt") or ""),
        paid_at=str(data.get("paidAt") or ""),
        bound_resource_id=str(data.get("boundResourceId") or ""),
        bound_resource_type=str(data.get("boundResourceType") or ""),
        bound_at=str(data.get("boundAt") or ""),
        doc_doctype=str(data.get("docDoctype") or ""),
        doc_docnum=str(data.get("docDocnum") or ""),
        doc_confirmation_code=str(data.get("docConfirmationCode") or ""),
        doc_cp=str(data.get("docCp") or ""),
        doc_sum=str(data.get("docSum") or ""),
        doc_currency_code=str(data.get("docCurrencyCode") or ""),
    )


def _save_checkout(data: Dict[str, Any], *, ttl_seconds: int) -> None:
    checkout_id = str(data.get("checkoutId") or "")
    if not checkout_id:
        raise AceCheckoutStoreError("invalid_checkout")
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    ref_hash = str(data.get("paymentReferenceHash") or "")
    if _use_memory_store or not redis_configured():
        with _memory_lock:
            _memory_checkouts[checkout_id] = deepcopy(data)
            if ref_hash:
                _memory_payment_refs[ref_hash] = checkout_id
        return
    r = get_redis()
    pipe = r.pipeline()
    pipe.set(_checkout_key(checkout_id), raw, ex=ttl_seconds)
    if ref_hash:
        pipe.set(_payment_ref_key(ref_hash), checkout_id, ex=ttl_seconds)
    pipe.execute()


def _load_checkout(checkout_id: str) -> Optional[Dict[str, Any]]:
    cid = (checkout_id or "").strip()
    if not cid:
        return None
    if _use_memory_store or not redis_configured():
        with _memory_lock:
            data = _memory_checkouts.get(cid)
            return deepcopy(data) if data else None
    raw = get_redis().get(_checkout_key(cid))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def resolve_checkout_by_payment_reference(payment_reference: str) -> Optional[Dict[str, Any]]:
    ref = (payment_reference or "").strip()
    if not ref:
        return None
    ref_hash = sha256_hex(ref)
    checkout_id = None
    if _use_memory_store or not redis_configured():
        with _memory_lock:
            checkout_id = _memory_payment_refs.get(ref_hash)
    else:
        checkout_id = get_redis().get(_payment_ref_key(ref_hash))
    if not checkout_id:
        return None
    return _load_checkout(str(checkout_id))


def create_checkout(offer: AcePaymentOffer) -> Tuple[AceCheckoutRecord, str, str]:
    checkout_id = str(uuid.uuid4())
    browser_token = generate_browser_token()
    payment_reference = generate_payment_reference()
    now = _utc_now_iso()
    record = AceCheckoutRecord(
        checkout_id=checkout_id,
        offer_code=offer.offer_code,
        builder=offer.builder,
        purchased_quantity=offer.quantity,
        expected_amount=format(offer.amount, "f"),
        expected_currency=offer.currency,
        state=CHECKOUT_STATE_PENDING,
        browser_token_hash=sha256_hex(browser_token),
        payment_reference_hash=sha256_hex(payment_reference),
        created_at=now,
    )
    _save_checkout(_record_to_dict(record), ttl_seconds=PENDING_TTL_SECONDS)
    logger.info(
        "ACE_CHECKOUT_CREATED checkoutId=%s offerCode=%s builder=%s quantity=%s",
        checkout_id,
        offer.offer_code,
        offer.builder,
        offer.quantity,
    )
    return record, browser_token, payment_reference


def get_checkout_record(checkout_id: str) -> Optional[AceCheckoutRecord]:
    data = _load_checkout(checkout_id)
    if not data:
        return None
    return _dict_to_record(data)


def verify_checkout_browser_token(checkout_id: str, browser_token: str) -> Tuple[Optional[AceCheckoutRecord], Optional[str]]:
    record = get_checkout_record(checkout_id)
    if record is None:
        return None, "checkout_not_found"
    if not verify_browser_token(browser_token, record.browser_token_hash):
        return None, "browser_token_invalid"
    return record, None


def checkout_status_public(record: AceCheckoutRecord) -> dict:
    return {
        "state": record.state,
        "offerCode": record.offer_code,
        "builder": record.builder,
        "quantity": record.purchased_quantity,
    }


def mark_checkout_paid(
    *,
    checkout_id: str,
    doctype: str,
    docnum: str,
    sum_value: str,
    currency_code: str,
    confirmation_code: str = "",
    cp: str = "",
    total_paid: str = "",
) -> Tuple[bool, str]:
    """
    Atomically mark checkout paid. Returns (accepted, code).
    accepted=True for success or idempotent replay of same document.
    """
    data = _load_checkout(checkout_id)
    if not data:
        return False, "checkout_not_found"

    doctype_clean = (doctype or "").strip()
    docnum_clean = str(docnum or "").strip()
    doc_identity = _doc_key(doctype_clean, docnum_clean)

    state = str(data.get("state") or "")
    if state in {CHECKOUT_STATE_PAID, CHECKOUT_STATE_BOUND}:
        if (
            str(data.get("docDoctype") or "") == doctype_clean
            and str(data.get("docDocnum") or "") == docnum_clean
        ):
            return True, "idempotent"
        return False, "checkout_already_paid"

    if state != CHECKOUT_STATE_PENDING:
        return False, "invalid_checkout_state"

    if _use_memory_store or not redis_configured():
        with _memory_lock:
            existing_doc_owner = _memory_docs.get(doc_identity)
            if existing_doc_owner and existing_doc_owner != checkout_id:
                return False, "document_already_claimed"
            data["state"] = CHECKOUT_STATE_PAID
            data["paidAt"] = _utc_now_iso()
            data["docDoctype"] = doctype_clean
            data["docDocnum"] = docnum_clean
            data["docConfirmationCode"] = (confirmation_code or "").strip()
            data["docCp"] = str(cp or "").strip()
            data["docSum"] = str(sum_value or "").strip()
            data["docCurrencyCode"] = (currency_code or "").strip()
            if total_paid:
                data["docTotalPaid"] = str(total_paid).strip()
            _memory_checkouts[checkout_id] = deepcopy(data)
            _memory_docs[doc_identity] = checkout_id
        logger.info("ACE_CHECKOUT_PAID checkoutId=%s offerCode=%s", checkout_id, data.get("offerCode"))
        return True, "paid"

    r = get_redis()
    claim_script = r.register_script(
        """
local checkout_raw = redis.call('GET', KEYS[1])
if not checkout_raw then
  return cjson.encode({ok=false, code='checkout_not_found'})
end
local data = cjson.decode(checkout_raw)
local state = data.state or ''
local doctype = ARGV[1]
local docnum = ARGV[2]
if state == 'paid' or state == 'bound' then
  if (data.docDoctype or '') == doctype and tostring(data.docDocnum or '') == docnum then
    return cjson.encode({ok=true, code='idempotent'})
  end
  return cjson.encode({ok=false, code='checkout_already_paid'})
end
if state ~= 'pending' then
  return cjson.encode({ok=false, code='invalid_checkout_state'})
end
local doc_owner = redis.call('GET', KEYS[2])
if doc_owner and doc_owner ~= ARGV[3] then
  return cjson.encode({ok=false, code='document_already_claimed'})
end
data.state = 'paid'
data.paidAt = ARGV[4]
data.docDoctype = doctype
data.docDocnum = docnum
data.docConfirmationCode = ARGV[5]
data.docCp = ARGV[6]
data.docSum = ARGV[7]
data.docCurrencyCode = ARGV[8]
if ARGV[9] ~= '' then
  data.docTotalPaid = ARGV[9]
end
local encoded = cjson.encode(data)
redis.call('SET', KEYS[1], encoded, 'EX', tonumber(ARGV[10]))
redis.call('SET', KEYS[2], ARGV[3], 'EX', tonumber(ARGV[10]))
return cjson.encode({ok=true, code='paid'})
"""
    )
    result_raw = claim_script(
        keys=[_checkout_key(checkout_id), doc_identity],
        args=[
            doctype_clean,
            docnum_clean,
            checkout_id,
            _utc_now_iso(),
            (confirmation_code or "").strip(),
            str(cp or "").strip(),
            str(sum_value or "").strip(),
            (currency_code or "").strip(),
            str(total_paid or "").strip(),
            str(PAID_BOUND_TTL_SECONDS),
        ],
    )
    try:
        result = json.loads(result_raw)
    except json.JSONDecodeError:
        return False, "checkout_store_error"
    if not result.get("ok"):
        return False, str(result.get("code") or "checkout_store_error")
    logger.info("ACE_CHECKOUT_PAID checkoutId=%s offerCode=%s", checkout_id, data.get("offerCode"))
    return True, str(result.get("code") or "paid")


def bind_checkout_to_resource(
    *,
    checkout_id: str,
    resource_id: str,
    resource_type: str,
) -> Tuple[bool, str]:
    resource_id_clean = (resource_id or "").strip()
    resource_type_clean = (resource_type or "").strip()
    if not resource_id_clean or not resource_type_clean:
        return False, "invalid_resource"

    bound_at = _utc_now_iso()

    if _use_memory_store or not redis_configured():
        return _bind_checkout_to_resource_memory(
            checkout_id=checkout_id,
            resource_id=resource_id_clean,
            resource_type=resource_type_clean,
            bound_at=bound_at,
        )

    r = get_redis()
    bind_script = r.register_script(
        """
local checkout_raw = redis.call('GET', KEYS[1])
if not checkout_raw then
  return cjson.encode({ok=false, code='checkout_not_found'})
end
local data = cjson.decode(checkout_raw)
local state = data.state or ''
local resource_id = ARGV[1]
local resource_type = ARGV[2]
local bound_at = ARGV[3]
local ttl = tonumber(ARGV[4])

if state == 'bound' then
  if (data.boundResourceId or '') == resource_id and (data.boundResourceType or '') == resource_type then
    return cjson.encode({ok=true, code='idempotent'})
  end
  return cjson.encode({ok=false, code='checkout_already_bound'})
end

if state ~= 'paid' then
  return cjson.encode({ok=false, code='checkout_not_paid'})
end

data.state = 'bound'
data.boundResourceId = resource_id
data.boundResourceType = resource_type
data.boundAt = bound_at
local encoded = cjson.encode(data)
redis.call('SET', KEYS[1], encoded, 'EX', ttl)
return cjson.encode({ok=true, code='bound'})
"""
    )
    result_raw = bind_script(
        keys=[_checkout_key(checkout_id)],
        args=[
            resource_id_clean,
            resource_type_clean,
            bound_at,
            str(PAID_BOUND_TTL_SECONDS),
        ],
    )
    try:
        result = json.loads(result_raw)
    except json.JSONDecodeError:
        return False, "checkout_store_error"
    if not result.get("ok"):
        return False, str(result.get("code") or "checkout_store_error")
    logger.info(
        "ACE_ENTITLEMENT_BOUND checkoutId=%s resourceType=%s resourceId=%s",
        checkout_id,
        resource_type_clean,
        resource_id_clean,
    )
    return True, str(result.get("code") or "bound")


def _bind_checkout_to_resource_memory(
    *,
    checkout_id: str,
    resource_id: str,
    resource_type: str,
    bound_at: str,
) -> Tuple[bool, str]:
    """In-memory bind with identical semantics to Redis Lua (single lock, atomic decision)."""
    with _memory_lock:
        data = _memory_checkouts.get(checkout_id)
        if not data:
            return False, "checkout_not_found"

        state = str(data.get("state") or "")
        if state == CHECKOUT_STATE_BOUND:
            if (
                str(data.get("boundResourceId") or "") == resource_id
                and str(data.get("boundResourceType") or "") == resource_type
            ):
                return True, "idempotent"
            return False, "checkout_already_bound"

        if state != CHECKOUT_STATE_PAID:
            return False, "checkout_not_paid"

        data = deepcopy(data)
        data["state"] = CHECKOUT_STATE_BOUND
        data["boundResourceId"] = resource_id
        data["boundResourceType"] = resource_type
        data["boundAt"] = bound_at
        _memory_checkouts[checkout_id] = data

    logger.info(
        "ACE_ENTITLEMENT_BOUND checkoutId=%s resourceType=%s resourceId=%s",
        checkout_id,
        resource_type,
        resource_id,
    )
    return True, "bound"


def checkout_allows_builder(record: AceCheckoutRecord, builder: str) -> bool:
    return record.builder == builder and record.state in {CHECKOUT_STATE_PAID, CHECKOUT_STATE_BOUND}


def checkout_bound_to_resource(record: AceCheckoutRecord, resource_id: str) -> bool:
    if record.state != CHECKOUT_STATE_BOUND:
        return False
    return str(record.bound_resource_id or "") == str(resource_id or "").strip()
