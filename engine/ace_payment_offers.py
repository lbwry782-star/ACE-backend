"""
Server-authoritative ACE payment offer registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


@dataclass(frozen=True)
class AcePaymentOffer:
    offer_code: str
    builder: str
    quantity: int
    amount: Decimal
    currency: str
    payment_page_url: str


_OFFERS: Dict[str, AcePaymentOffer] = {
    "b1-2": AcePaymentOffer(
        offer_code="b1-2",
        builder="builder1",
        quantity=2,
        amount=Decimal("45"),
        currency="ILS",
        payment_page_url="https://app.icount.co.il/m/78df1",
    ),
    "b1-3": AcePaymentOffer(
        offer_code="b1-3",
        builder="builder1",
        quantity=3,
        amount=Decimal("55"),
        currency="ILS",
        payment_page_url="https://app.icount.co.il/m/477e6",
    ),
    "b1-4": AcePaymentOffer(
        offer_code="b1-4",
        builder="builder1",
        quantity=4,
        amount=Decimal("65"),
        currency="ILS",
        payment_page_url="https://app.icount.co.il/m/f7c25",
    ),
    "b2-1": AcePaymentOffer(
        offer_code="b2-1",
        builder="builder2",
        quantity=1,
        amount=Decimal("80"),
        currency="ILS",
        payment_page_url=(
            "https://app.icount.co.il/m/8ca25/c6937615p12u6a98a2e"
            "?utm_source=iCount&utm_medium=paypage&utm_campaign=18"
        ),
    ),
    "b2-2": AcePaymentOffer(
        offer_code="b2-2",
        builder="builder2",
        quantity=2,
        amount=Decimal("140"),
        currency="ILS",
        payment_page_url=(
            "https://app.icount.co.il/m/0a7c0/c6937615p17u6a98a23"
            "?utm_source=iCount&utm_medium=paypage&utm_campaign=23"
        ),
    ),
}


def get_offer(offer_code: str) -> Optional[AcePaymentOffer]:
    return _OFFERS.get(str(offer_code or "").strip())


def list_offer_codes() -> Tuple[str, ...]:
    return tuple(_OFFERS.keys())


def offer_public_dict(offer: AcePaymentOffer) -> dict:
    return {
        "offerCode": offer.offer_code,
        "builder": offer.builder,
        "quantity": offer.quantity,
        "amount": format(offer.amount, "f"),
        "currency": offer.currency,
    }


def append_payment_reference_to_url(base_url: str, payment_reference: str) -> str:
    """Append m__ace_checkout_id while preserving existing query parameters."""
    parsed = urlparse(base_url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_pairs.append(("m__ace_checkout_id", payment_reference))
    new_query = urlencode(query_pairs)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        )
    )


def amounts_equal(received: str, expected: Decimal) -> bool:
    try:
        received_dec = Decimal(str(received).strip())
    except Exception:
        return False
    return received_dec == expected
