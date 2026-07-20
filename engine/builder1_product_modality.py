"""
Structured product modality classification for Builder1 compliance adjudication.
"""
from __future__ import annotations

import re
from enum import Enum


class ProductModality(str, Enum):
    PHYSICAL_PRODUCT = "PHYSICAL_PRODUCT"
    DIGITAL_PRODUCT = "DIGITAL_PRODUCT"
    SERVICE = "SERVICE"
    ORGANIZATION = "ORGANIZATION"
    PLACE = "PLACE"
    EVENT = "EVENT"


_DIGITAL_PATTERNS = (
    r"\b(app|application|software|platform|saas|api|cloud|digital agent|ai agent|"
    r"virtual assistant|dashboard|browser|online tool|web service)\b"
)
_SERVICE_PATTERNS = (
    r"\b(service|consulting|agency|support|subscription|membership|insurance|"
    r"banking|delivery service|maintenance|training)\b"
)
_ORGANIZATION_PATTERNS = r"\b(company|brand|organization|nonprofit|foundation|institution)\b"
_PLACE_PATTERNS = r"\b(hotel|restaurant|store|venue|city|resort|campus|location)\b"
_EVENT_PATTERNS = r"\b(event|festival|conference|concert|exhibition|tournament)\b"
_PHYSICAL_PATTERNS = (
    r"\b(shoe|sneaker|bottle|can|box|device|phone|watch|bag|jar|carton|"
    r"packaging|product unit|hardware|appliance|food|drink|cosmetic|tool)\b"
)


def derive_product_modality(*, product_name: str = "", product_description: str = "") -> ProductModality:
    text = f"{product_name} {product_description}".strip().lower()
    if not text:
        return ProductModality.PHYSICAL_PRODUCT
    if re.search(_EVENT_PATTERNS, text):
        return ProductModality.EVENT
    if re.search(_PLACE_PATTERNS, text):
        return ProductModality.PLACE
    if re.search(_ORGANIZATION_PATTERNS, text):
        return ProductModality.ORGANIZATION
    if re.search(_DIGITAL_PATTERNS, text):
        return ProductModality.DIGITAL_PRODUCT
    if re.search(_SERVICE_PATTERNS, text):
        return ProductModality.SERVICE
    if re.search(_PHYSICAL_PATTERNS, text):
        return ProductModality.PHYSICAL_PRODUCT
    if any(token in text for token in ("digital", "online", "virtual", "agent", "automation")):
        return ProductModality.DIGITAL_PRODUCT
    return ProductModality.PHYSICAL_PRODUCT


def resolve_product_modality(
    *,
    product_name: str = "",
    product_description: str = "",
    planning_internals: object = None,
) -> ProductModality:
    if isinstance(planning_internals, dict):
        raw = str(planning_internals.get("productModality") or "").strip().upper()
        if raw:
            try:
                return ProductModality(raw)
            except ValueError:
                pass
    return derive_product_modality(
        product_name=product_name,
        product_description=product_description,
    )
