"""
Structured product modality classification for Builder1 compliance adjudication.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Sequence


class ProductModality(str, Enum):
    PHYSICAL_PRODUCT = "PHYSICAL_PRODUCT"
    DIGITAL_PRODUCT = "DIGITAL_PRODUCT"
    SERVICE = "SERVICE"
    ORGANIZATION = "ORGANIZATION"
    PLACE = "PLACE"
    EVENT = "EVENT"




_DIGITAL_PATTERNS = re.compile(
    r"\b(app|application|software|platform|saas|api|cloud|digital agent|ai agent|"
    r"virtual assistant|dashboard|browser|online tool|web service)\b",
    re.IGNORECASE,
)
_EVENT_PATTERNS = re.compile(
    r"\b(event|festival|conference|concert|exhibition|tournament)\b",
    re.IGNORECASE,
)
_PLACE_PATTERNS = re.compile(
    r"\b(hotel|restaurant|store|venue|city|resort|campus|location)\b",
    re.IGNORECASE,
)
_ORGANIZATION_PATTERNS = re.compile(
    r"\b(company|brand|organization|nonprofit|foundation|institution)\b",
    re.IGNORECASE,
)
_PHYSICAL_PATTERNS = re.compile(
    r"\b(shoe|sneaker|bottle|can|box|device|phone|watch|bag|jar|carton|"
    r"packaging|product unit|hardware|appliance|food|drink|cosmetic|tool|book)\b",
    re.IGNORECASE,
)

# Professional / instructional service roles — checked before digital/physical/generic service.
_PRIMARY_SERVICE_PATTERNS = re.compile(
    r"\b("
    r"tutor(?:ing)?|"
    r"private teacher|"
    r"teacher|"
    r"instructor|"
    r"coach(?:ing)?|"
    r"consultant|"
    r"consulting|"
    r"attorney|"
    r"lawyer|"
    r"accountant|"
    r"therapist|"
    r"designer|"
    r"photographer|"
    r"technician|"
    r"repair service|"
    r"cleaning service|"
    r"lessons|"
    r"classes|"
    r"training|"
    r"professional service|"
    r"law firm|"
    r"consulting firm"
    r")\b",
    re.IGNORECASE,
)

# Generic service nouns — checked after explicit physical-product evidence.
_GENERIC_SERVICE_PATTERNS = re.compile(
    r"\b(service|agency|support|subscription|membership|insurance|"
    r"banking|delivery service|maintenance)\b",
    re.IGNORECASE,
)

_HEBREW_ORGANIZATION_PHRASES: tuple[str, ...] = (
    "בית ספר",
    "ארגון",
    "עמותה",
    "מוסד",
    "חברה",
)

_HEBREW_DIGITAL_PHRASES: tuple[str, ...] = (
    "אפליקציה",
    "אפליקציות",
    "תוכנה",
    "פלטפורמה",
    "מערכת מקוונת",
    "סוכן דיגיטלי",
    "סוכן ai",
)

_HEBREW_PHYSICAL_PHRASES: tuple[str, ...] = (
    "ספר לימוד",
    "ספר",
    "בקבוק",
    "מזון",
    "שתייה",
    "אריזה",
    "מוצר פיזי",
)

_HEBREW_PRIMARY_SERVICE_PHRASES: tuple[str, ...] = (
    "מורה פרטי",
    "מורה פרטית",
    "שיעורים פרטיים",
    "שיעור פרטי",
    "הכנה לבגרות",
    "עורך דין",
    "עורכת דין",
    "רואה חשבון",
    "שירותי",
)

_HEBREW_PRIMARY_SERVICE_WORDS: tuple[str, ...] = (
    "מורה",
    "מלמד",
    "מלמדת",
    "מדריך",
    "מדריכה",
    "מאמן",
    "מאמנת",
    "יועץ",
    "יועצת",
    "מטפל",
    "מטפלת",
    "צלם",
    "צלמת",
    "מעצב",
    "מעצבת",
    "טכנאי",
    "תיקונים",
    "ניקיון",
    "שירות",
)


def _hebrew_boundary_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.casefold())
    return re.compile(
        rf"(?<![\u0590-\u05FF]){escaped}(?![\u0590-\u05FF])",
        re.UNICODE,
    )


def _hebrew_patterns(phrases: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(_hebrew_boundary_pattern(phrase) for phrase in phrases)


_HEBREW_ORGANIZATION_RES = _hebrew_patterns(_HEBREW_ORGANIZATION_PHRASES)
_HEBREW_DIGITAL_RES = _hebrew_patterns(_HEBREW_DIGITAL_PHRASES)
_HEBREW_PHYSICAL_RES = _hebrew_patterns(_HEBREW_PHYSICAL_PHRASES)
_HEBREW_PRIMARY_SERVICE_PHRASE_RES = _hebrew_patterns(_HEBREW_PRIMARY_SERVICE_PHRASES)
_HEBREW_PRIMARY_SERVICE_WORD_RES = _hebrew_patterns(_HEBREW_PRIMARY_SERVICE_WORDS)


def _contains_hebrew_pattern(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    lowered = text.casefold()
    return any(pattern.search(lowered) for pattern in patterns)


def _matches_hebrew_organization(text: str) -> bool:
    return _contains_hebrew_pattern(text, _HEBREW_ORGANIZATION_RES)


def _matches_hebrew_digital(text: str) -> bool:
    return _contains_hebrew_pattern(text, _HEBREW_DIGITAL_RES)


def _matches_hebrew_physical(text: str) -> bool:
    return _contains_hebrew_pattern(text, _HEBREW_PHYSICAL_RES)


def _matches_hebrew_primary_service(text: str) -> bool:
    lowered = text.casefold()
    for pattern in _HEBREW_PRIMARY_SERVICE_PHRASE_RES:
        if pattern.search(lowered):
            return True
    for pattern in _HEBREW_PRIMARY_SERVICE_WORD_RES:
        if pattern.search(lowered):
            return True
    return False


def _matches_primary_service(text: str) -> bool:
    return bool(_PRIMARY_SERVICE_PATTERNS.search(text)) or _matches_hebrew_primary_service(text)


def derive_product_modality(*, product_name: str = "", product_description: str = "") -> ProductModality:
    text = f"{product_name} {product_description}".strip().lower()
    if not text:
        return ProductModality.PHYSICAL_PRODUCT
    if _EVENT_PATTERNS.search(text):
        return ProductModality.EVENT
    if _PLACE_PATTERNS.search(text):
        return ProductModality.PLACE
    if _ORGANIZATION_PATTERNS.search(text) or _matches_hebrew_organization(text):
        return ProductModality.ORGANIZATION
    if _DIGITAL_PATTERNS.search(text) or _matches_hebrew_digital(text):
        return ProductModality.DIGITAL_PRODUCT
    if _PHYSICAL_PATTERNS.search(text) or _matches_hebrew_physical(text):
        return ProductModality.PHYSICAL_PRODUCT
    if _matches_primary_service(text):
        return ProductModality.SERVICE
    if _GENERIC_SERVICE_PATTERNS.search(text):
        return ProductModality.SERVICE
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
