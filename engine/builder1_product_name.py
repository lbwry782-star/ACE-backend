"""
Builder1 product-name resolution stage (user-provided or auto-generated once per campaign).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from engine.builder1_plan_parser import _norm_key, _norm_text
from engine.builder1_staged_parsers import StageParseError, coerce_json_dict
from engine.video_language import (
    is_english_only_product_name_script,
    is_hebrew_or_english_product_name_script,
)

logger = logging.getLogger(__name__)

GENERIC_CATEGORY_LABELS = frozenset(
    {
        "shoe store",
        "restaurant",
        "coffee shop",
        "bakery",
        "hotel",
        "clothing store",
        "grocery store",
        "fitness center",
        "חנות נעליים",
        "מסעדה",
        "בית קפה",
        "מאפייה",
        "מלון",
        "חנות בגדים",
        "מרכול",
        "מכון כושר",
    }
)


def _is_generic_category_label(name: str) -> bool:
    normalized = _norm_text(name).casefold()
    if not normalized:
        return False
    if normalized in GENERIC_CATEGORY_LABELS:
        return True
    return bool(re.fullmatch(r"(?:the\s+)?(?:[\w\u0590-\u05FF]+\s+){0,2}(?:store|shop|restaurant|מסעדה|חנות)", normalized))


def _is_verbatim_description_copy(name: str, product_description: str) -> bool:
    normalized_name = _norm_key(name)
    normalized_description = _norm_key(product_description)
    if not normalized_name or not normalized_description:
        return False
    if normalized_name == normalized_description:
        return True
    if len(normalized_name) >= min(20, len(normalized_description)) and normalized_name in normalized_description:
        return True
    return False


def validate_resolved_product_name(
    name: str,
    *,
    product_description: str,
    detected_language: str,
) -> List[str]:
    reasons: List[str] = []
    resolved = _norm_text(name)
    if not resolved:
        reasons.append("product_name_resolution_empty")
        return reasons
    if _is_verbatim_description_copy(resolved, product_description):
        reasons.append("product_name_resolution_copied_description")
    if _is_generic_category_label(resolved):
        reasons.append("product_name_resolution_generic_category")
    if detected_language == "en" and not is_english_only_product_name_script(resolved):
        reasons.append("product_name_resolution_wrong_language")
    elif detected_language == "he" and not is_hebrew_or_english_product_name_script(resolved):
        reasons.append("product_name_resolution_wrong_language")
    return reasons


def parse_product_name_resolution(
    raw_payload: object,
    *,
    product_description: str,
    detected_language: str,
) -> str:
    reasons: List[str] = []
    try:
        obj = coerce_json_dict(raw_payload)
    except Exception as exc:
        raise StageParseError("product_name_resolution", ["product_name_resolution_not_object"]) from exc

    resolved = _norm_text(obj.get("productNameResolved"))
    if not resolved:
        reasons.append("product_name_resolution_missing_field")
    reasons.extend(
        validate_resolved_product_name(
            resolved,
            product_description=product_description,
            detected_language=detected_language,
        )
    )
    if reasons:
        raise StageParseError("product_name_resolution", reasons)
    return resolved


def enforce_authoritative_product_name(
    brand_physical: Any,
    *,
    product_name_resolved: str,
) -> Any:
    from dataclasses import replace

    return replace(brand_physical, product_name_resolved=product_name_resolved)
