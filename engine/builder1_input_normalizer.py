"""
Builder1 input normalization (campaign-series).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from engine.builder1_plan_spec import AD_COUNT_MAX, AD_COUNT_MIN


@dataclass
class NormalizedBuilder1Input:
    product_name: str
    product_description: str
    format: str
    ad_count: int
    brand_guidelines: Optional[Dict[str, Any]]


class Builder1InputError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    return " ".join(s.strip().split())


def normalize_format(value: object) -> str:
    raw = normalize_text(value).lower()
    aliases = {
        "horizontal": "landscape",
        "vertical": "portrait",
        "wide": "landscape",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"landscape", "portrait", "square"}:
        raise Builder1InputError("invalid_format")
    return normalized


def normalize_ad_count(value: object) -> int:
    if value is None:
        return AD_COUNT_MIN
    if isinstance(value, bool):
        raise Builder1InputError("invalid_ad_count")
    if isinstance(value, float):
        raise Builder1InputError("invalid_ad_count")
    if isinstance(value, str):
        raise Builder1InputError("invalid_ad_count")
    if not isinstance(value, int):
        raise Builder1InputError("invalid_ad_count")
    if value < AD_COUNT_MIN or value > AD_COUNT_MAX:
        raise Builder1InputError("invalid_ad_count")
    return value


def normalize_brand_guidelines(value: object) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    return value


def normalize_builder1_input(
    product_name: object,
    product_description: object,
    format_value: object,
    *,
    ad_count: object = None,
    brand_guidelines: object = None,
) -> NormalizedBuilder1Input:
    normalized_name = normalize_text(product_name)
    normalized_description = normalize_text(product_description)
    normalized_format = normalize_format(format_value)
    normalized_ad_count = normalize_ad_count(ad_count)
    if not normalized_description:
        raise Builder1InputError("missing_product_description")
    return NormalizedBuilder1Input(
        product_name=normalized_name,
        product_description=normalized_description,
        format=normalized_format,
        ad_count=normalized_ad_count,
        brand_guidelines=normalize_brand_guidelines(brand_guidelines),
    )
