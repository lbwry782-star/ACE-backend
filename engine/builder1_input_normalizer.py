"""
Disconnected Builder1 input normalization helpers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalizedBuilder1Input:
    product_name: str
    product_description: str
    format: str


class Builder1InputError(ValueError):
    pass


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


def normalize_builder1_input(
    product_name: object, product_description: object, format_value: object
) -> NormalizedBuilder1Input:
    normalized_name = normalize_text(product_name)
    normalized_description = normalize_text(product_description)
    normalized_format = normalize_format(format_value)
    if not normalized_description:
        raise Builder1InputError("missing_product_description")
    return NormalizedBuilder1Input(
        product_name=normalized_name,
        product_description=normalized_description,
        format=normalized_format,
    )
