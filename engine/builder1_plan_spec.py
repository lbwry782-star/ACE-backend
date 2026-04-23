"""
Disconnected Builder1 rebuild planning spec.

This module is intentionally standalone and not wired into app routes yet.
"""
from __future__ import annotations

from dataclasses import dataclass

FORMAT_LANDSCAPE = "landscape"
FORMAT_PORTRAIT = "portrait"
FORMAT_SQUARE = "square"

MODE_SIDE_BY_SIDE = "SIDE_BY_SIDE"
MODE_REPLACEMENT = "REPLACEMENT"
REPLACEMENT_THRESHOLD = 85


@dataclass
class Builder1Input:
    """Minimal input contract for Builder1 planning stage."""

    product_name: str
    product_description: str
    format: str


@dataclass
class Builder1Plan:
    """Planning-only output contract for Builder1 stage 1."""

    product_name: str
    product_description: str
    format: str
    detected_language: str
    product_name_resolved: str
    advertising_promise: str
    object_a: str
    object_a_secondary: str
    object_b: str
    visual_similarity_score: int
    mode_decision: str
    visual_description: str


def decide_mode(similarity_score: int) -> str:
    """Return replacement mode when score meets threshold, else side-by-side."""

    return MODE_REPLACEMENT if int(similarity_score) >= REPLACEMENT_THRESHOLD else MODE_SIDE_BY_SIDE
