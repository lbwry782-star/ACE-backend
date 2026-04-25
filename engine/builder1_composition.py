"""
Builder1 composition metadata (deterministic rules, no model call).
"""
from __future__ import annotations

from typing import Any


def generate_builder1_composition_o3(
    *,
    format: str,
    detectedLanguage: str,
    productNameResolved: str,
    headlineProductName: str,
    headlineText: str,
    headlineFull: str,
    objectA: str,
    objectASecondary: str,
    objectB: str,
    modeDecision: str,
    visualDescription: str,
    visualPrompt: str,
) -> dict[str, Any]:
    del productNameResolved, headlineFull, objectA, objectASecondary, objectB, modeDecision, visualDescription, visualPrompt
    line1 = (headlineProductName or "").strip()
    line2 = (headlineText or "").strip()
    if not line1:
        raise ValueError("headline_line1_missing")
    if not line2:
        raise ValueError("headline_line2_missing")

    fmt = (format or "").strip().lower()
    lang = (detectedLanguage or "").strip().lower()
    if fmt == "landscape":
        layout = "headline_left_visual_right" if lang == "he" else "visual_left_headline_right"
    elif fmt == "portrait":
        layout = "headline_below_visual"
    else:
        layout = "headline_below_visual"

    if layout == "headline_below_visual":
        visual_weight = "dominant"
        headline_weight = "secondary"
    else:
        visual_weight = "equal"
        headline_weight = "equal"

    return {
        "compositionLayout": layout,
        "headlineAlign": "center",
        "headlineLines": {"line1": line1, "line2": line2},
        "headlineRelativeSize": "max_allowed_but_not_larger_than_visual",
        "visualWeight": visual_weight,
        "headlineWeight": headline_weight,
        "safeMarginRule": "minimum_1cm",
        "compositionNotes": (
            "Deterministic safe composition: center-aligned headline, 1cm minimum margin, "
            "and layout/weight selected by format and language rules."
        ),
    }
