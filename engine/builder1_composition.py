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
        # TODO: For square, choose below/beside by visual orientation (horizontal -> below, vertical -> beside).
        layout = "headline_below_visual"

    if layout == "headline_below_visual":
        visual_weight = 0.72
        headline_weight = 0.28
    else:
        visual_weight = 0.5
        headline_weight = 0.5

    return {
        "compositionLayout": layout,
        "headlineAlign": "center",
        "headlineLines": {"line1": line1, "line2": line2},
        "headlineRelativeSize": "max_allowed_but_not_larger_than_visual",
        "visualWeight": visual_weight,
        "headlineWeight": headline_weight,
        "safeMarginRule": "minimum_1cm_between_elements_and_edges",
        "safeMarginCss": "clamp(24px, 4vw, 48px)",
        "headlineSizeRule": "largest_possible_not_larger_than_visual",
        "productNameScale": 1.25,
        "headlineTextScale": 1.0,
        "compositionNotes": (
            "Deterministic composition from format/language with centered headline, "
            "at least 1cm safe spacing, and max headline sizing within visual dominance rules."
        ),
    }
