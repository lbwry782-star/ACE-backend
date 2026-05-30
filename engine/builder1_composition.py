"""
Builder1 composition metadata (deterministic rules, no model call).
"""
from __future__ import annotations

from typing import Any

# Substrings (ASCII) matched case-insensitively against combined plan/visual text.
_SQUARE_VERTICAL_HINTS_EN: tuple[str, ...] = (
    "vertical",
    "portrait",
    "upright",
    "narrow",
    "column",
    "skyscraper",
    "tower",
    "tall ",
    " tall",
    " tall,",
    "full-height",
    "phone portrait",
    "portrait mode",
    "vertical format",
    "portrait-oriented",
    "portrait oriented",
    "standing upright",
    "upright bottle",
    "upright can",
    "upright pack",
)
_SQUARE_HORIZONTAL_HINTS_EN: tuple[str, ...] = (
    "horizontal",
    "landscape",
    "panoramic",
    "banner",
    "wide ",
    " wide",
    " wide,",
    "widescreen",
    "horizontal format",
    "landscape-oriented",
    "landscape oriented",
    "lying flat",
    "flat lay",
    "flat-lay",
    "tabletop",
    "wide shot",
    "establishing wide",
    "spanning",
)
# Hebrew substrings matched on the original (non-lowercased) blob.
_SQUARE_VERTICAL_HINTS_HE: tuple[str, ...] = (
    "אנכי",
    "אנכית",
    "אנכיים",
    "באנכי",
    "לאורך",
    "גבוה",
    "גבוהה",
    "גבוהים",
)
_SQUARE_HORIZONTAL_HINTS_HE: tuple[str, ...] = (
    "אופקי",
    "אופקית",
    "אופקיים",
    "רוחבי",
    "רוחבית",
    "רחב",
    "רחבה",
    "רחבים",
    "לרוחב",
)


def _square_vertical_signal_scores(
    *,
    visual_description: str,
    object_a: str,
    object_b: str,
    mode_decision: str,
    visual_prompt: str,
) -> tuple[int, int, str]:
    """Return (vertical_hits, horizontal_hits, short_reason) for square orientation."""

    parts = [
        visual_description or "",
        object_a or "",
        object_b or "",
        mode_decision or "",
        visual_prompt or "",
    ]
    blob_full = " ".join(parts)
    blob_en = blob_full.lower()

    v_en = sum(1 for h in _SQUARE_VERTICAL_HINTS_EN if h in blob_en)
    h_en = sum(1 for h in _SQUARE_HORIZONTAL_HINTS_EN if h in blob_en)
    v_he = sum(1 for h in _SQUARE_VERTICAL_HINTS_HE if h in blob_full)
    h_he = sum(1 for h in _SQUARE_HORIZONTAL_HINTS_HE if h in blob_full)

    vertical = v_en + v_he
    horizontal = h_en + h_he

    if vertical > horizontal and vertical > 0:
        reason = f"vertical_keywords={vertical}>horizontal_keywords={horizontal}"
    elif horizontal > vertical:
        reason = f"horizontal_keywords={horizontal}>vertical_keywords={vertical}"
    else:
        reason = "no_clear_vertical_bias_default_horizontal_below"

    return vertical, horizontal, reason


def _square_treat_as_vertical_visual(
    *,
    visual_description: str,
    object_a: str,
    object_b: str,
    mode_decision: str,
    visual_prompt: str,
) -> tuple[bool, str]:
    vertical, horizontal, reason = _square_vertical_signal_scores(
        visual_description=visual_description,
        object_a=object_a,
        object_b=object_b,
        mode_decision=mode_decision,
        visual_prompt=visual_prompt,
    )
    if vertical > horizontal and vertical > 0:
        return True, reason
    return False, reason


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
    del productNameResolved, headlineFull
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
        square_note = ""
    elif fmt == "portrait":
        layout = "headline_below_visual"
        square_note = ""
    elif fmt == "square":
        vertical_visual, orient_reason = _square_treat_as_vertical_visual(
            visual_description=visualDescription or "",
            object_a=objectA or "",
            object_b=objectB or "",
            mode_decision=modeDecision or "",
            visual_prompt=visualPrompt or "",
        )
        if vertical_visual:
            layout = "headline_left_visual_right" if lang == "he" else "visual_left_headline_right"
            placement = "Hebrew headline_left_visual_right" if lang == "he" else "LTR visual_left_headline_right"
            square_note = (
                f"Square: treated as vertical visual ({orient_reason}); "
                f"side-by-side layout ({placement})."
            )
        else:
            layout = "headline_below_visual"
            square_note = (
                f"Square: treated as horizontal visual ({orient_reason}); "
                "headline_below_visual."
            )
    else:
        layout = "headline_below_visual"
        square_note = ""

    if layout == "headline_below_visual":
        visual_weight = 0.72
        headline_weight = 0.28
    else:
        visual_weight = 0.5
        headline_weight = 0.5

    base_notes = (
        "Deterministic composition from format/language with centered headline, "
        "at least 1cm safe spacing, and max headline sizing within visual dominance rules."
    )
    composition_notes = f"{base_notes} {square_note}".strip() if square_note else base_notes

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
        "compositionNotes": composition_notes,
    }
