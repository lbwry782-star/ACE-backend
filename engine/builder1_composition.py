"""
Builder1 campaign-series composition metadata (deterministic, no model calls).
"""
from __future__ import annotations

from typing import Any, Dict

from engine.builder1_plan_spec import Builder1SeriesPlan, graphic_generator_to_dict


def build_builder1_series_composition_metadata(series_plan: Builder1SeriesPlan) -> Dict[str, Any]:
    """
    Serialize shared graphic-generator metadata for accessibility, ZIP, and display.
    Copy is rendered inside the generated image — no Frontend overlay is required.
    """
    g = graphic_generator_to_dict(series_plan.graphic_generator)
    return {
        "format": series_plan.format,
        "brandSlogan": series_plan.brand_slogan,
        "productNameResolved": series_plan.product_name_resolved,
        "graphicGenerator": g,
        "mediumParticipates": series_plan.medium_participates,
        "mediumRole": series_plan.medium_role,
        "imageContainsFinalCopy": True,
    }
