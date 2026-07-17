"""
Builder1 campaign-series composition metadata (deterministic, no model calls).
"""
from __future__ import annotations

from typing import Any, Dict

from engine.builder1_plan_spec import Builder1SeriesPlan, graphic_generator_to_dict


def build_builder1_series_composition_metadata(series_plan: Builder1SeriesPlan) -> Dict[str, Any]:
    """
    Serialize shared graphic-generator metadata for Frontend overlay rendering.
    Same layout identity for every ad in the campaign.
    """
    g = graphic_generator_to_dict(series_plan.graphic_generator)
    return {
        "format": series_plan.format,
        "brandSlogan": series_plan.brand_slogan,
        "graphicGenerator": g,
        "mediumParticipates": series_plan.medium_participates,
        "mediumRole": series_plan.medium_role,
    }


# Legacy o3 composition removed from active Builder1 production path.
def generate_builder1_composition_o3(*_args, **_kwargs):
    raise NotImplementedError(
        "generate_builder1_composition_o3 removed; use build_builder1_series_composition_metadata"
    )
