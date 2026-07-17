"""
Builder1 campaign-series visual prompt builder (active production).
"""
from __future__ import annotations

from engine.builder1_plan_spec import (
    Builder1AdPlan,
    Builder1SeriesPlan,
    graphic_generator_to_dict,
)

MEDIUM_PROHIBITION = (
    "Do not show billboards, posters, ad frames, presentation boards, phone screens, "
    "social-media interfaces, magazine mockups, or any advertising medium used only as a container."
)


def _palette_line(plan: Builder1SeriesPlan) -> str:
    colors = ", ".join(plan.graphic_generator.color_palette)
    return f"Campaign color palette (exact, do not deviate): {colors}."


def _graphic_identity_block(plan: Builder1SeriesPlan) -> str:
    g = plan.graphic_generator
    c = g.composition
    return "\n".join(
        [
            _palette_line(plan),
            f"Image style (shared across campaign): {g.image_style}.",
            f"Composition grid: {c.grid}.",
            f"Visual area: {c.visual_area}.",
            f"Copy area (reserve empty, no text rendered): {c.copy_area}.",
            f"Alignment: {c.alignment}.",
            f"Slogan placement reserve: {c.slogan_placement}.",
            f"Brand signature placement reserve: {c.brand_placement}.",
            f"Spacing system: {g.spacing}.",
            f"Visual treatment: {g.visual_treatment}.",
            f"Background treatment: {g.background_treatment}.",
        ]
    )


def _campaign_identity_block(plan: Builder1SeriesPlan) -> str:
    return "\n".join(
        [
            f"Shared conceptual generator: {plan.conceptual_generator}.",
            f"Shared physical generator family: {plan.physical_generator}.",
            f"Physical generator role: {plan.physical_generator_campaign_role}.",
            f"Series principle: {plan.series_generator.principle}.",
        ]
    )


def build_visual_prompt(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    """
    Build one English image prompt for a single ad using the shared campaign plan.
    """
    medium_block = ""
    if series_plan.medium_participates:
        medium_block = f"Medium participation (justified): {series_plan.medium_role}."
    else:
        medium_block = MEDIUM_PROHIBITION

    parts = [
        "Professional advertising visual. No written text, letters, words, slogans, captions, watermarks, or logos in the image.",
        f"Format aspect: {series_plan.format}.",
        _graphic_identity_block(series_plan),
        _campaign_identity_block(series_plan),
        f"Ad {ad_plan.index} — {ad_plan.variation_label}.",
        f"This ad's new contribution: {ad_plan.new_contribution}.",
        f"Physical execution: {ad_plan.physical_execution}.",
        f"Visual execution: {ad_plan.visual_execution}.",
        f"Scene: {ad_plan.scene_description}.",
        medium_block,
        "Negative: text, typography, interface elements, inconsistent art direction, unrelated decorative objects, accidental logos.",
        "Object colors must not redefine the campaign palette.",
    ]
    return "\n".join(parts)
