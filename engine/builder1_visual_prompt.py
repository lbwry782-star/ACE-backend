"""
Builder1 campaign-series visual prompt builder (active production).
"""
from __future__ import annotations

from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan

MEDIUM_PROHIBITION = (
    "Do not show billboards, posters, ad frames, presentation boards, phone screens, "
    "social-media interfaces, magazine mockups, or any advertising medium used only as a container."
)


def build_campaign_graphic_identity_block(series_plan: Builder1SeriesPlan) -> str:
    """
    Identical graphic identity block repeated verbatim in every ad image prompt.
    """
    g = series_plan.graphic_generator
    p = g.palette
    c = g.copy_safe_area
    return "\n".join(
        [
            "=== CAMPAIGN GRAPHIC IDENTITY (IDENTICAL IN EVERY AD — DO NOT VARY) ===",
            f"Palette dominant {p.dominant}, secondary {p.secondary}, accent {p.accent}, background {p.background}, text {p.text}.",
            f"Layout template: {g.layout_template}.",
            f"Copy-safe area: {c.width_percent}% on the {c.side} — keep this region visually calm and uncluttered for future headline/slogan overlay.",
            f"Headline reserve: {g.headline_placement}, alignment {g.headline_alignment}, max width {g.headline_max_width_percent}%, color {g.headline_color}, treatment {g.headline_treatment}.",
            f"Brand block reserve: {g.brand_block_placement}. Slogan reserve: {g.slogan_placement}.",
            f"Image style: {g.image_style}. Background: {g.background_treatment}. Border: {g.border_treatment}.",
            f"Framing rule: {g.framing_rule}.",
            f"Recurring graphic device: {g.recurring_graphic_device}.",
            f"Recurring device rule (must appear in this ad): {g.recurring_graphic_device_rule}.",
            "Compose the visual around the copy-safe area — subject and action must not obstruct the reserved headline zone.",
            "=== END CAMPAIGN GRAPHIC IDENTITY ===",
        ]
    )


def _campaign_strategy_block(series_plan: Builder1SeriesPlan) -> str:
    return "\n".join(
        [
            f"Shared conceptual action: {series_plan.conceptual_generator_action}.",
            f"Conceptual transformation: {series_plan.conceptual_generator_transformation}.",
            f"Shared physical generator family: {series_plan.physical_generator}.",
            f"Series principle: {series_plan.series_generator.principle}.",
        ]
    )


def build_visual_prompt(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    medium_block = (
        f"Medium participation (justified): {series_plan.medium_role}."
        if series_plan.medium_participates
        else MEDIUM_PROHIBITION
    )
    parts = [
        "Professional advertising visual. No written text, letters, words, slogans, captions, watermarks, or logos in the image.",
        f"Format aspect: {series_plan.format}.",
        build_campaign_graphic_identity_block(series_plan),
        _campaign_strategy_block(series_plan),
        f"Ad {ad_plan.index} — {ad_plan.variation_label}.",
        f"Conceptual execution for this ad: {ad_plan.conceptual_execution}.",
        f"Conceptual action proof: {ad_plan.conceptual_action_proof}.",
        f"New contribution: {ad_plan.new_contribution}.",
        f"Physical execution: {ad_plan.physical_execution}.",
        f"Visual execution: {ad_plan.visual_execution}.",
        f"Scene: {ad_plan.scene_description}.",
        medium_block,
        "Negative: text, typography, interface elements, inconsistent art direction, unrelated decorative objects, accidental logos.",
        "Object colors must not redefine the campaign palette.",
    ]
    return "\n".join(parts)
