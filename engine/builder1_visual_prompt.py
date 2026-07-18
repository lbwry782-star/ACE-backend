"""
Builder1 campaign-series visual prompt builder (active production).
"""
from __future__ import annotations

from engine.builder1_no_logo import BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK
from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_visibility import ProductVisibilityPolicy

MEDIUM_PROHIBITION = (
    "Do not show this advertisement inside a billboard, framed poster mockup, phone screen, "
    "presentation board, magazine mockup, or floating canvas. The image itself IS the finished advertisement."
)


def _resolve_visibility_policy(series_plan: Builder1SeriesPlan) -> ProductVisibilityPolicy:
    raw = (series_plan.product_visibility_policy or "").strip().upper()
    try:
        return ProductVisibilityPolicy(raw)
    except ValueError:
        internals = series_plan.planning_internals or {}
        raw = str(internals.get("productVisibilityPolicy") or "FORBIDDEN").strip().upper()
        try:
            return ProductVisibilityPolicy(raw)
        except ValueError:
            return ProductVisibilityPolicy.FORBIDDEN


def build_campaign_graphic_identity_block(series_plan: Builder1SeriesPlan) -> str:
    g = series_plan.graphic_generator
    p = g.palette
    c = g.copy_safe_area
    return "\n".join(
        [
            "=== CAMPAIGN GRAPHIC IDENTITY (IDENTICAL IN EVERY AD — REPRODUCE EXACTLY) ===",
            f"Exact palette — dominant {p.dominant}, secondary {p.secondary}, accent {p.accent}, background {p.background}, text {p.text}.",
            f"Layout template: {g.layout_template}. Visual/copy division must match this template.",
            f"Typography style: {g.typography_style}. Headline scale: {g.headline_scale}. Brand scale: {g.brand_scale}. Slogan scale: {g.slogan_scale}.",
            f"Headline position: {g.headline_placement}, alignment {g.headline_alignment}, max width {g.headline_max_width_percent}%.",
            f"Brand block position: {g.brand_block_placement}. Slogan position: {g.slogan_placement}.",
            f"Copy composition zone: {c.width_percent}% on the {c.side} — typeset brand name, slogan and optional headline inside this zone as integrated ad design.",
            f"Image style: {g.image_style}. Background: {g.background_treatment}. Border: {g.border_treatment}.",
            f"Shape language: {g.shape_language}. Framing rule: {g.framing_rule}. Spacing rule: {g.spacing_rule}.",
            f"Recurring graphic device: {g.recurring_graphic_device}.",
            f"Recurring device rule (must be visibly present in this ad): {g.recurring_graphic_device_rule}.",
            "The recurring graphic device is a campaign composition element only — not a product logo, packaging brand mark, or symbol beside the product name.",
            "Render the recurring graphic device prominently. Do not omit it.",
            "=== END CAMPAIGN GRAPHIC IDENTITY ===",
        ]
    )


def build_text_to_render_block(
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
) -> str:
    headline_line = (
        f'Headline:\n"{ad_plan.headline}"'
        if ad_plan.headline
        else "Headline:\n(null — do not render any headline text)"
    )
    return "\n".join(
        [
            "=== TEXT TO RENDER EXACTLY ===",
            f'Brand name:\n"{series_plan.product_name_resolved}"',
            f'Brand slogan:\n"{series_plan.brand_slogan}"',
            headline_line,
            "Rules:",
            "- Render the brand name as plain readable advertising typography only.",
            "- Do not print the brand name on any object, label, packaging, badge, seal, emblem, or sign.",
            "- Do not accompany the brand name with any symbol, icon, emblem, monogram, badge, seal, or logo mark.",
            "- Render these strings exactly as written.",
            "- Do not translate, paraphrase, replace words, or invent additional copy.",
            "- Preserve the original language, punctuation, and word order.",
            "- Do not add placeholder text, lorem ipsum, interface labels, unrelated logos, or watermarks.",
            "- Integrate the copy visually into the advertisement composition — not as external captions.",
            "=== END TEXT TO RENDER EXACTLY ===",
        ]
    )


def _forbidden_main_visual_block(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    transferred = series_plan.transferred_object or series_plan.physical_generator
    action = series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
    return "\n".join(
        [
            "=== MAIN VISUAL (ONLY SUBJECT) ===",
            f"MAIN VISUAL: {transferred}",
            f"ACTION: {action}",
            f"Ad variation: {ad_plan.variation_label}.",
            f"Composition execution: {ad_plan.physical_execution or ad_plan.visual_execution}.",
            "This transferred external object is the sole hero subject of the advertisement.",
            "=== END MAIN VISUAL ===",
            "=== ADVERTISED PRODUCT ===",
            "ADVERTISED PRODUCT: not depicted",
            "=== END ADVERTISED PRODUCT ===",
            "=== PACKAGING ===",
            "PACKAGING: not depicted",
            "=== END PACKAGING ===",
            "=== BRAND IDENTIFICATION ===",
            "Product Name and slogan appear only as plain readable advertising typography.",
            "Do not attach brand identification to any object, package, label, or sign.",
            "=== END BRAND IDENTIFICATION ===",
        ]
    )


def _secondary_exception_main_visual_block(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    transferred = series_plan.transferred_object or series_plan.physical_generator
    action = series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
    return "\n".join(
        [
            "=== MAIN VISUAL ===",
            f"MAIN VISUAL: {transferred}",
            f"ACTION: {action}",
            f"Ad variation: {ad_plan.variation_label}.",
            "The transferred object remains the main visual.",
            "The advertised product may appear only as a small secondary unbranded element.",
            "=== END MAIN VISUAL ===",
        ]
    )


def build_visual_prompt(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    policy = _resolve_visibility_policy(series_plan)
    medium_block = (
        f"Medium participation (justified): {series_plan.medium_role}."
        if series_plan.medium_participates
        else MEDIUM_PROHIBITION
    )
    hebrew_block = ""
    if series_plan.detected_language == "he":
        hebrew_block = (
            "Hebrew composition: main visual on the right or center; RTL reading flow; "
            f"fixed brand slogan at {series_plan.graphic_generator.slogan_placement}."
        )
    headline_rule = (
        "Optional ad headline for this execution only — render it exactly as specified."
        if ad_plan.headline
        else "No ad headline for this execution — do not invent headline text."
    )
    main_visual_block = (
        _forbidden_main_visual_block(series_plan, ad_plan)
        if policy == ProductVisibilityPolicy.FORBIDDEN
        else _secondary_exception_main_visual_block(series_plan, ad_plan)
    )
    parts = [
        "Create a complete finished advertisement that fills the entire image frame edge to edge.",
        f"Format: {series_plan.format}. The output is the final ad itself, not a background for later overlay.",
        BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK,
        main_visual_block,
        f"Fixed brand slogan (typography only): {series_plan.brand_slogan}.",
        f"Slogan-implied action for the transferred object: {series_plan.slogan_action}.",
        "MARKETING TEXT must NOT appear inside the image.",
        headline_rule,
        hebrew_block,
        build_campaign_graphic_identity_block(series_plan),
        build_text_to_render_block(series_plan, ad_plan),
        medium_block,
        "Prohibit any text beyond the exact brand name, brand slogan, and optional headline specified above.",
        "Prohibit additional slogans, paragraphs, captions, UI elements, stock watermarks, or decorative logos.",
        "Marketing body copy must NOT appear in the image.",
        "Object colors must not redefine the campaign palette.",
        "The final advertisement must visibly demonstrate the shared art direction, palette, typography hierarchy, and recurring graphic device.",
    ]
    return "\n".join(parts)
