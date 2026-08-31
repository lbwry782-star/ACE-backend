"""
Builder1 campaign-series visual prompt builder (active production).
"""
from __future__ import annotations

from typing import List

from engine.builder1_graphic_device_necessity import (
    build_no_device_annotation_guard_block,
    has_recurring_graphic_device,
)
from engine.builder1_no_logo import BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK
from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_shot_methodology import BUILDER1_FORBIDDEN_PRODUCT_SHOT_LANGUAGE
from engine.builder1_literal_embodiment import BUILDER1_IMAGE_EXPRESSIVE_OBJECT_RULE
from engine.builder1_methodology_reasons import NO_LOGO_REASON, POSITIVE_IMAGE_PROMPT_REASON
from engine.builder1_object_design_integrity import (
    BUILDER1_OBJECT_DESIGN_PALETTE_BOUNDARY,
    build_composition_execution_lines,
    build_object_design_prompt_block,
    object_design_fields_for_ad_index,
)
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    VisualExecutionRoute,
    infer_visual_execution_route,
    plan_approves_product_as_main_visual,
    policy_is_legacy_secondary_only,
    policy_prohibits_product_depiction,
    policy_uses_route_selection,
    resolve_product_visibility_policy,
)

MEDIUM_PROHIBITION = (
    "Do not show this advertisement inside a billboard, framed poster mockup, phone screen, "
    "presentation board, magazine mockup, or floating canvas. The image itself IS the finished advertisement."
)


def _composition_lines(ad_plan: Builder1AdPlan) -> List[str]:
    return build_composition_execution_lines(
        physical_execution=ad_plan.physical_execution,
        visual_execution=ad_plan.visual_execution,
    )


def _object_design_block(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan, *, product_led: bool) -> str:
    return build_object_design_prompt_block(
        object_design_fields_for_ad_index(series_plan, ad_plan.index),
        skip_for_product_led=product_led,
    )


def _resolve_visibility_policy(series_plan: Builder1SeriesPlan) -> ProductVisibilityPolicy:
    return resolve_product_visibility_policy(series_plan.product_visibility_policy)


def _visual_route(series_plan: Builder1SeriesPlan) -> VisualExecutionRoute:
    internals = series_plan.planning_internals or {}
    raw = str(internals.get("visualExecutionRoute") or "").strip().upper()
    if raw:
        try:
            return VisualExecutionRoute(raw)
        except ValueError:
            pass
    return infer_visual_execution_route(
        physical_generator_is_product=bool(
            internals.get("physicalGeneratorIsProduct") or internals.get("productIsPhysicalGenerator")
        ),
        product_evidence_required=bool(internals.get("productEvidenceRequired")),
    )


def build_campaign_graphic_identity_block(series_plan: Builder1SeriesPlan) -> str:
    g = series_plan.graphic_generator
    p = g.palette
    c = g.copy_safe_area
    lines = [
        "=== CAMPAIGN GRAPHIC IDENTITY (IDENTICAL IN EVERY AD — REPRODUCE EXACTLY) ===",
        f"Exact palette — dominant {p.dominant}, secondary {p.secondary}, accent {p.accent}, background {p.background}, text {p.text}.",
        f"Layout template: {g.layout_template}. Visual/copy division must match this template.",
        f"Typography style: {g.typography_style}. Headline scale: {g.headline_scale}. Brand scale: {g.brand_scale}. Slogan scale: {g.slogan_scale}.",
        f"Headline position: {g.headline_placement}, alignment {g.headline_alignment}, max width {g.headline_max_width_percent}%.",
        f"Brand block position: {g.brand_block_placement}. Slogan position: {g.slogan_placement}.",
        f"Copy composition zone: {c.width_percent}% on the {c.side} — typeset brand name, slogan and optional headline inside this zone as integrated ad design.",
        f"Image style: {g.image_style}. Background: {g.background_treatment}. Border: {g.border_treatment}.",
        f"Shape language: {g.shape_language}. Framing rule: {g.framing_rule}. Spacing rule: {g.spacing_rule}.",
    ]
    if has_recurring_graphic_device(g.recurring_graphic_device, g.recurring_graphic_device_rule):
        lines.extend(
            [
                f"Recurring graphic device: {g.recurring_graphic_device}.",
                f"Recurring device rule (must be visibly present in this ad): {g.recurring_graphic_device_rule}.",
                "The recurring graphic device is a campaign composition element only — not a product logo, packaging brand mark, or symbol beside the product name.",
                "Render the recurring graphic device prominently. Do not omit it.",
            ]
        )
    else:
        lines.append(build_no_device_annotation_guard_block(border_treatment=g.border_treatment))
    lines.append("=== END CAMPAIGN GRAPHIC IDENTITY ===")
    return "\n".join(lines)


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
    composition = _composition_lines(ad_plan)
    return "\n".join(
        [
            "=== MAIN VISUAL (ONLY SUBJECT) ===",
            f"MAIN VISUAL: {transferred}",
            f"ACTION: {action}",
            f"Ad variation: {ad_plan.variation_label}.",
            *composition,
            "Center the selected external expressive object and its physical action as the advertisement's visual proof.",
            "Product Name and slogan appear only as plain typography — not on objects, packaging, or signs.",
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


def _product_led_main_visual_block(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    action = series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
    composition = _composition_lines(ad_plan)
    return "\n".join(
        [
            "=== MAIN VISUAL (PRODUCT-LED — APPROVED) ===",
            f"MAIN VISUAL: the advertised product itself — {series_plan.product_description}",
            f"ACTION: {action}",
            f"Ad variation: {ad_plan.variation_label}.",
            *composition,
            "The product itself carries the advertising idea through its form, property, arrangement, or transformation.",
            "This is an approved product-led execution — not a generic packshot.",
            "Product Name and slogan appear only as plain typography — never as an invented logo or packaging mark.",
            "=== END MAIN VISUAL ===",
        ]
    )


def _integrated_main_visual_block(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    transferred = series_plan.transferred_object or series_plan.physical_generator
    action = series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
    composition = _composition_lines(ad_plan)
    return "\n".join(
        [
            "=== MAIN VISUAL (PRODUCT-INTEGRATED ANALOGY — APPROVED) ===",
            f"MAIN VISUAL: {transferred}",
            f"ACTION: {action}",
            f"Ad variation: {ad_plan.variation_label}.",
            *composition,
            "The advertised product may appear as a participant in this mechanism.",
            "The transferred analogy remains the governing visual law.",
            "Product Name as plain typography only — no invented logo or packaging mark.",
            "=== END MAIN VISUAL ===",
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


def _creative_analogy_main_visual_block(
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
    *,
    product_required: bool = False,
) -> str:
    transferred = series_plan.transferred_object or series_plan.physical_generator
    action = series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
    if product_required:
        product_line = (
            "ADVERTISED PRODUCT: must appear in the image. "
            "The transferred object may remain the main visual while the product appears as required participant or secondary element."
        )
        header = "=== MAIN VISUAL (ANALOGY-LED — PRODUCT VISIBILITY REQUIRED) ==="
    else:
        product_line = "ADVERTISED PRODUCT: not depicted unless explicitly integrated in the approved mechanism above."
        header = "=== MAIN VISUAL (ANALOGY-LED — APPROVED) ==="
    composition = _composition_lines(ad_plan)
    return "\n".join(
        [
            header,
            f"MAIN VISUAL: {transferred}",
            f"ACTION: {action}",
            f"Ad variation: {ad_plan.variation_label}.",
            *composition,
            "Center the transferred external object and its physical action as the advertisement's visual proof.",
            "=== END MAIN VISUAL ===",
            "=== ADVERTISED PRODUCT ===",
            product_line,
            "=== END ADVERTISED PRODUCT ===",
        ]
    )


def build_visual_prompt(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> str:
    policy = _resolve_visibility_policy(series_plan)
    route = _visual_route(series_plan)
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
    if policy_prohibits_product_depiction(policy):
        main_visual_block = _forbidden_main_visual_block(series_plan, ad_plan)
        preserve_object_rule = (
            "Preserve the selected external expressive object as MAIN VISUAL. "
            "Do not substitute the advertised product, product category, literal slogan noun, road/path/maze/car/train imagery, "
            "or other literal illustration unless that object was explicitly selected in planning."
        )
    elif policy_is_legacy_secondary_only(policy):
        main_visual_block = _secondary_exception_main_visual_block(series_plan, ad_plan)
        preserve_object_rule = "Preserve the transferred object as MAIN VISUAL."
    elif policy_uses_route_selection(policy) and (
        route == VisualExecutionRoute.PRODUCT_LED or plan_approves_product_as_main_visual(
            series_plan, ad_index=ad_plan.index
        )
    ):
        main_visual_block = _product_led_main_visual_block(series_plan, ad_plan)
        preserve_object_rule = (
            "Execute the approved product-led creative mechanism faithfully. "
            "Do not substitute an external analogy when the plan selected the product itself as the idea carrier."
        )
    elif route == VisualExecutionRoute.PRODUCT_INTEGRATED_ANALOGY and policy_uses_route_selection(policy):
        main_visual_block = _integrated_main_visual_block(series_plan, ad_plan)
        preserve_object_rule = "Preserve both the transferred analogy and approved product participation."
    elif policy_uses_route_selection(policy):
        main_visual_block = _creative_analogy_main_visual_block(
            series_plan,
            ad_plan,
            product_required=policy == ProductVisibilityPolicy.PRODUCT_VISIBILITY_REQUIRED,
        )
        preserve_object_rule = (
            "Preserve the selected external expressive object as MAIN VISUAL when the plan is analogy-led."
            if policy == ProductVisibilityPolicy.CREATIVE_DECISION
            else "Product visibility is required; hierarchy follows the approved route."
        )

    product_led = policy_uses_route_selection(policy) and (
        route == VisualExecutionRoute.PRODUCT_LED
        or plan_approves_product_as_main_visual(series_plan, ad_index=ad_plan.index)
    )
    object_design_block = _object_design_block(series_plan, ad_plan, product_led=product_led)

    parts = [
        "Create a complete finished advertisement that fills the entire image frame edge to edge.",
        f"Format: {series_plan.format}. The output is the final ad itself, not a background for later overlay.",
        BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK,
        main_visual_block,
        object_design_block,
        POSITIVE_IMAGE_PROMPT_REASON,
        NO_LOGO_REASON,
    ]
    if policy_prohibits_product_depiction(policy):
        parts.append(BUILDER1_FORBIDDEN_PRODUCT_SHOT_LANGUAGE)
    parts.extend(
        [
            BUILDER1_IMAGE_EXPRESSIVE_OBJECT_RULE,
            preserve_object_rule,
            "Do not add discarded slogan nouns back into the scene when planning selected a non-literal expressive object.",
            f"Fixed brand slogan (typography only): {series_plan.brand_slogan}.",
            f"Slogan-implied action: {series_plan.slogan_action}.",
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
            BUILDER1_OBJECT_DESIGN_PALETTE_BOUNDARY.strip(),
            (
                "The final advertisement must visibly demonstrate the shared art direction, palette, typography hierarchy, and recurring graphic device."
                if has_recurring_graphic_device(
                    series_plan.graphic_generator.recurring_graphic_device,
                    series_plan.graphic_generator.recurring_graphic_device_rule,
                )
                else "The final advertisement must visibly demonstrate the shared art direction, palette, typography hierarchy, and composition rules."
            ),
        ]
    )
    return "\n".join(parts)
