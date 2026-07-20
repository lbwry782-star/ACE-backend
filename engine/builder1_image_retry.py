"""
Builder1 cumulative image retry prompts and violation history helpers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from engine.builder1_no_logo import BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK
from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_visibility import (
    BUILDER1_NO_PRODUCT_STRICT_CORRECTION_BLOCK,
    ProductVisibilityPolicy,
)

logger = logging.getLogger(__name__)

LOGO_VIOLATION_CODES = frozenset(
    {
        "invented_product_logo",
        "supplied_logo_displayed",
        "logo_like_brand_symbol",
        "packaging_contains_brand_mark",
        "campaign_device_used_as_logo",
        "product_name_rendered_as_logo",
        "product_name_not_text_only",
    }
)

VISIBILITY_VIOLATION_CODES = frozenset(
    {
        "product_visible_without_explicit_request",
        "packaging_visible_without_explicit_request",
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
    }
)

PERMANENT_FORBIDDEN_GLOBAL_CONSTRAINTS = "\n".join(
    [
        "=== GLOBAL IMAGE CONSTRAINTS (PERMANENT — ALWAYS APPLY) ===",
        "Advertised product must not be depicted.",
        "Product packaging must not be depicted.",
        "Product Name appears only as plain readable advertising typography.",
        "No supplied or invented logo, emblem, badge, seal, monogram, trademark-like symbol, or brand mark.",
        "The recurring campaign graphic device may appear only as a compositional motif — never as a logo.",
        "The transferred external physical object remains the dominant main visual.",
        "The transferred-object action remains unchanged.",
        "The fixed campaign slogan remains unchanged.",
        "The approved graphic system, palette, and series coherence remain recognizable.",
        "Do not introduce a substitute product object disguised as the transferred generator.",
        "=== END GLOBAL IMAGE CONSTRAINTS ===",
    ]
)

CAMPAIGN_DEVICE_AS_LOGO_CORRECTION = "\n".join(
    [
        "=== CAMPAIGN DEVICE CORRECTION (MANDATORY) ===",
        "Preserve the recurring graphic device as a compositional motif integrated into the ad layout.",
        "Move it away from Product Name and slogan; do not place it in a brand-signature position beside the name.",
        "Do not render it small, isolated, emblematic, stamped, enclosed, badge-like, or mark-like.",
        "Keep Product Name as plain readable typography only — not inside a device frame.",
        "Do not remove the central campaign idea or delete the recurring device entirely.",
        "Do not invent a replacement logo or brand symbol.",
        "Preserve the fixed slogan unchanged.",
        "Preserve the approved graphic system, palette, and series coherence.",
        "=== END CAMPAIGN DEVICE CORRECTION ===",
    ]
)

PRODUCT_VISIBILITY_CORRECTION = "\n".join(
    [
        "=== PRODUCT VISIBILITY CORRECTION (MANDATORY) ===",
        "Remove all advertised-product units from the image.",
        "Remove all packaging, containers, bottles, cans, boxes, cartons, jars, bags, and devices matching the product.",
        "Remove objects matching the advertised product description.",
        "Preserve ONLY the transferred external physical object as the hero subject.",
        "Do not replace the product with a disguised substitute object.",
        "Preserve Product Name and slogan only as plain readable advertising typography.",
        "=== END PRODUCT VISIBILITY CORRECTION ===",
    ]
)

MINIMAL_SAFE_EXECUTION_BLOCK = "\n".join(
    [
        "=== MINIMAL SAFE EXECUTION (ATTEMPT 3 — MANDATORY) ===",
        "Preserve the fixed campaign concept, transferred physical generator, transferred-object action,",
        "Product Name, fixed slogan, campaign palette, and composition family.",
        "Use one dominant transferred external object as the hero subject.",
        "Render Product Name and slogan only as plain readable typography.",
        "Do not add small decorative icons, isolated symbolic marks, badges, seals, monograms, emblems, or hearts beside Product Name.",
        "Express the recurring graphic device only as a large integrated compositional feature or broad pattern away from the brand block.",
        "Do not depict the advertised product, packaging, or substitute product objects.",
        "Do not invent a logo, emblem, or brand mark.",
        "=== END MINIMAL SAFE EXECUTION ===",
    ]
)

CORRECTION_PROFILE_NORMAL = "normal_execution"
CORRECTION_PROFILE_CUMULATIVE = "cumulative_targeted"
CORRECTION_PROFILE_MINIMAL_SAFE = "minimal_safe"


def normalize_violation_union(violations: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(code).strip() for code in violations if str(code).strip()))


def build_minimal_safe_execution_block(
    *,
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
) -> str:
    positive = build_positive_main_visual_block(series_plan=series_plan, ad_plan=ad_plan)
    return f"{MINIMAL_SAFE_EXECUTION_BLOCK}\n\n{positive}"


def entry_hard_violations(entry: Dict[str, object]) -> List[str]:
    hard = entry.get("hardViolations")
    if isinstance(hard, list) and hard:
        return normalize_violation_union(hard)
    return normalize_violation_union(entry.get("violations") or [])


def entry_advisories(entry: Dict[str, object]) -> List[str]:
    return normalize_violation_union(entry.get("advisories") or [])


def union_hard_violations_for_ad(history: Dict[str, List[Dict[str, object]]], ad_index: int) -> List[str]:
    entries = history.get(str(ad_index)) or history.get(ad_index) or []
    union: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            union.extend(entry_hard_violations(entry))
    return normalize_violation_union(union)


def union_advisories_for_ad(history: Dict[str, List[Dict[str, object]]], ad_index: int) -> List[str]:
    entries = history.get(str(ad_index)) or history.get(ad_index) or []
    union: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            union.extend(entry_advisories(entry))
    return normalize_violation_union(union)


def build_positive_main_visual_block(
    *,
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
) -> str:
    transferred = (
        series_plan.transferred_object
        or series_plan.physical_generator
        or ad_plan.physical_execution
        or "the approved transferred external object"
    )
    action = (
        series_plan.transferred_object_action
        or series_plan.physical_generator_campaign_role
        or ad_plan.physical_execution
        or "the approved transferred physical action"
    )
    return "\n".join(
        [
            "=== POSITIVE MAIN VISUAL (RESTATED) ===",
            f"MAIN VISUAL: {transferred}",
            f"ACTION: {action}",
            f'BRAND TEXT: Product Name "{series_plan.product_name_resolved}" and fixed slogan '
            f'"{series_plan.brand_slogan}" as plain readable typography only.',
            "Describe a complete finished advertisement image from these positive instructions.",
            "=== END POSITIVE MAIN VISUAL ===",
        ]
    )


def build_violation_specific_corrections(violations: Sequence[str]) -> List[str]:
    blocks: List[str] = []
    normalized = set(normalize_violation_union(violations))
    logo_related = normalized & LOGO_VIOLATION_CODES
    if "campaign_device_used_as_logo" in normalized:
        blocks.append(CAMPAIGN_DEVICE_AS_LOGO_CORRECTION)
    elif logo_related:
        blocks.append(BUILDER1_NO_LOGO_IMAGE_PROMPT_BLOCK)
    visibility_related = normalized & VISIBILITY_VIOLATION_CODES
    if visibility_related:
        blocks.append(PRODUCT_VISIBILITY_CORRECTION)
        if "product_visible_without_explicit_request" in normalized or visibility_related:
            blocks.append(BUILDER1_NO_PRODUCT_STRICT_CORRECTION_BLOCK)
    return blocks


def build_cumulative_image_correction_block(
    *,
    violations: Sequence[str],
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
    campaign_id: str = "",
    ad_index: int = 0,
    plan_revision: int = 1,
) -> str:
    union = normalize_violation_union(violations)
    if not union:
        return ""
    logger.info(
        "BUILDER1_IMAGE_RETRY_CORRECTION campaignId=%s adIndex=%s planRevision=%s violations=%s",
        campaign_id or "",
        ad_index,
        plan_revision,
        union,
    )
    sections = [
        PERMANENT_FORBIDDEN_GLOBAL_CONSTRAINTS,
        "=== ACCUMULATED VIOLATIONS TO CORRECT ===",
        ", ".join(union),
        "=== END ACCUMULATED VIOLATIONS ===",
        *build_violation_specific_corrections(union),
        build_positive_main_visual_block(series_plan=series_plan, ad_plan=ad_plan),
    ]
    return "\n\n".join(section for section in sections if section.strip())


def parse_image_attempt_history(raw: object) -> Dict[str, List[Dict[str, object]]]:
    if not isinstance(raw, dict):
        return {}
    parsed: Dict[str, List[Dict[str, object]]] = {}
    for key, entries in raw.items():
        if not isinstance(entries, list):
            continue
        normalized_entries: List[Dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            violations = entry.get("violations")
            hard = entry.get("hardViolations")
            advisories = entry.get("advisories")
            if not isinstance(violations, list):
                violations = hard if isinstance(hard, list) else []
            if not isinstance(hard, list):
                hard = violations if isinstance(violations, list) else []
            if not isinstance(advisories, list):
                advisories = []
            if not violations and not hard and not advisories:
                continue
            try:
                attempt = int(entry.get("attempt"))
            except (TypeError, ValueError):
                continue
            normalized_entries.append(
                {
                    "attempt": attempt,
                    "violations": normalize_violation_union(violations or hard),
                    "hardViolations": normalize_violation_union(hard or violations),
                    "advisories": normalize_violation_union(advisories),
                    "evidenceSummary": str(entry.get("evidenceSummary") or "").strip(),
                    "correctionProfile": str(entry.get("correctionProfile") or "").strip(),
                }
            )
        if normalized_entries:
            parsed[str(key)] = normalized_entries
    return parsed


def union_violations_for_ad(history: Dict[str, List[Dict[str, object]]], ad_index: int) -> List[str]:
    return union_hard_violations_for_ad(history, ad_index)


def next_attempt_number(history: Dict[str, List[Dict[str, object]]], ad_index: int) -> int:
    entries = history.get(str(ad_index)) or history.get(ad_index) or []
    return len(entries) + 1
