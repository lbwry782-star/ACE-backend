"""
Builder1 image prompt preflight — structural checks before image API calls.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_visibility import ProductVisibilityPolicy

logger = logging.getLogger(__name__)

PREFLIGHT_FORBIDDEN_SECTION_MARKERS = (
    "advertisedproductvisual",
    "productdescription",
    "heroproduct",
    "productshot",
    "packagingshot",
)


@dataclass(frozen=True)
class ImagePromptPreflightResult:
    ok: bool
    reasons: List[str]


class ImagePromptPreflightError(Exception):
    """Campaign plan or prompt is structurally incompatible with image generation."""

    def __init__(self, reasons: List[str]):
        self.reasons = reasons
        super().__init__(f"image_prompt_preflight_failed:{','.join(reasons)}")


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _resolve_policy(series_plan: Builder1SeriesPlan) -> ProductVisibilityPolicy:
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


def _ad_requests_product_visibility(series_plan: Builder1SeriesPlan, ad_plan: Builder1AdPlan) -> bool:
    internals = series_plan.planning_internals or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    extra = ad_internals.get(ad_plan.index) or ad_internals.get(str(ad_plan.index)) or {}
    if not isinstance(extra, dict):
        return False
    if extra.get("productVisible") is True:
        return True
    if extra.get("packagingVisible") is True:
        return True
    if extra.get("productIsMainVisual") is True:
        return True
    if extra.get("productIsPhysicalGenerator") is True:
        return True
    if extra.get("productVisibilityRequired") is True:
        return True
    return False


def validate_image_prompt_plan(
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
) -> ImagePromptPreflightResult:
    reasons: List[str] = []
    policy = _resolve_policy(series_plan)
    transferred = _norm(series_plan.transferred_object or series_plan.physical_generator)
    transferred_action = _norm(
        series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
    )

    if policy != ProductVisibilityPolicy.FORBIDDEN:
        if not transferred:
            reasons.append("missing_transferred_object")
        return ImagePromptPreflightResult(ok=not reasons, reasons=reasons)

    if not transferred:
        reasons.append("missing_transferred_object")
    if not transferred_action:
        reasons.append("missing_transferred_object_action")
    if _ad_requests_product_visibility(series_plan, ad_plan):
        reasons.append("plan_requests_product_visibility")

    return ImagePromptPreflightResult(ok=not reasons, reasons=reasons)


def validate_forbidden_visual_prompt_text(
    prompt: str,
    *,
    series_plan: Builder1SeriesPlan,
) -> ImagePromptPreflightResult:
    reasons: List[str] = []
    policy = _resolve_policy(series_plan)
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        return ImagePromptPreflightResult(ok=True, reasons=[])

    lowered = prompt.lower()
    if "main visual:" not in lowered and "primary object:" not in lowered:
        reasons.append("missing_main_visual_block")
    if "advertised product:" not in lowered or "not depicted" not in lowered:
        reasons.append("missing_advertised_product_not_depicted")
    if "packaging:" not in lowered or "not depicted" not in lowered.split("packaging:", 1)[-1][:80]:
        reasons.append("missing_packaging_not_depicted")

    description = _norm(series_plan.product_description).lower()
    if description and len(description) >= 12:
        desc_tokens = [t for t in re.findall(r"[a-zA-Z\u0590-\u05FF]{5,}", description) if t]
        hits = sum(1 for token in desc_tokens[:8] if token in lowered)
        if hits >= 3:
            reasons.append("product_description_leaked_into_prompt")

    for marker in PREFLIGHT_FORBIDDEN_SECTION_MARKERS:
        if marker in re.sub(r"[^a-z0-9]", "", lowered):
            reasons.append("forbidden_visual_section_present")

    product_name = _norm(series_plan.product_name_resolved).lower()
    if product_name:
        name_on_object_patterns = (
            rf"{re.escape(product_name)}\s+(label|packaging|bottle|box|jar|can)",
            r"label.*" + re.escape(product_name),
        )
        for pattern in name_on_object_patterns:
            if re.search(pattern, lowered):
                reasons.append("product_name_on_object_or_package")

    return ImagePromptPreflightResult(ok=not reasons, reasons=reasons)


def run_image_prompt_preflight(
    *,
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
    prompt: str,
    campaign_id: str = "",
    ad_index: int = 0,
) -> None:
    plan_result = validate_image_prompt_plan(series_plan, ad_plan)
    text_result = validate_forbidden_visual_prompt_text(prompt, series_plan=series_plan)
    reasons = list(dict.fromkeys(plan_result.reasons + text_result.reasons))
    if reasons:
        logger.error(
            "BUILDER1_IMAGE_PROMPT_PREFLIGHT_FAIL campaignId=%s adIndex=%s reasons=%s",
            campaign_id or "",
            ad_index,
            reasons,
        )
        raise ImagePromptPreflightError(reasons)
