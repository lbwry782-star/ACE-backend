"""
Builder1 image prompt preflight — structural checks before image API calls.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from engine.builder1_failure_classification import (
    PlanProductVisibilityConflictError,
    validate_ad_plan_for_forbidden_image,
)
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


class ImagePromptPreflightClassification(str, Enum):
    PLAN_VALID_NO_PRODUCT = "PLAN_VALID_NO_PRODUCT"
    PLAN_CONTRADICTS_PRODUCT_POLICY = "PLAN_CONTRADICTS_PRODUCT_POLICY"


@dataclass(frozen=True)
class ImagePromptPreflightResult:
    ok: bool
    classification: ImagePromptPreflightClassification
    reasons: List[str]


class ImagePromptPreflightError(Exception):
    """Campaign plan or prompt is structurally incompatible with image generation."""

    def __init__(
        self,
        reasons: List[str],
        *,
        classification: ImagePromptPreflightClassification = ImagePromptPreflightClassification.PLAN_CONTRADICTS_PRODUCT_POLICY,
    ):
        self.reasons = reasons
        self.classification = classification
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


def classify_image_prompt_plan(
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
    *,
    prompt: str = "",
) -> ImagePromptPreflightResult:
    policy = _resolve_policy(series_plan)
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        transferred = _norm(series_plan.transferred_object or series_plan.physical_generator)
        if not transferred:
            return ImagePromptPreflightResult(
                ok=False,
                classification=ImagePromptPreflightClassification.PLAN_CONTRADICTS_PRODUCT_POLICY,
                reasons=["missing_transferred_object"],
            )
        return ImagePromptPreflightResult(
            ok=True,
            classification=ImagePromptPreflightClassification.PLAN_VALID_NO_PRODUCT,
            reasons=[],
        )

    reasons = list(dict.fromkeys(validate_ad_plan_for_forbidden_image(series_plan, ad_plan)))
    if prompt:
        reasons.extend(validate_forbidden_visual_prompt_text(prompt, series_plan=series_plan).reasons)
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return ImagePromptPreflightResult(
            ok=False,
            classification=ImagePromptPreflightClassification.PLAN_CONTRADICTS_PRODUCT_POLICY,
            reasons=reasons,
        )
    return ImagePromptPreflightResult(
        ok=True,
        classification=ImagePromptPreflightClassification.PLAN_VALID_NO_PRODUCT,
        reasons=[],
    )


def validate_forbidden_visual_prompt_text(
    prompt: str,
    *,
    series_plan: Builder1SeriesPlan,
) -> ImagePromptPreflightResult:
    reasons: List[str] = []
    policy = _resolve_policy(series_plan)
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        return ImagePromptPreflightResult(
            ok=True,
            classification=ImagePromptPreflightClassification.PLAN_VALID_NO_PRODUCT,
            reasons=[],
        )

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

    if reasons:
        return ImagePromptPreflightResult(
            ok=False,
            classification=ImagePromptPreflightClassification.PLAN_CONTRADICTS_PRODUCT_POLICY,
            reasons=reasons,
        )
    return ImagePromptPreflightResult(
        ok=True,
        classification=ImagePromptPreflightClassification.PLAN_VALID_NO_PRODUCT,
        reasons=[],
    )


def run_image_prompt_preflight(
    *,
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
    prompt: str,
    campaign_id: str = "",
    ad_index: int = 0,
) -> ImagePromptPreflightClassification:
    result = classify_image_prompt_plan(series_plan, ad_plan, prompt=prompt)
    if result.ok:
        return result.classification

    logger.error(
        "BUILDER1_IMAGE_PROMPT_PREFLIGHT_FAIL campaignId=%s adIndex=%s classification=%s reasons=%s",
        campaign_id or "",
        ad_index,
        result.classification.value,
        result.reasons,
    )
    if result.classification == ImagePromptPreflightClassification.PLAN_CONTRADICTS_PRODUCT_POLICY:
        raise PlanProductVisibilityConflictError(result.reasons, ad_index=ad_index)
    raise ImagePromptPreflightError(result.reasons, classification=result.classification)
