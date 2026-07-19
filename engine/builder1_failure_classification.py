"""
Builder1 failure classification — image execution vs plan contradiction.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import List, Sequence

from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan
from engine.builder1_product_identity_guard import detect_series_plan_visual_subject_conflicts
from engine.builder1_product_visibility import ProductVisibilityPolicy

logger = logging.getLogger(__name__)

PLAN_CONTRADICTION_VIOLATIONS = frozenset(
    {
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
        "packaging_visible_without_explicit_request",
    }
)


class Builder1FailureClass(str, Enum):
    IMAGE_EXECUTION = "IMAGE_EXECUTION"
    PLAN_CONTRADICTION = "PLAN_CONTRADICTION"


class Builder1FailureAction(str, Enum):
    REGENERATE_IMAGE = "REGENERATE_IMAGE"
    REPAIR_FROM_PHYSICAL = "REPAIR_FROM_PHYSICAL"


class PlanProductVisibilityConflictError(Exception):
    """Structured campaign plan contradicts FORBIDDEN product visibility."""

    def __init__(self, reasons: List[str], *, ad_index: int = 0):
        self.reasons = reasons
        self.ad_index = ad_index
        super().__init__(f"plan_product_visibility_conflict:{','.join(reasons)}")


class PlanContradictionComplianceError(Exception):
    """Image compliance indicates the stored plan contradicts visibility policy."""

    def __init__(self, violations: List[str], *, ad_index: int):
        self.violations = violations
        self.ad_index = ad_index
        super().__init__(f"plan_contradiction_compliance:{','.join(violations)}")


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


def validate_forbidden_plan_visibility(series_plan: Builder1SeriesPlan) -> List[str]:
    policy = _resolve_policy(series_plan)
    if policy != ProductVisibilityPolicy.FORBIDDEN:
        return []

    reasons = detect_series_plan_visual_subject_conflicts(series_plan)

    transferred = (series_plan.transferred_object or series_plan.physical_generator or "").strip()
    if not transferred:
        reasons.append("missing_transferred_object")

    internals = series_plan.planning_internals or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    for ad in series_plan.ads:
        extra = ad_internals.get(ad.index) or ad_internals.get(str(ad.index)) or {}
        if isinstance(extra, dict):
            if extra.get("productVisible") is True:
                reasons.append(f"ad_{ad.index}_product_visible_true")
            if extra.get("packagingVisible") is True:
                reasons.append(f"ad_{ad.index}_packaging_visible_true")
            if extra.get("productIsMainVisual") is True:
                reasons.append(f"ad_{ad.index}_product_is_main_visual")
            if extra.get("productIsPhysicalGenerator") is True:
                reasons.append(f"ad_{ad.index}_product_is_physical_generator")

    return list(dict.fromkeys(reasons))


def classify_compliance_failure(
    *,
    violations: Sequence[str],
    series_plan: Builder1SeriesPlan,
) -> tuple[Builder1FailureClass, Builder1FailureAction, List[str]]:
    plan_reasons = validate_forbidden_plan_visibility(series_plan)
    violation_set = set(violations)

    if violation_set & PLAN_CONTRADICTION_VIOLATIONS:
        return (
            Builder1FailureClass.PLAN_CONTRADICTION,
            Builder1FailureAction.REPAIR_FROM_PHYSICAL,
            list(dict.fromkeys(plan_reasons + list(violations))),
        )

    if plan_reasons:
        return (
            Builder1FailureClass.PLAN_CONTRADICTION,
            Builder1FailureAction.REPAIR_FROM_PHYSICAL,
            plan_reasons,
        )

    if violation_set == {"product_visible_without_explicit_request"} or (
        "product_visible_without_explicit_request" in violation_set and not (violation_set & PLAN_CONTRADICTION_VIOLATIONS)
    ):
        return (
            Builder1FailureClass.IMAGE_EXECUTION,
            Builder1FailureAction.REGENERATE_IMAGE,
            list(violations),
        )

    if violations:
        return (
            Builder1FailureClass.IMAGE_EXECUTION,
            Builder1FailureAction.REGENERATE_IMAGE,
            list(violations),
        )

    return Builder1FailureClass.IMAGE_EXECUTION, Builder1FailureAction.REGENERATE_IMAGE, []


def log_failure_classification(
    *,
    campaign_id: str,
    ad_index: int,
    failure_class: Builder1FailureClass,
    action: Builder1FailureAction,
) -> None:
    logger.info(
        "BUILDER1_FAILURE_CLASSIFIED campaignId=%s adIndex=%s failureClass=%s action=%s",
        campaign_id or "",
        ad_index,
        failure_class.value,
        action.value,
    )


def validate_ad_plan_for_forbidden_image(
    series_plan: Builder1SeriesPlan,
    ad_plan: Builder1AdPlan,
) -> List[str]:
    reasons = validate_forbidden_plan_visibility(series_plan)
    internals = series_plan.planning_internals or {}
    ad_internals = internals.get("adInternals") if isinstance(internals.get("adInternals"), dict) else {}
    extra = ad_internals.get(ad_plan.index) or ad_internals.get(str(ad_plan.index)) or {}
    if isinstance(extra, dict):
        if extra.get("productVisible") is True:
            reasons.append("plan_requests_product_visibility")
        if extra.get("productIsPhysicalGenerator") is True:
            reasons.append("plan_requests_product_as_generator")
    return list(dict.fromkeys(reasons))
