"""
Builder1 failure classification — image execution vs plan contradiction.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import List, Sequence

from engine.builder1_plan_spec import Builder1AdPlan, Builder1SeriesPlan, series_plan_to_store_dict
from engine.builder1_product_identity_guard import detect_series_plan_visual_subject_conflicts
from engine.builder1_product_visibility import ProductVisibilityPolicy, resolve_product_visibility_policy

logger = logging.getLogger(__name__)

PIXEL_PLAN_CONTRADICTION_VIOLATIONS = frozenset(
    {
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
        "packaging_visible_without_explicit_request",
    }
)

PLAN_CATEGORY_INTEGRITY_VIOLATIONS = frozenset(
    {
        "competing_category_visual",
        "advertising_mechanism_not_observable",
        "public_analogy_too_complex",
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
    internals = series_plan.planning_internals or {}
    raw = series_plan.product_visibility_policy or internals.get("productVisibilityPolicy")
    return resolve_product_visibility_policy(raw)


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


def _structured_plan_conflict_reasons(series_plan: Builder1SeriesPlan) -> List[str]:
    reasons = validate_forbidden_plan_visibility(series_plan)
    from engine.builder1_advertising_comprehension import scan_plan_physical_repair_reasons

    plan_dict = series_plan_to_store_dict(series_plan)
    reasons.extend(scan_plan_physical_repair_reasons(plan_dict))
    return list(dict.fromkeys(reasons))


def plan_has_category_integrity_violation(series_plan: Builder1SeriesPlan) -> List[str]:
    from engine.builder1_advertising_comprehension import scan_plan_category_integrity

    reasons = scan_plan_category_integrity(series_plan_to_store_dict(series_plan))
    return [code for code in reasons if code in PLAN_CATEGORY_INTEGRITY_VIOLATIONS]


def classify_compliance_failure(
    *,
    violations: Sequence[str],
    series_plan: Builder1SeriesPlan,
    preflight_conflict: bool = False,
    hard_violations: Sequence[str] | None = None,
) -> tuple[Builder1FailureClass, Builder1FailureAction, List[str], dict[str, object]]:
    effective_hard = list(hard_violations if hard_violations is not None else violations)
    plan_reasons = _structured_plan_conflict_reasons(series_plan)
    structured_plan_conflict = bool(plan_reasons)
    violation_set = set(effective_hard)
    evidence = {
        "structuredPlanConflict": structured_plan_conflict,
        "preflightConflict": bool(preflight_conflict),
        "pixelReviewViolations": list(violations),
        "hardViolations": list(effective_hard),
    }
    plan_category_failures = [
        code for code in plan_reasons if code in PLAN_CATEGORY_INTEGRITY_VIOLATIONS
    ]
    if plan_category_failures:
        evidence["planCategoryIntegrityFailure"] = True

    if structured_plan_conflict or preflight_conflict:
        return (
            Builder1FailureClass.PLAN_CONTRADICTION,
            Builder1FailureAction.REPAIR_FROM_PHYSICAL,
            list(dict.fromkeys(plan_reasons + list(effective_hard))),
            evidence,
        )

    category_violations = violation_set & PLAN_CATEGORY_INTEGRITY_VIOLATIONS
    if category_violations:
        plan_category_failures = plan_has_category_integrity_violation(series_plan)
        if plan_category_failures:
            evidence["planCategoryIntegrityFailure"] = True
            return (
                Builder1FailureClass.PLAN_CONTRADICTION,
                Builder1FailureAction.REPAIR_FROM_PHYSICAL,
                list(dict.fromkeys(plan_category_failures + list(effective_hard))),
                evidence,
            )
        evidence["planCategoryIntegrityFailure"] = False
        return (
            Builder1FailureClass.IMAGE_EXECUTION,
            Builder1FailureAction.REGENERATE_IMAGE,
            list(effective_hard),
            evidence,
        )

    if not effective_hard:
        return (
            Builder1FailureClass.IMAGE_EXECUTION,
            Builder1FailureAction.REGENERATE_IMAGE,
            [],
            evidence,
        )

    if violation_set & PIXEL_PLAN_CONTRADICTION_VIOLATIONS:
        if structured_plan_conflict:
            return (
                Builder1FailureClass.PLAN_CONTRADICTION,
                Builder1FailureAction.REPAIR_FROM_PHYSICAL,
                list(dict.fromkeys(plan_reasons + list(effective_hard))),
                evidence,
            )
        return (
            Builder1FailureClass.IMAGE_EXECUTION,
            Builder1FailureAction.REGENERATE_IMAGE,
            list(effective_hard),
            evidence,
        )

    return (
        Builder1FailureClass.IMAGE_EXECUTION,
        Builder1FailureAction.REGENERATE_IMAGE,
        list(effective_hard),
        evidence,
    )


def log_failure_classification(
    *,
    campaign_id: str,
    ad_index: int,
    failure_class: Builder1FailureClass,
    action: Builder1FailureAction,
    evidence: dict[str, object] | None = None,
    plan_revision: int = 1,
) -> None:
    payload = evidence or {}
    revision = max(1, int(plan_revision or 1))
    logger.info(
        "BUILDER1_FAILURE_CLASSIFIED campaignId=%s adIndex=%s failureClass=%s action=%s "
        "structuredPlanConflict=%s preflightConflict=%s pixelReviewViolations=%s planRevision=%s",
        campaign_id or "",
        ad_index,
        failure_class.value,
        action.value,
        str(payload.get("structuredPlanConflict", False)).lower(),
        str(payload.get("preflightConflict", False)).lower(),
        payload.get("pixelReviewViolations") or [],
        revision,
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
