"""
Builder1 single-ad image generation (active production — one image per user action).
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from engine.builder1_campaign_store import (
    CampaignStoreError,
    cumulative_violations_for_ad,
    get_campaign_session,
    record_image_attempt_violations,
)
from engine.builder1_failure_classification import (
    Builder1FailureClass,
    PlanContradictionComplianceError,
    PlanProductVisibilityConflictError,
    classify_compliance_failure,
    log_failure_classification,
)
from engine.builder1_image_compliance import (
    ComplianceReviewer,
    ImageComplianceError,
    ImageComplianceResult,
    ImageComplianceUnavailableError,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_prompt_preflight import run_image_prompt_preflight
from engine.builder1_image_retry import (
    CORRECTION_PROFILE_CUMULATIVE,
    CORRECTION_PROFILE_MINIMAL_SAFE,
    CORRECTION_PROFILE_NORMAL,
    VISIBILITY_VIOLATION_CODES,
    build_cumulative_image_correction_block,
    build_minimal_safe_execution_block,
    next_attempt_number,
    normalize_violation_union,
    union_hard_violations_for_ad,
    union_violations_for_ad,
)
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_product_visibility import ProductVisibilityPolicy
from engine.builder1_visual_prompt import build_visual_prompt

logger = logging.getLogger(__name__)

ImageCaller = Callable[[str, str], bytes]
MAX_INTERNAL_IMAGE_ATTEMPTS = 3


class ImageRateLimitError(Exception):
    def __init__(self, message: str = "image_rate_limited", *, retry_after_seconds: Optional[int] = None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


@dataclass
class Builder1AdImageResult:
    index: int
    visual_prompt: str
    image_bytes: bytes
    image_generation_duration_ms: int = 0
    compliance_review_duration_ms: int = 0
    compliance_regeneration_count: int = 0
    compliance_advisories: List[str] | None = None


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {"RateLimitError", "APIStatusError"}:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


def _retry_after_seconds(exc: Exception) -> Optional[int]:
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", None) or {}
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return None


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


def _load_history(campaign_id: Optional[str], ad_index: int) -> dict:
    if not campaign_id:
        return {}
    try:
        session = get_campaign_session(campaign_id)
    except CampaignStoreError:
        return {}
    return dict(session.image_attempt_history or {})


def _evidence_summary(review: ImageComplianceResult) -> str:
    codes = list(review.hard_violations or []) + list(review.advisories or [])
    return ", ".join(codes[:6])


def _record_attempt(
    *,
    campaign_id: Optional[str],
    ad_index: int,
    attempt: int,
    review: ImageComplianceResult,
    correction_profile: str,
) -> dict:
    if not campaign_id:
        return {}
    session = record_image_attempt_violations(
        campaign_id,
        ad_index=ad_index,
        attempt=attempt,
        hard_violations=list(review.hard_violations or []),
        advisories=list(review.advisories or []),
        evidence_summary=_evidence_summary(review),
        correction_profile=correction_profile,
    )
    return dict(session.image_attempt_history or {})


def _generate_with_retry(
    *,
    ad_index: int,
    prompt: str,
    format_value: str,
    image_caller: ImageCaller,
) -> bytes:
    last_exc: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            logger.info("BUILDER1_SERIES_IMAGE_START adIndex=%s attempt=%s", ad_index, attempt)
            image_bytes = image_caller(prompt, format_value)
            logger.info("BUILDER1_SERIES_IMAGE_OK adIndex=%s", ad_index)
            return image_bytes
        except ImageRateLimitError:
            raise
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                retry_after = _retry_after_seconds(exc)
                logger.error(
                    "BUILDER1_IMAGE_RATE_LIMITED adIndex=%s retryAfterSeconds=%s",
                    ad_index,
                    retry_after,
                )
                raise ImageRateLimitError(retry_after_seconds=retry_after) from exc
            logger.error(
                "BUILDER1_SERIES_IMAGE_FAILED adIndex=%s attempt=%s err=%s",
                ad_index,
                attempt,
                exc,
            )
    raise RuntimeError(f"image_generation_failed_ad_{ad_index}") from last_exc


def _build_prompt_for_level(
    *,
    level: int,
    base_prompt: str,
    series_plan: Builder1SeriesPlan,
    ad,
    hard_violations: Sequence[str],
    campaign_id: str,
    ad_index: int,
    plan_revision: int,
) -> tuple[str, str]:
    if level <= 1:
        return base_prompt, CORRECTION_PROFILE_NORMAL
    if level == 2:
        block = build_cumulative_image_correction_block(
            violations=hard_violations,
            series_plan=series_plan,
            ad_plan=ad,
            campaign_id=campaign_id,
            ad_index=ad_index,
            plan_revision=plan_revision,
        )
        return f"{base_prompt}\n\n{block}", CORRECTION_PROFILE_CUMULATIVE
    minimal = build_minimal_safe_execution_block(series_plan=series_plan, ad_plan=ad)
    return f"{base_prompt}\n\n{minimal}", CORRECTION_PROFILE_MINIMAL_SAFE


def _handle_hard_failure(
    *,
    review: ImageComplianceResult,
    series_plan: Builder1SeriesPlan,
    campaign_id: Optional[str],
    ad_index: int,
    plan_revision: int,
) -> None:
    failure_class, action, _details, evidence = classify_compliance_failure(
        violations=list(review.raw_violations or review.violations or []),
        hard_violations=list(review.hard_violations or []),
        series_plan=series_plan,
    )
    log_failure_classification(
        campaign_id=campaign_id or "",
        ad_index=ad_index,
        failure_class=failure_class,
        action=action,
        evidence=evidence,
        plan_revision=plan_revision,
    )
    if failure_class == Builder1FailureClass.PLAN_CONTRADICTION:
        raise PlanContradictionComplianceError(list(review.hard_violations or []), ad_index=ad_index)


def generate_builder1_ad_image(
    series_plan: Builder1SeriesPlan,
    ad_index: int,
    image_caller: ImageCaller,
    *,
    compliance_reviewer: Optional[ComplianceReviewer] = None,
    campaign_id: Optional[str] = None,
    job_id: Optional[str] = None,
    plan_revision: int = 1,
    cumulative_violations: Optional[Sequence[str]] = None,
) -> Builder1AdImageResult:
    """
    Generate exactly one image for one planned ad index.
    Up to three materially different internal execution attempts before user retry.
    """
    if plan_revision < 1:
        raise CampaignStoreError("missing_plan_revision")

    ad = next(a for a in series_plan.ads if a.index == ad_index)
    base_prompt = build_visual_prompt(series_plan, ad)
    history = _load_history(campaign_id, ad_index)
    prior_hard = normalize_violation_union(
        list(cumulative_violations or []) + union_hard_violations_for_ad(history, ad_index)
    )
    if prior_hard:
        base_prompt = (
            f"{base_prompt}\n\n"
            f"{build_cumulative_image_correction_block(violations=prior_hard, series_plan=series_plan, ad_plan=ad, campaign_id=campaign_id or '', ad_index=ad_index, plan_revision=plan_revision)}"
        )

    try:
        run_image_prompt_preflight(
            series_plan=series_plan,
            ad_plan=ad,
            prompt=base_prompt,
            campaign_id=campaign_id or "",
            ad_index=ad_index,
        )
    except PlanProductVisibilityConflictError:
        raise

    policy = _resolve_visibility_policy(series_plan)
    review_product_description = (
        series_plan.product_description
        if policy != ProductVisibilityPolicy.FORBIDDEN
        else ""
    )

    image_generation_duration_ms = 0
    compliance_review_duration_ms = 0
    regeneration_count = 0
    last_prompt = base_prompt
    collected_advisories: List[str] = []
    session_hard_violations: List[str] = list(prior_hard)

    for level in range(1, MAX_INTERNAL_IMAGE_ATTEMPTS + 1):
        hard_union = normalize_violation_union(
            session_hard_violations + union_hard_violations_for_ad(history, ad_index)
        )
        prompt, correction_profile = _build_prompt_for_level(
            level=level,
            base_prompt=base_prompt,
            series_plan=series_plan,
            ad=ad,
            hard_violations=hard_union,
            campaign_id=campaign_id or "",
            ad_index=ad_index,
            plan_revision=plan_revision,
        )
        last_prompt = prompt

        image_started = time.perf_counter()
        image_bytes = _generate_with_retry(
            ad_index=ad_index,
            prompt=prompt,
            format_value=series_plan.format,
            image_caller=image_caller,
        )
        image_generation_duration_ms += int((time.perf_counter() - image_started) * 1000)
        if level > 1:
            regeneration_count += 1

        compliance_started = time.perf_counter()
        try:
            review = review_builder1_ad_image_compliance(
                image_bytes,
                product_name=series_plan.product_name_resolved,
                ad_index=ad_index,
                campaign_id=campaign_id,
                job_id=job_id,
                product_description=review_product_description,
                visibility_policy=series_plan.product_visibility_policy,
                transferred_object=series_plan.transferred_object or series_plan.physical_generator,
                reviewer=compliance_reviewer,
                series_plan=series_plan,
                plan_revision=plan_revision,
            )
        except ImageComplianceUnavailableError as exc:
            raise ImageComplianceUnavailableError(
                exc.reason_code,
                ad_index=ad_index,
                image_bytes=image_bytes,
                visual_prompt=last_prompt,
                contract_repair_attempted=getattr(exc, "contract_repair_attempted", False),
            ) from exc
        compliance_review_duration_ms += int((time.perf_counter() - compliance_started) * 1000)

        if review.advisories:
            collected_advisories = list(dict.fromkeys(collected_advisories + list(review.advisories)))

        if review.passed:
            if review.advisories and campaign_id:
                record_image_attempt_violations(
                    campaign_id,
                    ad_index=ad_index,
                    attempt=next_attempt_number(history, ad_index),
                    hard_violations=[],
                    advisories=list(review.advisories),
                    evidence_summary=_evidence_summary(review),
                    correction_profile=correction_profile,
                )
            return Builder1AdImageResult(
                index=ad_index,
                visual_prompt=last_prompt,
                image_bytes=image_bytes,
                image_generation_duration_ms=image_generation_duration_ms,
                compliance_review_duration_ms=compliance_review_duration_ms,
                compliance_regeneration_count=regeneration_count,
                compliance_advisories=list(dict.fromkeys(collected_advisories)),
            )

        _handle_hard_failure(
            review=review,
            series_plan=series_plan,
            campaign_id=campaign_id,
            ad_index=ad_index,
            plan_revision=plan_revision,
        )

        attempt_number = next_attempt_number(history, ad_index)
        history = _record_attempt(
            campaign_id=campaign_id,
            ad_index=ad_index,
            attempt=attempt_number,
            review=review,
            correction_profile=correction_profile,
        )
        session_hard_violations = normalize_violation_union(
            session_hard_violations + list(review.hard_violations or [])
        )
        logger.info(
            "BUILDER1_IMAGE_COMPLIANCE_REGENERATE campaignId=%s jobId=%s adIndex=%s planRevision=%s level=%s hardViolations=%s advisories=%s",
            campaign_id or "",
            job_id or "",
            ad_index,
            plan_revision,
            level,
            review.hard_violations or [],
            review.advisories or [],
        )

        if level >= MAX_INTERNAL_IMAGE_ATTEMPTS:
            raise ImageComplianceError(
                list(review.hard_violations or []),
                ad_index=ad_index,
                hard_violations=list(review.hard_violations or []),
                advisories=list(review.advisories or []),
            )

    raise ImageComplianceError([], ad_index=ad_index)


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def generate_builder1_series_images(*_args, **_kwargs):
    raise NotImplementedError("Use generate_builder1_ad_image for incremental generation")


def generate_builder1_image(*_args, **_kwargs):
    raise NotImplementedError("Use generate_builder1_ad_image")
