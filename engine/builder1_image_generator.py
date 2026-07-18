"""
Builder1 single-ad image generation (active production — one image per user action).
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from engine.builder1_image_compliance import (
    BUILDER1_IMAGE_COMPLIANCE_CORRECTION_BLOCK,
    ComplianceReviewer,
    ImageComplianceError,
    ImageComplianceUnavailableError,
    review_builder1_ad_image_compliance,
)
from engine.builder1_image_prompt_preflight import (
    ImagePromptPreflightError,
    run_image_prompt_preflight,
)
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_product_visibility import (
    ProductVisibilityPolicy,
    build_no_product_strict_correction,
    build_visibility_compliance_correction,
)
from engine.builder1_visual_prompt import build_visual_prompt

logger = logging.getLogger(__name__)

ImageCaller = Callable[[str, str], bytes]

VISIBILITY_VIOLATION_CODES = frozenset(
    {
        "product_visible_without_explicit_request",
        "packaging_visible_without_explicit_request",
        "product_used_as_physical_generator",
        "product_used_as_main_visual",
    }
)


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


def _build_compliance_correction_block(
    violations: list[str],
    *,
    series_plan: Builder1SeriesPlan,
    campaign_id: str,
    ad_index: int,
) -> str:
    visibility_violations = [code for code in violations if code in VISIBILITY_VIOLATION_CODES]
    if "product_visible_without_explicit_request" in visibility_violations:
        transferred = series_plan.transferred_object or series_plan.physical_generator
        action = series_plan.transferred_object_action or series_plan.physical_generator_campaign_role
        logger.info(
            "BUILDER1_IMAGE_RETRY_CORRECTION campaignId=%s adIndex=%s "
            "violation=product_visible_without_explicit_request correctionProfile=NO_PRODUCT_STRICT",
            campaign_id or "",
            ad_index,
        )
        return build_no_product_strict_correction(
            transferred_object=transferred,
            transferred_object_action=action,
        )

    blocks = [BUILDER1_IMAGE_COMPLIANCE_CORRECTION_BLOCK]
    if visibility_violations:
        blocks.append(build_visibility_compliance_correction(visibility_violations))
    return "\n\n".join(blocks)


def generate_builder1_ad_image(
    series_plan: Builder1SeriesPlan,
    ad_index: int,
    image_caller: ImageCaller,
    *,
    compliance_reviewer: Optional[ComplianceReviewer] = None,
    campaign_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Builder1AdImageResult:
    """
    Generate exactly one image for one planned ad index.
    One retry on transient non-rate-limit failures. Rate limits are not retried here.
    """
    ad = next(a for a in series_plan.ads if a.index == ad_index)
    prompt = build_visual_prompt(series_plan, ad)
    run_image_prompt_preflight(
        series_plan=series_plan,
        ad_plan=ad,
        prompt=prompt,
        campaign_id=campaign_id or "",
        ad_index=ad_index,
    )

    image_started = time.perf_counter()
    image_bytes = _generate_with_retry(
        ad_index=ad_index,
        prompt=prompt,
        format_value=series_plan.format,
        image_caller=image_caller,
    )
    image_generation_duration_ms = int((time.perf_counter() - image_started) * 1000)

    policy = _resolve_visibility_policy(series_plan)
    review_product_description = (
        series_plan.product_description
        if policy != ProductVisibilityPolicy.FORBIDDEN
        else ""
    )

    compliance_started = time.perf_counter()
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
    )
    compliance_review_duration_ms = int((time.perf_counter() - compliance_started) * 1000)
    if review.passed:
        return Builder1AdImageResult(
            index=ad_index,
            visual_prompt=prompt,
            image_bytes=image_bytes,
            image_generation_duration_ms=image_generation_duration_ms,
            compliance_review_duration_ms=compliance_review_duration_ms,
            compliance_regeneration_count=0,
        )

    logger.info(
        "BUILDER1_IMAGE_COMPLIANCE_REGENERATE campaignId=%s jobId=%s adIndex=%s violations=%s",
        campaign_id or "",
        job_id or "",
        ad_index,
        review.violations,
    )
    corrected_prompt = f"{prompt}\n\n{_build_compliance_correction_block(review.violations, series_plan=series_plan, campaign_id=campaign_id or '', ad_index=ad_index)}"
    regen_started = time.perf_counter()
    image_bytes = _generate_with_retry(
        ad_index=ad_index,
        prompt=corrected_prompt,
        format_value=series_plan.format,
        image_caller=image_caller,
    )
    image_generation_duration_ms += int((time.perf_counter() - regen_started) * 1000)
    review2_started = time.perf_counter()
    review2 = review_builder1_ad_image_compliance(
        image_bytes,
        product_name=series_plan.product_name_resolved,
        ad_index=ad_index,
        campaign_id=campaign_id,
        job_id=job_id,
        product_description=review_product_description,
        visibility_policy=series_plan.product_visibility_policy,
        transferred_object=series_plan.transferred_object or series_plan.physical_generator,
        reviewer=compliance_reviewer,
    )
    compliance_review_duration_ms += int((time.perf_counter() - review2_started) * 1000)
    if review2.passed:
        return Builder1AdImageResult(
            index=ad_index,
            visual_prompt=corrected_prompt,
            image_bytes=image_bytes,
            image_generation_duration_ms=image_generation_duration_ms,
            compliance_review_duration_ms=compliance_review_duration_ms,
            compliance_regeneration_count=1,
        )
    if not review2.violations:
        raise ImageComplianceUnavailableError("malformed_response", ad_index=ad_index)
    raise ImageComplianceError(review2.violations, ad_index=ad_index)


def image_bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def generate_builder1_series_images(*_args, **_kwargs):
    raise NotImplementedError("Use generate_builder1_ad_image for incremental generation")


def generate_builder1_image(*_args, **_kwargs):
    raise NotImplementedError("Use generate_builder1_ad_image")
