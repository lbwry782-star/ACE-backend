"""Shared Builder1 test helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator
from unittest.mock import patch

from engine.builder1_image_compliance import ImageComplianceResult


@contextmanager
def public_builder1_flask_security() -> Iterator[None]:
    """
    Process-only public Ori Flask security: ACE_SECURITY_ENABLED=true and mocked
    paid-auth success. Does not edit ENV files, touch Redis, or change product code.
    """
    from engine.ace_local_dev_config import reset_local_dev_config_for_tests
    import app as app_module

    saved = os.environ.get("ACE_SECURITY_ENABLED")
    reset_local_dev_config_for_tests()
    os.environ["ACE_SECURITY_ENABLED"] = "true"
    try:
        with patch.object(
            app_module,
            "_ace_security_peek_builder1_create",
            return_value=(None, None, None, None),
        ), patch.object(
            app_module,
            "_ace_security_guard_builder1_create",
            return_value=(None, None, None),
        ), patch.object(
            app_module,
            "_ace_security_guard_builder1_campaign",
            return_value=None,
        ), patch.object(
            app_module,
            "_ace_security_guard_builder1_job",
            return_value=None,
        ):
            yield
    finally:
        if saved is None:
            os.environ.pop("ACE_SECURITY_ENABLED", None)
        else:
            os.environ["ACE_SECURITY_ENABLED"] = saved


def attach_public_builder1_flask_security(test_case: Any) -> None:
    cm = public_builder1_flask_security()
    test_case._public_builder1_flask_security_cm = cm
    cm.__enter__()


def detach_public_builder1_flask_security(test_case: Any) -> None:
    cm = getattr(test_case, "_public_builder1_flask_security_cm", None)
    if cm is None:
        return
    test_case._public_builder1_flask_security_cm = None
    cm.__exit__(None, None, None)


def pass_compliance_reviewer(**_kwargs: Any) -> ImageComplianceResult:
    """Explicit test-only injected reviewer — approves without a real multimodal call."""
    return ImageComplianceResult(passed=True, violations=[], confidence="high")


def seed_builder1_image_job(
    *,
    job_id: str,
    campaign_id: str,
    ad_index: int,
    target_ad_count: int,
    plan_revision: int = 1,
    stage: str = "generating_images",
) -> None:
    from engine.builder1_jobs_store import create_builder1_job, update_builder1_job

    create_builder1_job(
        job_id=job_id,
        campaign_id=campaign_id,
        target_ad_count=target_ad_count,
        stage=stage,
    )
    update_builder1_job(
        job_id,
        planRevision=plan_revision,
        retryAdIndex=ad_index,
        campaignId=campaign_id,
    )


def marketing_text_words(count: int = 50, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(1, count + 1))


def marketing_text_hebrew(count: int = 50) -> str:
    return " ".join(f"מילה{i}" for i in range(1, count + 1))


def marketing_text_hebrew_with_brand(count: int = 50, brand: str = "TestBrand") -> str:
    words: list[str] = []
    brand_at = max(1, count // 2)
    word_num = 1
    for i in range(1, count + 1):
        if i == brand_at:
            words.append(brand)
        else:
            words.append(f"מילה{word_num}")
            word_num += 1
    return " ".join(words)


def marketing_text_english_with_hebrew_brand(count: int = 50, brand: str = "מותג") -> str:
    words: list[str] = []
    brand_at = 5
    word_num = 1
    for i in range(1, count + 1):
        if i == brand_at:
            words.append(brand)
        else:
            words.append(f"word{word_num}")
            word_num += 1
    return " ".join(words)


DEFAULT_STRATEGY_BOUNDARY_FIELDS: Dict[str, Any] = {
    "campaignExecutableNow": True,
    "requiresClientConsultation": False,
    "clientActionLevel": "none",
    "implementationCostLevel": "none",
    "simpleStrategicAction": None,
}


def strategy_scan_candidate(
    *,
    index: int,
    lens: str,
    problem: str | None = None,
    advantage: str | None = None,
    brief_support: str = "Follows from brief reinforced shell mention",
    **boundary_overrides: Any,
) -> Dict[str, Any]:
    candidate = {
        "id": f"S{index:02d}",
        "lens": lens,
        "strategicProblem": problem or f"Distinct buyer problem {index}",
        "relativeAdvantage": advantage or f"Distinct advantage {index}",
        "briefSupport": brief_support,
        "advantageSource": "observable_product_mechanism",
        "claimRisk": "low",
        **DEFAULT_STRATEGY_BOUNDARY_FIELDS,
    }
    candidate.update(boundary_overrides)
    return candidate


def marketing_text_with_punctuation() -> str:
    words = [f"word{i}" for i in range(1, 50)]
    words[0] = "Hello,"
    words.append("finished.")
    return " ".join(words)


def direct_product_route_assessment(
    *,
    readable: bool = True,
    advantage_direct: bool = False,
    mechanism_available: bool = False,
    mechanism_summary: str = "",
    unique_gain: bool = True,
    unique_gain_text: str = "External object demonstrates the advantage through a causal mechanism unavailable from a generic product presentation.",
    translation_cost: str = "NONE",
    recommended_route: str = "ANALOGY_LED",
    route_reason: str = "External analogy supplies stronger causal proof than a direct product presentation.",
) -> Dict[str, Any]:
    return {
        "productOrCategoryImmediatelyReadable": readable,
        "relativeAdvantageDirectlyExpressibleWithProduct": advantage_direct,
        "productLedAdvertisingMechanismAvailable": mechanism_available,
        "productLedMechanismSummary": mechanism_summary,
        "externalAnalogyAddsUniquePersuasiveGain": unique_gain,
        "externalAnalogyUniqueGain": unique_gain_text if unique_gain else "",
        "additionalTranslationCost": translation_cost,
        "recommendedRoute": recommended_route,
        "routeDecisionReason": route_reason,
    }


def direct_product_route_assessment_product_led(
    *,
    mechanism_summary: str = "Product form demonstrates the relative advantage through a visible transformation.",
    route_reason: str = "Direct product mechanism is the strongest advertising route.",
) -> Dict[str, Any]:
    return direct_product_route_assessment(
        readable=True,
        advantage_direct=True,
        mechanism_available=True,
        mechanism_summary=mechanism_summary,
        unique_gain=False,
        unique_gain_text="",
        recommended_route="PRODUCT_LED",
        route_reason=route_reason,
    )


def direct_product_route_assessment_integrated(
    *,
    mechanism_summary: str = "Product participates as evidence inside a larger external mechanism.",
    unique_gain_text: str = "External mechanism proves the advantage while product supplies necessary physical evidence.",
    route_reason: str = "Integrated route combines product evidence with external analogy.",
) -> Dict[str, Any]:
    return direct_product_route_assessment(
        readable=True,
        advantage_direct=True,
        mechanism_available=True,
        mechanism_summary=mechanism_summary,
        unique_gain=True,
        unique_gain_text=unique_gain_text,
        recommended_route="PRODUCT_INTEGRATED_ANALOGY",
        route_reason=route_reason,
    )
