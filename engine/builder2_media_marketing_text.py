"""
Builder2 media-only marketing text — deterministic reuse without model calls.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from engine.builder2_headline_decision_contract import headline_decision_requires_headline


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _job_marketing_text(job_data: Optional[Dict[str, Any]]) -> str:
    if not isinstance(job_data, dict):
        return ""
    for key in ("marketingText", "marketing_text"):
        text = _clean_text(job_data.get(key))
        if text:
            return text
    return ""


def _state_marketing_text(state: Dict[str, Any]) -> str:
    media = state.get("mediaResume")
    if isinstance(media, dict):
        text = _clean_text(media.get("marketingText"))
        if text:
            return text
    return ""


def _plan_marketing_text(plan: Dict[str, Any]) -> str:
    for key in ("marketingText", "deliveryMarketingText", "existingMarketingText"):
        text = _clean_text(plan.get(key))
        if text:
            return text
    return ""


def build_deterministic_media_marketing_fallback(
    *,
    product_name: str,
    headline_decision: str,
    headline_text: str = "",
) -> str:
    name = _clean_text(product_name) or "Product"
    if headline_decision_requires_headline(headline_decision):
        headline = _clean_text(headline_text)
        if headline:
            return f"{name}. {headline}"
    return f"{name}. Video delivery."


def resolve_media_resume_marketing_text(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_data: Optional[Dict[str, Any]] = None,
    product_name: str = "",
    headline_decision: str = "omit",
) -> Tuple[str, str]:
    headline = _clean_text(plan.get("headlineText"))
    for source, text in (
        ("persisted_job", _job_marketing_text(job_data)),
        ("delivery_existing", _state_marketing_text(state)),
        ("winner_existing", _plan_marketing_text(plan)),
    ):
        if text:
            return text, source
    return (
        build_deterministic_media_marketing_fallback(
            product_name=product_name or _clean_text(plan.get("productNameResolved")),
            headline_decision=headline_decision,
            headline_text=headline,
        ),
        "deterministic_fallback",
    )


def build_media_marketing_dry_run_report(
    *,
    state: Dict[str, Any],
    plan: Dict[str, Any],
    job_data: Optional[Dict[str, Any]] = None,
    product_name: str = "",
    headline_decision: str = "omit",
) -> Dict[str, Any]:
    _, source = resolve_media_resume_marketing_text(
        state=state,
        plan=plan,
        job_data=job_data,
        product_name=product_name,
        headline_decision=headline_decision,
    )
    return {
        "marketingCopyRequired": False,
        "marketingCopySource": source,
        "marketingCopyModelAllowed": False,
    }
