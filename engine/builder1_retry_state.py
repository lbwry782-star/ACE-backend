"""
Builder1 campaign retry mode constants and helpers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

RETRY_MODE_NONE = "none"
RETRY_MODE_IMAGE_ONLY = "image_only"
RETRY_MODE_REPAIR_FROM_PHYSICAL = "repair_from_physical"

VALID_RETRY_MODES = frozenset(
    {
        RETRY_MODE_NONE,
        RETRY_MODE_IMAGE_ONLY,
        RETRY_MODE_REPAIR_FROM_PHYSICAL,
    }
)


def normalize_retry_mode(value: Optional[str]) -> str:
    normalized = (value or RETRY_MODE_NONE).strip().lower()
    if normalized in VALID_RETRY_MODES:
        return normalized
    return RETRY_MODE_NONE


def resolve_authoritative_retry_mode(
    *,
    status: str,
    retry_mode: Optional[str],
) -> str:
    mode = normalize_retry_mode(retry_mode)
    if status == "physical_repair_required":
        return RETRY_MODE_REPAIR_FROM_PHYSICAL
    if status == "image_retry_required" and mode == RETRY_MODE_NONE:
        return RETRY_MODE_IMAGE_ONLY
    return mode


def public_retry_fields(
    *,
    session: Any,
    retry_ad_index: Optional[int] = None,
    repair_in_progress: Optional[bool] = None,
) -> Dict[str, Any]:
    mode = resolve_authoritative_retry_mode(
        status=getattr(session, "status", "") or "active",
        retry_mode=getattr(session, "retry_mode", None),
    )
    ad_index = retry_ad_index
    if ad_index is None:
        ad_index = getattr(session, "failed_ad_index", None) or session.next_ad_index
    retryable = mode != RETRY_MODE_NONE or bool(getattr(session, "failed_ad_index", None))
    return {
        "retryable": retryable,
        "retryMode": mode,
        "retryAdIndex": ad_index,
        "planningComplete": bool(getattr(session, "planning_complete", True)),
        "repairInProgress": bool(
            repair_in_progress
            if repair_in_progress is not None
            else getattr(session, "repair_in_progress", False)
        ),
        "planRevision": int(getattr(session, "plan_revision", 1) or 1),
        "status": getattr(session, "status", None) or "active",
    }
