"""
Builder1 campaign-series planning entry point (active production).
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, TypeAlias

from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_plan_parser import (
    Builder1SeriesPlanParseError,
    validate_series_plan_structure,
)
from engine.builder1_plan_spec import Builder1SeriesPlan
from engine.builder1_planning_contract import (
    BUILDER1_PLANNING_SYSTEM_PROMPT,
    build_builder1_planning_user_prompt,
    build_builder1_series_repair_user_prompt,
)

logger = logging.getLogger(__name__)

MAX_PLANNING_ATTEMPTS = 3


def _preview_text(value: object, *, limit: int = 500) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _coerce_plan_dict(raw_payload: object) -> Dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        text = raw_payload.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no_json_object")
        obj = json.loads(text[start : end + 1])
        if not isinstance(obj, dict):
            raise ValueError("model_output_not_object")
        return obj
    raise ValueError("model_output_not_object")


class Builder1PlannerError(RuntimeError):
    pass


PlanningModelCaller: TypeAlias = Callable[[str, str], object]


def _try_parse(
    raw_payload: object,
    *,
    normalized_product_name: str,
    normalized_product_description: str,
    normalized_format: str,
    ad_count: int,
) -> tuple[Optional[Builder1SeriesPlan], list[str], Optional[Dict[str, Any]]]:
    raw_dict = _coerce_plan_dict(raw_payload)
    candidate, reasons = validate_series_plan_structure(
        raw_dict,
        expected_format=normalized_format,
        expected_ad_count=ad_count,
        product_name=normalized_product_name,
        product_description=normalized_product_description,
    )
    return candidate, reasons, raw_dict


def plan_builder1(
    product_name: object,
    product_description: object,
    format_value: object,
    model_caller: PlanningModelCaller,
    *,
    ad_count: int = 2,
    brand_guidelines: Optional[Dict[str, Any]] = None,
) -> Builder1SeriesPlan:
    """Plan one Builder1 campaign (2–4 ads). No creative-output memory."""
    normalized = normalize_builder1_input(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        ad_count=ad_count,
        brand_guidelines=brand_guidelines,
    )
    logger.info(
        "BUILDER1_SERIES_PLANNING_START productName=%s format=%s adCount=%s",
        normalized.product_name,
        normalized.format,
        normalized.ad_count,
    )

    user_prompt = build_builder1_planning_user_prompt(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        ad_count=normalized.ad_count,
        brand_guidelines=brand_guidelines,
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_PLANNING_ATTEMPTS + 1):
        logger.info("BUILDER1_SERIES_PLANNING_MODEL_CALL_START attempt=%s", attempt)
        try:
            raw_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            last_error = exc
            logger.error(
                "BUILDER1_SERIES_PLAN_REJECTED stage=model_call attempt=%s err=%s",
                attempt,
                exc,
            )
            continue

        raw_preview = _preview_text(raw_payload)
        logger.info("BUILDER1_SERIES_PLANNING_PARSE_START attempt=%s preview=%s", attempt, raw_preview)
        try:
            plan, reasons, raw_dict = _try_parse(
                raw_payload,
                normalized_product_name=normalized.product_name,
                normalized_product_description=normalized.product_description,
                normalized_format=normalized.format,
                ad_count=normalized.ad_count,
            )
        except Exception as exc:
            last_error = exc
            logger.error(
                "BUILDER1_SERIES_PLAN_REJECTED stage=coerce attempt=%s err=%s",
                attempt,
                exc,
            )
            continue

        if plan is not None:
            forced_name = normalized.product_name or plan.product_name_resolved
            return replace(
                plan,
                product_name=forced_name,
                product_name_resolved=forced_name,
                product_description=normalized.product_description,
                format=normalized.format,
                ad_count=normalized.ad_count,
            )

        logger.error(
            "BUILDER1_SERIES_PLAN_REJECTED stage=validation attempt=%s reasons=%s",
            attempt,
            reasons,
        )

        logger.info("BUILDER1_SERIES_REPAIR attempt=%s reasons=%s", attempt, reasons)
        repair_prompt = build_builder1_series_repair_user_prompt(
            product_name=normalized.product_name,
            product_description=normalized.product_description,
            format_value=normalized.format,
            ad_count=normalized.ad_count,
            broken_plan_json=json.dumps(raw_dict, ensure_ascii=False),
            rejection_reasons=reasons,
            brand_guidelines=brand_guidelines,
        )
        try:
            repaired_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, repair_prompt)
            repaired_plan, repair_reasons, _ = _try_parse(
                repaired_payload,
                normalized_product_name=normalized.product_name,
                normalized_product_description=normalized.product_description,
                normalized_format=normalized.format,
                ad_count=normalized.ad_count,
            )
            if repaired_plan is not None:
                forced_name = normalized.product_name or repaired_plan.product_name_resolved
                final = replace(
                    repaired_plan,
                    product_name=forced_name,
                    product_name_resolved=forced_name,
                    product_description=normalized.product_description,
                    format=normalized.format,
                    ad_count=normalized.ad_count,
                )
                logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", final.ad_count)
                return final
            logger.error(
                "BUILDER1_SERIES_PLAN_REJECTED stage=repair_validation reasons=%s",
                repair_reasons,
            )
        except Exception as exc:
            last_error = exc
            logger.error("BUILDER1_SERIES_PLAN_REJECTED stage=repair err=%s", exc)

    if last_error:
        raise Builder1PlannerError("planning_failed") from last_error
    raise Builder1PlannerError("planning_failed")
