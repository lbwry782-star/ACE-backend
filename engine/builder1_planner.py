"""
Builder1 campaign-series planning entry point (active production).
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, TypeAlias

from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_plan_parser import validate_series_plan_structure
from engine.builder1_plan_spec import Builder1SeriesPlan, series_plan_to_store_dict
from engine.builder1_planning_contract import (
    BUILDER1_PLANNING_SYSTEM_PROMPT,
    build_builder1_planning_user_prompt,
    build_builder1_series_repair_user_prompt,
    build_builder1_strategy_repair_user_prompt,
    new_campaign_exploration_seed,
    shuffled_exploration_lens_order,
)
from engine.builder1_strategy_judge import judge_builder1_strategy

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


def _finalize_plan(plan: Builder1SeriesPlan, normalized) -> Builder1SeriesPlan:
    forced_name = normalized.product_name or plan.product_name_resolved
    return replace(
        plan,
        product_name=forced_name,
        product_name_resolved=forced_name,
        product_description=normalized.product_description,
        format=normalized.format,
        ad_count=normalized.ad_count,
    )


def _run_strategy_judge(
    *,
    product_description: str,
    plan: Builder1SeriesPlan,
    model_caller: PlanningModelCaller,
) -> tuple[bool, list[str], Dict[str, Any]]:
    plan_dict = series_plan_to_store_dict(plan)
    result = judge_builder1_strategy(
        product_description=product_description,
        plan_dict=plan_dict,
        model_caller=model_caller,
    )
    return result.passed, result.rejection_reason_codes, plan_dict


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
    lens_order = shuffled_exploration_lens_order()
    exploration_seed = new_campaign_exploration_seed()
    logger.info(
        "BUILDER1_SERIES_PLANNING_START productName=%s format=%s adCount=%s",
        normalized.product_name,
        normalized.format,
        normalized.ad_count,
    )
    logger.info("BUILDER1_STRATEGY_SCAN_START seed=%s lensOrder=%s", exploration_seed, ",".join(lens_order))

    user_prompt = build_builder1_planning_user_prompt(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        ad_count=normalized.ad_count,
        brand_guidelines=brand_guidelines,
        exploration_lens_order=lens_order,
        campaign_exploration_seed=exploration_seed,
    )

    last_error: Optional[Exception] = None
    final_plan: Optional[Builder1SeriesPlan] = None
    last_raw: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_PLANNING_ATTEMPTS + 1):
        logger.info("BUILDER1_SERIES_PLANNING_MODEL_CALL_START attempt=%s", attempt)
        try:
            raw_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            last_error = exc
            logger.error("BUILDER1_SERIES_PLAN_REJECTED stage=model_call attempt=%s err=%s", attempt, exc)
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
            last_raw = raw_dict
        except Exception as exc:
            last_error = exc
            logger.error("BUILDER1_SERIES_PLAN_REJECTED stage=coerce attempt=%s err=%s", attempt, exc)
            continue

        if plan is not None:
            final_plan = _finalize_plan(plan, normalized)
            break

        logger.error("BUILDER1_SERIES_PLAN_REJECTED stage=validation attempt=%s reasons=%s", attempt, reasons)
        logger.info("BUILDER1_SERIES_REPAIR attempt=%s reasons=%s", attempt, reasons)
        repair_prompt = build_builder1_series_repair_user_prompt(
            product_name=normalized.product_name,
            product_description=normalized.product_description,
            format_value=normalized.format,
            ad_count=normalized.ad_count,
            broken_plan_json=json.dumps(raw_dict, ensure_ascii=False),
            rejection_reasons=reasons,
            brand_guidelines=brand_guidelines,
            exploration_lens_order=lens_order,
            campaign_exploration_seed=exploration_seed,
        )
        try:
            repaired_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, repair_prompt)
            repaired_plan, repair_reasons, repaired_raw = _try_parse(
                repaired_payload,
                normalized_product_name=normalized.product_name,
                normalized_product_description=normalized.product_description,
                normalized_format=normalized.format,
                ad_count=normalized.ad_count,
            )
            last_raw = repaired_raw
            if repaired_plan is not None:
                final_plan = _finalize_plan(repaired_plan, normalized)
                break
            logger.error("BUILDER1_SERIES_PLAN_REJECTED stage=repair_validation reasons=%s", repair_reasons)
        except Exception as exc:
            last_error = exc
            logger.error("BUILDER1_SERIES_PLAN_REJECTED stage=repair err=%s", exc)

    if final_plan is None:
        if last_error:
            raise Builder1PlannerError("planning_failed") from last_error
        raise Builder1PlannerError("planning_failed")

    passed, judge_codes, plan_dict = _run_strategy_judge(
        product_description=normalized.product_description,
        plan=final_plan,
        model_caller=model_caller,
    )
    if not passed:
        logger.info("BUILDER1_SERIES_REPAIR stage=strategy_judge reasons=%s", judge_codes)
        strategy_repair = build_builder1_strategy_repair_user_prompt(
            product_name=normalized.product_name,
            product_description=normalized.product_description,
            format_value=normalized.format,
            ad_count=normalized.ad_count,
            broken_plan_json=json.dumps(last_raw or plan_dict, ensure_ascii=False),
            judge_reason_codes=judge_codes,
            brand_guidelines=brand_guidelines,
            exploration_lens_order=lens_order,
            campaign_exploration_seed=exploration_seed,
        )
        repaired_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, strategy_repair)
        repaired_plan, repair_reasons, _ = _try_parse(
            repaired_payload,
            normalized_product_name=normalized.product_name,
            normalized_product_description=normalized.product_description,
            normalized_format=normalized.format,
            ad_count=normalized.ad_count,
        )
        if repaired_plan is None:
            raise Builder1PlannerError("strategy_judge_failed")
        final_plan = _finalize_plan(repaired_plan, normalized)
        passed2, judge_codes2, plan_dict2 = _run_strategy_judge(
            product_description=normalized.product_description,
            plan=final_plan,
            model_caller=model_caller,
        )
        if not passed2:
            raise Builder1PlannerError("strategy_judge_failed")

    logger.info(
        "BUILDER1_STRATEGY_SELECTED adCount=%s brandSlogan=%s advantage=%s",
        final_plan.ad_count,
        final_plan.brand_slogan,
        final_plan.relative_advantage,
    )
    logger.info("BUILDER1_SERIES_PLANNING_OK adCount=%s", final_plan.ad_count)
    return final_plan
