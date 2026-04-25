"""
Disconnected Builder1 planning entry point scaffold.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, TypeAlias

from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_plan_parser import parse_builder1_plan
from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_planning_contract import (
    BUILDER1_PLANNING_SYSTEM_PROMPT,
    build_builder1_planning_user_prompt,
)

logger = logging.getLogger(__name__)


class Builder1PlannerError(RuntimeError):
    pass


PlanningModelCaller: TypeAlias = Callable[[str, str], object]


def plan_builder1(
    product_name: object,
    product_description: object,
    format_value: object,
    model_caller: PlanningModelCaller,
) -> Builder1Plan:
    normalized = normalize_builder1_input(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
    )
    user_prompt = build_builder1_planning_user_prompt(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
    )
    try:
        raw_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        raise Builder1PlannerError("planning_model_call_failed") from exc
    plan = parse_builder1_plan(raw_payload)
    forced_resolved_name = normalized.product_name or plan.product_name_resolved
    final_plan = replace(
        plan,
        product_name=forced_resolved_name,
        product_name_resolved=forced_resolved_name,
        product_description=normalized.product_description,
        format=normalized.format,
    )
    logger.info(
        "BUILDER1_PLAN_OK "
        "product_name=%r product_description=%r format=%r detected_language=%r "
        "advertising_promise=%r object_a=%r object_a_secondary=%r object_b=%r "
        "visual_similarity_score=%s mode_decision=%r visual_description=%r",
        final_plan.product_name,
        final_plan.product_description,
        final_plan.format,
        final_plan.detected_language,
        final_plan.advertising_promise,
        final_plan.object_a,
        final_plan.object_a_secondary,
        final_plan.object_b,
        final_plan.visual_similarity_score,
        final_plan.mode_decision,
        final_plan.visual_description,
    )
    logger.info(
        "BUILDER1_MODE_DECISION "
        "object_a=%r object_a_secondary=%r object_b=%r visual_similarity_score=%s mode_decision=%r",
        final_plan.object_a,
        final_plan.object_a_secondary,
        final_plan.object_b,
        final_plan.visual_similarity_score,
        final_plan.mode_decision,
    )
    return final_plan
