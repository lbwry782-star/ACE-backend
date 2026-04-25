"""
Disconnected Builder1 planning entry point scaffold.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, TypeAlias

from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_memory import (
    get_builder1_memory_snapshot,
    remember_object_a,
    was_object_a_used,
)
from engine.builder1_plan_parser import parse_builder1_plan
from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_planning_contract import (
    BUILDER1_PLANNING_SYSTEM_PROMPT,
    build_builder1_planning_user_prompt,
)

logger = logging.getLogger(__name__)

_REPLACEMENT_BLOCKLIST: set[tuple[str, str]] = {
    ("megaphone", "trumpet"),
    ("megaphone", "conch shell"),
    ("bullhorn", "trumpet"),
    ("bullhorn", "conch shell"),
}


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
    memory = get_builder1_memory_snapshot()
    remembered_object_a = memory.get("object_a") or []
    recent_object_a = remembered_object_a[-10:]
    logger.info(
        "BUILDER1_MEMORY_INJECTED_TO_PLANNING object_a_count=%s recent_object_a=%r",
        len(remembered_object_a),
        recent_object_a,
    )
    user_prompt = build_builder1_planning_user_prompt(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        remembered_object_a=remembered_object_a,
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
    if was_object_a_used(final_plan.object_a):
        logger.info(
            "BUILDER1_MEMORY_OBJECT_A_REPEAT_DETECTED object_a=%r action=%r",
            final_plan.object_a,
            "logged_only",
        )
    if final_plan.mode_decision == "REPLACEMENT":
        pair = (
            (final_plan.object_a or "").strip().lower(),
            (final_plan.object_b or "").strip().lower(),
        )
        if pair in _REPLACEMENT_BLOCKLIST:
            previous_score = final_plan.visual_similarity_score
            reason = "blocked_invalid_replacement_pair_not_replacement_grade"
            suffix = (
                "Pair downgraded to SIDE_BY_SIDE because it is not replacement-grade "
                "(requires changed role/grip/context interaction)."
            )
            merged_visual_description = (final_plan.visual_description or "").strip()
            if merged_visual_description:
                merged_visual_description = f"{merged_visual_description} {suffix}"
            else:
                merged_visual_description = suffix
            final_plan = replace(
                final_plan,
                visual_similarity_score=84,
                mode_decision="SIDE_BY_SIDE",
                visual_description=merged_visual_description,
            )
            logger.info(
                "BUILDER1_REPLACEMENT_DOWNGRADED "
                "object_a=%r object_b=%r previous_score=%s new_score=%s reason=%r",
                final_plan.object_a,
                final_plan.object_b,
                previous_score,
                final_plan.visual_similarity_score,
                reason,
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
    logger.info("BUILDER1_MEMORY_OBJECT_A_REMEMBER_CALL object_a=%r", final_plan.object_a)
    remember_object_a(final_plan.object_a)
    return final_plan
