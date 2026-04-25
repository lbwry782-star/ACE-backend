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

_SHARED_FEATURE_ONLY_HINTS: tuple[str, ...] = (
    "cone",
    "bell",
    "handle",
    "tube",
    "rectangle",
    "roundness",
    "color",
)

_POSTURE_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "blown_instrument": ("trumpet", "trombone", "clarinet", "flute", "saxophone", "conch shell"),
    "voice_amplifier": ("megaphone", "bullhorn"),
    "drink_container": ("cup", "mug", "bottle", "glass", "can"),
    "seat": ("chair", "stool", "bench", "sofa"),
    "wearable": ("shoe", "boot", "hat", "helmet", "glove"),
}

_NEAR_IDENTICAL_VARIANTS: set[frozenset[str]] = {
    frozenset({"megaphone", "bullhorn"}),
}


def _norm_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _first_matching_category(name: str) -> str | None:
    n = _norm_text(name)
    if not n:
        return None
    for category, words in _POSTURE_CATEGORY_KEYWORDS.items():
        if any(word in n for word in words):
            return category
    return None


def _is_near_identical_variant(object_a: str, object_b: str) -> bool:
    a = _norm_text(object_a)
    b = _norm_text(object_b)
    if not a or not b:
        return False
    if a == b:
        return True
    return frozenset({a, b}) in _NEAR_IDENTICAL_VARIANTS


def _replacement_grade_downgrade_reason(plan: Builder1Plan) -> str | None:
    a = _norm_text(plan.object_a)
    b = _norm_text(plan.object_b)
    secondary = _norm_text(plan.object_a_secondary)
    visual_desc = _norm_text(plan.visual_description)
    pair = (a, b)

    if pair in _REPLACEMENT_BLOCKLIST:
        return "blocked_invalid_replacement_pair_not_replacement_grade"
    if _is_near_identical_variant(a, b):
        return "object_b_is_synonym_or_near_identical_variant_of_object_a"

    a_cat = _first_matching_category(a)
    b_cat = _first_matching_category(b)
    if a_cat and b_cat and a_cat != b_cat:
        return "object_b_is_different_functional_category_and_requires_different_use_posture"

    if secondary and any(k in secondary for k in ("hand", "grip", "holder", "stand", "mount")):
        if a_cat and b_cat and a_cat != b_cat:
            return "object_a_secondary_interaction_would_need_to_change_for_object_b"

    if visual_desc and "without changing" not in visual_desc:
        if any(k in visual_desc for k in ("different grip", "different posture", "reposition", "reconfigure")):
            return "object_b_cannot_occupy_object_a_context_without_scene_change"

    if visual_desc and any(hint in visual_desc for hint in _SHARED_FEATURE_ONLY_HINTS):
        if any(
            marker in visual_desc
            for marker in (
                "only",
                "just",
                "shared",
                "similar feature",
                "same shape",
            )
        ):
            return "similarity_based_only_on_single_shared_feature"

    return None


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
        reason = _replacement_grade_downgrade_reason(final_plan)
        if reason:
            previous_score = final_plan.visual_similarity_score
            suffix = f"[Downgraded from REPLACEMENT: {reason}]"
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
                "object_a=%r object_b=%r object_a_secondary=%r previous_score=%s new_score=%s reason=%r",
                final_plan.object_a,
                final_plan.object_b,
                final_plan.object_a_secondary,
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
