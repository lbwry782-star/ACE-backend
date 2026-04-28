"""
Disconnected Builder1 planning entry point scaffold.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, TypeAlias

from engine.ace_usage_memory import get_used_object_a, remember_object_a as remember_object_a_ace
from engine.builder1_input_normalizer import normalize_builder1_input
from engine.builder1_plan_parser import parse_builder1_plan
from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_planning_contract import (
    BUILDER1_PLANNING_SYSTEM_PROMPT,
    build_builder1_planning_user_prompt,
)

logger = logging.getLogger(__name__)

_NON_PHYSICAL_SECONDARY_HINTS: tuple[str, ...] = (
    "idea",
    "concept",
    "promise",
    "benefit",
    "emotion",
    "feeling",
    "message",
    "slogan",
    "quality",
    "value",
)
_SCENE_CHANGE_HINTS: tuple[str, ...] = (
    "adapted",
    "changed grip",
    "different grip",
    "different posture",
    "different pose",
    "repositioned",
    "reconfigured",
    "adjusted hand",
)
_WEAK_REPLACEMENT_RATIONALE_HINTS: tuple[str, ...] = (
    "book-like",
    "magazine",
    "pages",
    "flat rectangular",
    "flat rectangle",
    "rectangular body",
    "open like a book",
    "hinge-like",
    "occupies the exact spot",
    "same spot",
)
_REPLACEMENT_CONTINUITY_HINTS: tuple[str, ...] = (
    "without changing pose",
    "without changing context",
    "without changing interaction",
    "unchanged interaction",
    "same role",
    "same secondary interaction",
    "preserves secondary interaction",
)
_HANDHELD_RATIONALE_HINTS: tuple[str, ...] = (
    "same hand",
    "same grip",
    "same position",
    "holding",
    "held",
    "grips",
    "occupies the same position",
)
_FLAT_CARD_PAPER_OBJECT_HINTS: tuple[str, ...] = (
    "card",
    "business card",
    "paper",
    "sheet",
    "flyer",
    "magazine",
    "book",
    "page",
    "postcard",
)


def _norm_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _repair_reasons(plan: Builder1Plan, *, object_a_repeated: bool) -> list[str]:
    reasons: list[str] = []
    a = _norm_text(plan.object_a)
    b = _norm_text(plan.object_b)
    secondary = _norm_text(plan.object_a_secondary)
    visual_desc = _norm_text(plan.visual_description)

    if object_a_repeated:
        reasons.append("object_a_already_used_in_memory")
    if a and b and a == b:
        reasons.append("object_b_identical_to_object_a")
    if not secondary:
        reasons.append("object_a_secondary_missing")
    elif any(h in secondary for h in _NON_PHYSICAL_SECONDARY_HINTS):
        reasons.append("object_a_secondary_not_concrete_physical")
    if plan.mode_decision == "REPLACEMENT" and any(h in visual_desc for h in _SCENE_CHANGE_HINTS):
        reasons.append("replacement_visual_description_indicates_scene_or_grip_change")
    if plan.mode_decision == "REPLACEMENT":
        has_weak_rationale = any(h in visual_desc for h in _WEAK_REPLACEMENT_RATIONALE_HINTS)
        has_continuity_explanation = any(h in visual_desc for h in _REPLACEMENT_CONTINUITY_HINTS)
        if has_weak_rationale and not has_continuity_explanation:
            reasons.append("replacement_rationale_too_generic_flat_booklike")
        has_hand_secondary = "hand" in secondary
        has_handheld_rationale = any(h in visual_desc for h in _HANDHELD_RATIONALE_HINTS)
        object_b_flat_paper_like = any(h in b for h in _FLAT_CARD_PAPER_OBJECT_HINTS)
        if has_hand_secondary and has_handheld_rationale and object_b_flat_paper_like:
            reasons.append("replacement_handheld_flat_object_needs_reconsideration")
    if plan.visual_similarity_score < 70:
        reasons.append("visual_similarity_below_side_by_side_minimum")
    return reasons


def _product_name_repair_reason(
    *,
    user_product_name: str,
    product_description: str,
    model_product_name_resolved: str,
) -> str | None:
    """When user left productName blank, reject description-as-name fallback."""

    if _norm_text(user_product_name):
        return None
    if _norm_text(model_product_name_resolved) == _norm_text(product_description):
        return "product_name_blank_model_used_product_description"
    return None


def _build_repair_user_prompt(
    *,
    product_name: str,
    product_description: str,
    format_value: str,
    remembered_object_a: list[str],
    previous_plan: Builder1Plan,
    reasons: list[str],
) -> str:
    memory_list = ", ".join(remembered_object_a) if remembered_object_a else "(none)"
    reasons_text = ", ".join(reasons)
    return (
        "The previous Builder1 plan violated explicit rules and must be corrected.\n"
        f"Violations: {reasons_text}\n"
        "Choose again and return the same JSON schema fields only.\n"
        "Preserve user context exactly:\n"
        f"- Product name: {product_name}\n"
        f"- Product description: {product_description}\n"
        f"- Format: {format_value}\n"
        f"- detectedLanguage must remain: {previous_plan.detected_language}\n"
        "Rules to obey:\n"
        f"- Object A must be fresh and not in memory: {memory_list}\n"
        "- Object A secondary must be a classic physical companion/context object of Object A.\n"
        "- REPLACEMENT only if Object B can replace Object A without changing pose/context/secondary interaction.\n"
        "- Weak generic rationale is invalid for REPLACEMENT: flat/open/rectangular/book-like/same-spot wording alone is never enough for 85+.\n"
        "- Hand-held replacement warning: same hand/same grip/same position is not enough for REPLACEMENT when objectB is flat/card/paper-like.\n"
        "- If a flat/card/paper object replaces another device/object merely by being held in the same hand or position, reconsider and either choose SIDE_BY_SIDE (<85) or choose a different objectB.\n"
        "- If Object B cannot preserve Object A's exact physical role, pose, context, and objectASecondary interaction, choose SIDE_BY_SIDE with score below 85.\n"
        "- Pairs with visualSimilarityScore below 70 are invalid for Builder1 and must be replaced with a new object pair.\n"
        "- Use score bands: 70-84 for SIDE_BY_SIDE, or 85+ for REPLACEMENT only when true replacement-grade continuity is satisfied.\n"
        "- If Product name is blank, generate a short new productNameResolved; do not copy productDescription into productNameResolved.\n"
        "- Choose a new plan if needed.\n"
        "- Otherwise choose SIDE_BY_SIDE.\n"
        "Previous invalid plan (for correction reference only):\n"
        f"- objectA: {previous_plan.object_a}\n"
        f"- objectASecondary: {previous_plan.object_a_secondary}\n"
        f"- objectB: {previous_plan.object_b}\n"
        f"- visualSimilarityScore: {previous_plan.visual_similarity_score}\n"
        f"- modeDecision: {previous_plan.mode_decision}\n"
        f"- visualDescription: {previous_plan.visual_description}\n"
    )


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
    used_object_a_ace = get_used_object_a("builder1")
    recent_object_a_ace = used_object_a_ace[-10:]
    remembered_object_a = used_object_a_ace
    logger.info(
        "BUILDER1_MEMORY_INJECTED_TO_PLANNING_ACE object_a_count=%s recent_object_a=%r",
        len(used_object_a_ace),
        recent_object_a_ace,
    )
    user_prompt = build_builder1_planning_user_prompt(
        product_name=normalized.product_name,
        product_description=normalized.product_description,
        format_value=normalized.format,
        remembered_object_a=remembered_object_a,
    )
    if used_object_a_ace:
        user_prompt = (
            f"{user_prompt}\n"
            "Object A memory from ACE Redis (avoid reusing or near-equivalent ideas):\n"
            f"- previous_object_a_ace: {', '.join(used_object_a_ace)}\n"
            "- Do not reuse any Object A listed above.\n"
        )
    try:
        raw_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        raise Builder1PlannerError("planning_model_call_failed") from exc
    plan = parse_builder1_plan(raw_payload)
    forced_resolved_name = normalized.product_name or plan.product_name_resolved
    name_reason = _product_name_repair_reason(
        user_product_name=normalized.product_name,
        product_description=normalized.product_description,
        model_product_name_resolved=forced_resolved_name,
    )
    final_plan = replace(
        plan,
        product_name=forced_resolved_name,
        product_name_resolved=forced_resolved_name,
        product_description=normalized.product_description,
        format=normalized.format,
    )
    reasons = _repair_reasons(final_plan, object_a_repeated=False)
    if name_reason:
        reasons.append(name_reason)
    if reasons:
        logger.info(
            "BUILDER1_PLAN_REPAIR_REQUESTED "
            "reasons=%r object_a=%r object_a_secondary=%r object_b=%r mode_decision=%r visual_similarity_score=%s",
            reasons,
            final_plan.object_a,
            final_plan.object_a_secondary,
            final_plan.object_b,
            final_plan.mode_decision,
            final_plan.visual_similarity_score,
        )
        repair_user_prompt = _build_repair_user_prompt(
            product_name=normalized.product_name,
            product_description=normalized.product_description,
            format_value=normalized.format,
            remembered_object_a=remembered_object_a,
            previous_plan=final_plan,
            reasons=reasons,
        )
        try:
            repaired_payload = model_caller(BUILDER1_PLANNING_SYSTEM_PROMPT, repair_user_prompt)
        except Exception as exc:
            raise Builder1PlannerError("planning_model_repair_call_failed") from exc
        repaired_plan = parse_builder1_plan(repaired_payload)
        forced_repaired_name = normalized.product_name or repaired_plan.product_name_resolved
        final_plan = replace(
            repaired_plan,
            product_name=forced_repaired_name,
            product_name_resolved=forced_repaired_name,
            product_description=normalized.product_description,
            format=normalized.format,
        )
        logger.info(
            "BUILDER1_PLAN_REPAIR_OK "
            "object_a=%r object_a_secondary=%r object_b=%r mode_decision=%r visual_similarity_score=%s",
            final_plan.object_a,
            final_plan.object_a_secondary,
            final_plan.object_b,
            final_plan.mode_decision,
            final_plan.visual_similarity_score,
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
    if (final_plan.object_a or "").strip():
        remember_object_a_ace("builder1", final_plan.object_a)
    return final_plan
