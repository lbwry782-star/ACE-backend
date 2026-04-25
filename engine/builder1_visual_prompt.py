"""
Disconnected Builder1 visual-prompt builder scaffold.
"""
from __future__ import annotations

import logging

from engine.builder1_plan_spec import Builder1Plan, MODE_REPLACEMENT, MODE_SIDE_BY_SIDE

logger = logging.getLogger(__name__)

_NEGATIVE_CONSTRAINTS = (
    "pure white background",
    "no text",
    "no letters",
    "no numbers",
    "no logos",
    "no brand marks",
    "no packaging text",
    "no signage text",
    "no watermark",
)


def _base_constraints_text() -> str:
    return ", ".join(_NEGATIVE_CONSTRAINTS)


def build_visual_prompt(plan: Builder1Plan) -> str:
    base = (
        "Photorealistic studio product photography. "
        f"{_base_constraints_text()}."
    )
    supporting = (plan.visual_description or "").strip()

    if plan.mode_decision == MODE_SIDE_BY_SIDE:
        core = (
            f"Show {plan.object_a} and {plan.object_b} partially overlapping. "
            "Do not include any secondary object."
        )
    elif plan.mode_decision == MODE_REPLACEMENT:
        logger.info(
            "BUILDER1_REPLACEMENT_VISUAL_RULES "
            "object_a=%r object_a_secondary=%r object_b=%r "
            "rule_object_a_absent=true rule_object_b_replaces_object_a=true rule_secondary_remains=true",
            plan.object_a,
            plan.object_a_secondary,
            plan.object_b,
        )
        core = (
            f"Do not show {plan.object_a}. "
            f"Show {plan.object_b} replacing {plan.object_a} in {plan.object_a}'s position/context. "
            f"Keep {plan.object_a_secondary} visible and interacting with {plan.object_b}. "
            f"Only {plan.object_b} and {plan.object_a_secondary} should appear as the main visual objects."
        )
    else:
        raise ValueError("unsupported_mode")

    if supporting:
        return f"{base} {core} Supporting visual direction: {supporting}."
    return f"{base} {core}"
