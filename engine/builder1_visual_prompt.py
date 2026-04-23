"""
Disconnected Builder1 visual-prompt builder scaffold.
"""
from __future__ import annotations

from engine.builder1_plan_spec import Builder1Plan, MODE_REPLACEMENT, MODE_SIDE_BY_SIDE

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
        core = (
            f"Show {plan.object_b} as the main object replacing {plan.object_a}. "
            f"Include {plan.object_a_secondary} interacting naturally with {plan.object_b} as if it belongs to it. "
            f"Do not show {plan.object_a} itself."
        )
    else:
        raise ValueError("unsupported_mode")

    if supporting:
        return f"{base} {core} Supporting visual direction: {supporting}."
    return f"{base} {core}"
