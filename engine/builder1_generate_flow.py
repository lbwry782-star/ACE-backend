"""
Disconnected Builder1 generate-flow scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.builder1_image_generator import ImageModelCaller, generate_builder1_image
from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_planner import PlanningModelCaller, plan_builder1


@dataclass
class Builder1GenerateResult:
    plan: Builder1Plan
    visual_prompt: str
    image_bytes: bytes


def generate_builder1_ad(
    product_name: object,
    product_description: object,
    format_value: object,
    planning_model_caller: PlanningModelCaller,
    image_caller: ImageModelCaller,
) -> Builder1GenerateResult:
    plan = plan_builder1(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
        model_caller=planning_model_caller,
    )
    image_result = generate_builder1_image(
        plan=plan,
        image_caller=image_caller,
    )
    return Builder1GenerateResult(
        plan=plan,
        visual_prompt=image_result.visual_prompt,
        image_bytes=image_result.image_bytes,
    )
