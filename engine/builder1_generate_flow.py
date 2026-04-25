"""
Disconnected Builder1 generate-flow scaffold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from engine.builder1_image_generator import ImageModelCaller, generate_builder1_image
from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_planner import PlanningModelCaller, plan_builder1

logger = logging.getLogger(__name__)


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
    logger.info(
        "BUILDER1_PLAN_OK "
        "product_name=%r product_description=%r format=%r detected_language=%r "
        "advertising_promise=%r object_a=%r object_a_secondary=%r object_b=%r "
        "visual_similarity_score=%s mode_decision=%r visual_description=%r",
        plan.product_name,
        plan.product_description,
        plan.format,
        plan.detected_language,
        plan.advertising_promise,
        plan.object_a,
        plan.object_a_secondary,
        plan.object_b,
        plan.visual_similarity_score,
        plan.mode_decision,
        plan.visual_description,
    )
    logger.info(
        "BUILDER1_MODE_DECISION "
        "object_a=%r object_a_secondary=%r object_b=%r visual_similarity_score=%s mode_decision=%r",
        plan.object_a,
        plan.object_a_secondary,
        plan.object_b,
        plan.visual_similarity_score,
        plan.mode_decision,
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
