"""
Disconnected Builder1 dry orchestration scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.builder1_input_normalizer import (
    NormalizedBuilder1Input,
    normalize_builder1_input,
)
from engine.builder1_plan_parser import parse_builder1_plan
from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_planning_contract import build_builder1_planning_user_prompt
from engine.builder1_visual_prompt import build_visual_prompt


@dataclass
class Builder1DryRunResult:
    normalized_input: NormalizedBuilder1Input
    planning_user_prompt: str
    parsed_plan: Builder1Plan
    visual_prompt: str


def run_builder1_dry_pipeline(
    product_name: object,
    product_description: object,
    format_value: object,
    model_payload: object,
) -> Builder1DryRunResult:
    normalized_input = normalize_builder1_input(
        product_name=product_name,
        product_description=product_description,
        format_value=format_value,
    )
    planning_user_prompt = build_builder1_planning_user_prompt(
        product_name=normalized_input.product_name,
        product_description=normalized_input.product_description,
        format_value=normalized_input.format,
    )
    parsed_plan = parse_builder1_plan(model_payload)
    visual_prompt = build_visual_prompt(parsed_plan)
    return Builder1DryRunResult(
        normalized_input=normalized_input,
        planning_user_prompt=planning_user_prompt,
        parsed_plan=parsed_plan,
        visual_prompt=visual_prompt,
    )
