"""
Disconnected Builder1 image-generation entry point scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

from engine.builder1_plan_spec import Builder1Plan
from engine.builder1_visual_prompt import build_visual_prompt


class Builder1ImageGenerationError(RuntimeError):
    pass


ImageModelCaller: TypeAlias = Callable[[str, str], bytes]


@dataclass
class Builder1ImageResult:
    visual_prompt: str
    image_bytes: bytes


def generate_builder1_image(
    plan: Builder1Plan,
    image_caller: ImageModelCaller,
) -> Builder1ImageResult:
    visual_prompt = build_visual_prompt(plan)
    try:
        image_bytes = image_caller(visual_prompt, plan.format)
    except Exception as exc:
        raise Builder1ImageGenerationError("builder1_image_call_failed") from exc
    return Builder1ImageResult(visual_prompt=visual_prompt, image_bytes=image_bytes)
