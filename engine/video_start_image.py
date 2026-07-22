"""
ACE video start frame: single gpt-image still from o3 plan (isolated from /preview /generate image engine).

Used only as Runway gen4_turbo promptImage when planning succeeds.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

from engine import openai_retry

logger = logging.getLogger(__name__)

_VIDEO_START_IMAGE_TIMEOUT = float((os.environ.get("VIDEO_START_IMAGE_TIMEOUT_SECONDS") or "90").strip() or "90")


def _plan_first_scene_variation(plan: Dict[str, Any]) -> str:
    variations = plan.get("sceneVariations")
    if isinstance(variations, list):
        for item in variations:
            text = str(item or "").strip()
            if text:
                return text
    scene = (plan.get("sceneConcept") or "").strip()
    if " | " in scene:
        return scene.split(" | ", 1)[0].strip()
    return scene


def build_ace_start_frame_image_prompt(plan: Dict[str, Any]) -> str:
    """
    Opening still for Builder2 variation-montage plans (Runway gen4_turbo promptImage).
    No text, logos, or headline in frame.
    """
    no_text = (
        "No text, letters, words, numbers as graphics, captions, labels, signage, packaging typography, "
        "title cards, watermarks, headline, UI, or brand names in the image — blank/generic surfaces only."
    )

    product = (plan.get("productNameResolved") or "").strip()
    core_visual = (plan.get("coreVisualIdea") or "").strip()
    opening = (plan.get("openingFrameDescription") or "").strip()
    first_variation = _plan_first_scene_variation(plan)
    video_prompt = (plan.get("videoPrompt") or plan.get("videoPromptCore") or "").strip()

    scene_focus = opening
    if not scene_focus and (plan.get("structureType") or "").strip() == "continuous_event":
        sequence = plan.get("sequence") or {}
        scene_focus = (sequence.get("beginning") or "").strip()
    if not scene_focus:
        scene_focus = first_variation or core_visual or video_prompt[:400]
    essence = core_visual or scene_focus

    from engine.builder2_runway_config import resolve_builder2_video_duration_seconds

    n = resolve_builder2_video_duration_seconds()
    product_clause = f"Product context (do not show as readable text): {product}. " if product else ""
    is_continuous = (plan.get("structureType") or "").strip() == "continuous_event"
    shot_kind = "continuous event" if is_continuous else "commercial montage"
    brief = (
        f"Single photorealistic still frame, opening shot for a silent {n}-second {shot_kind}. "
        f"The still must be the opening moment from which action can develop naturally over {n} seconds — "
        "not the final resolution. "
        f"{product_clause}"
        f"Core visual idea: {essence}. "
        f"Opening moment to animate: {scene_focus}. "
        "Realistic human scene when applicable; clear composition; soft natural lighting; realistic materials. "
        f"{no_text}"
    )
    return brief


def generate_video_start_image_data_uri(plan: Dict[str, Any]) -> Optional[str]:
    """
    Generate one start image from plan; return data URI for Runway promptImage, or None on any failure.
    """
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_START_IMAGE skip: OPENAI_API_KEY missing")
        return None

    from engine.builder2_runway_config import resolve_builder2_start_image_size

    prompt = build_ace_start_frame_image_prompt(plan)
    model = (os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1.5").strip()
    size = resolve_builder2_start_image_size()
    quality = (os.environ.get("VIDEO_START_IMAGE_QUALITY") or "low").strip()

    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(_VIDEO_START_IMAGE_TIMEOUT),
        max_retries=0,
    )
    try:
        response = openai_retry.openai_call_with_retry(
            lambda: client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
            ),
            endpoint="images",
        )
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            logger.error("VIDEO_START_IMAGE FAIL empty response data")
            return None
        data_uri = f"data:image/png;base64,{b64}"
        logger.info("VIDEO_START_IMAGE build_ok model=%s size=%s quality=%s", model, size, quality)
        logger.info("VIDEO_START_IMAGE source=generated_from_plan")
        return data_uri
    except Exception as e:
        logger.warning("VIDEO_START_IMAGE FAIL err_type=%s err=%s", type(e).__name__, e)
        return None
