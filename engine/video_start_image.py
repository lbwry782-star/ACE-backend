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


def build_ace_start_frame_image_prompt(plan: Dict[str, Any]) -> str:
    """
    One still: both object A and object B visible together (ACE single interaction).
    No text, logos, or headline in frame.
    """
    oa = (plan.get("objectA") or "").strip()
    ob = (plan.get("objectB") or "").strip()

    no_text = (
        "No text, letters, words, numbers as graphics, captions, labels, signage, packaging typography, "
        "title cards, watermarks, headline, UI, or brand names in the image — blank/generic surfaces only."
    )

    opening = (plan.get("openingFrameDescription") or "").strip()

    brief = (
        f"Single photorealistic still frame, tight unified composition. "
        f"Both subjects visible: {oa} and {ob}, close together or slightly overlapping, same world and lighting. "
        f"Soft natural lighting, realistic materials. {no_text}"
    )
    if opening:
        return f"Creative brief: {opening} {brief}"
    return brief


def generate_video_start_image_data_uri(plan: Dict[str, Any]) -> Optional[str]:
    """
    Generate one start image from plan; return data URI for Runway promptImage, or None on any failure.
    """
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("VIDEO_START_IMAGE skip: OPENAI_API_KEY missing")
        return None

    prompt = build_ace_start_frame_image_prompt(plan)
    model = (os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-1.5").strip()
    size = (os.environ.get("VIDEO_START_IMAGE_SIZE") or "1536x1024").strip()
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
