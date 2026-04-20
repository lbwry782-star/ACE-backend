"""
Compatibility bridge for Builder1 preview job wiring used by app.py.

Exceptions and GOAL_PAIR_* constants below are local copies matching
engine.side_by_side_v1. create_goal_pair_background is implemented here;
poll/cancel remain in side_by_side_v1. Prompt building stays in legacy module.
"""

import logging
import os
from typing import Optional

import httpx
from openai import OpenAI

from engine.side_by_side_v1 import (
    _build_goal_pair_prompt,
    poll_goal_pair_response,
    cancel_goal_pair_response,
)

logger = logging.getLogger(__name__)

# Exceptions for STEP0_BUNDLE timeout/errors (so app can return 504/500)
class Step0BundleTimeoutError(Exception):
    """STEP0_BUNDLE OpenAI call timed out."""
    pass


class Step0BundleOpenAIError(Exception):
    """STEP0_BUNDLE OpenAI call failed (non-timeout)."""
    pass


# Phase 2D: background mode max wait for GOAL_PAIR polling (then fallback)
GOAL_PAIR_BG_MAX_WAIT_SECONDS = 180  # 180s total; only trigger timeout fallback if status stays queued/in_progress beyond this
GOAL_PAIR_BG_CREATE_TIMEOUT_SECONDS = 15  # create with background=True returns quickly
GOAL_PAIR_BG_POLL_INTERVAL_SECONDS = 2  # initial backoff (progressive: 2→3→5→8→10s cap)

# Phase 2B: o3-pro single call for advertising_goal + 3 pairs (strict JSON).
GOAL_PAIR_MIN_SIMILARITY_ACCEPT = 40

GOAL_PAIR_RETRY_INSTRUCTION = """Follow the method again: anchor shape from product, shape search for silhouette similarity, cross-domain (A and B from different functional domains), second link (conceptual association only). a_sub and b_sub must be external, separate contextual objects (e.g. straw, bone, flower), never parts of the primary (no sole, horn opening, pip faces, cap, handle, etc.). Derive advertising_goal from the pair. Return the same JSON schema."""

__all__ = [
    "Step0BundleTimeoutError",
    "Step0BundleOpenAIError",
    "create_goal_pair_background",
    "poll_goal_pair_response",
    "cancel_goal_pair_response",
    "GOAL_PAIR_BG_MAX_WAIT_SECONDS",
    "GOAL_PAIR_BG_POLL_INTERVAL_SECONDS",
    "GOAL_PAIR_MIN_SIMILARITY_ACCEPT",
    "GOAL_PAIR_RETRY_INSTRUCTION",
]


def create_goal_pair_background(
    product_name: str,
    product_description: str,
    request_id: str,
    retry_instruction: Optional[str] = None,
) -> Optional[str]:
    """
    Create o3-pro GOAL_PAIR request in OpenAI Background Mode. No retries.
    If retry_instruction is set, appends it to the prompt (for one extra attempt after low similarity).
    Returns response_id (str) for polling, or None on create failure.
    """
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(GOAL_PAIR_BG_CREATE_TIMEOUT_SECONDS),
        max_retries=0,
    )
    prompt = _build_goal_pair_prompt(product_name or "", product_description or "description")
    if retry_instruction:
        prompt = prompt.rstrip() + "\n\n" + retry_instruction.strip()
    # Temporary debug: Stage 2 prompt length/preview for analysis (do not log full prompt in production).
    _chars = len(prompt)
    _tokens_est = _chars // 4
    _preview = prompt[:500].replace("\n", " ")
    logger.info(f"STAGE2_PROMPT_LENGTH chars={_chars} request_id={request_id}")
    logger.info(f"STAGE2_PROMPT_TOKENS_EST={_tokens_est} request_id={request_id}")
    logger.info(f"STAGE2_PROMPT_PREVIEW {_preview!r} request_id={request_id}")
    try:
        response = client.responses.create(
            model="o3-pro",
            input=prompt,
            reasoning={"effort": "low"},
            background=True,
        )
        response_id = getattr(response, "id", None)
        if not response_id:
            logger.info(f"STAGE2_RESULT_FAIL request_id={request_id} reason=no_response_id")
            logger.error("GOAL_PAIR_BG_CREATE_FAIL no response id returned")
            return None
        logger.info(f"GOAL_PAIR_BG_CREATE_OK response_id={response_id} request_id={request_id}")
        return response_id
    except Exception as e:
        logger.info(f"STAGE2_RESULT_FAIL request_id={request_id} reason=create_error")
        logger.error(f"GOAL_PAIR_BG_CREATE_FAIL error={e} request_id={request_id}")
        return None
