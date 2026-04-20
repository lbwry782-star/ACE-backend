"""
Compatibility bridge for Builder1 preview job wiring used by app.py.

Exceptions and GOAL_PAIR_* constants below are local copies matching
engine.side_by_side_v1. create_goal_pair_background and _build_goal_pair_prompt
are implemented here; poll/cancel remain in side_by_side_v1.
"""

import logging
import os
from typing import Optional

import httpx
from openai import OpenAI

from engine.side_by_side_v1 import (
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

GOAL_PAIR_O3_PROMPT_TEMPLATE = """Product: {product_name}
{product_description}

Return ONLY valid JSON. Schema is unchanged:
{{ "advertising_goal": "<string>", "pairs": [ {{ "a_primary": "<string>", "a_sub": "<string>", "b_primary": "<string>", "b_sub": "<string>", "silhouette_similarity": <number 0-100> }} ] }}
The field name remains "advertising_goal" for schema reasons, but it represents the advertising MESSAGE (not a business goal).

CORE PRINCIPLE
This is not a rational selling task. This is a bold conceptual MESSAGE emerging from a visual metaphor.
The MESSAGE may be surprising, poetic, playful, or even slightly irrational. It does NOT need to logically justify the product. It must feel conceptually sharp and intuitively powerful.
Trust the viewer's intelligence. Do NOT simplify ideas to make them obvious.

METHOD — FOLLOW STRICTLY IN ORDER

STEP 1 — ANCHOR SHAPE (PRODUCT-FIRST)
From productName + productDescription: Infer one primary anchor object representing the product domain.
Analyze only its dominant outer shape: overall silhouette, aspect ratio, orientation (horizontal/vertical/diagonal), curvature vs angularity, mass distribution, negative space.
This anchor is used ONLY for shape search.

STEP 2 — SHAPE SEARCH
Search for objects with strong silhouette similarity to the anchor.
Prefer: same orientation, same aspect ratio, similar mass balance, clear iconic outer contour.
Lighting/glow counts only if it defines silhouette.

STEP 3 — CROSS-DOMAIN (HARD RULE)
A and B MUST come from clearly different functional domains. Maximize functional difference.
If two objects serve similar practical functions, the pair is INVALID — even if shape is similar.
NO functional similarity. NO same-category substitutions. NO near-identical object classes.

STEP 4 — SECOND LINK (CONCEPTUAL ONLY)
After strong shape similarity is found, identify ONE additional conceptual link: idiom, trope, cultural association, known situation, outcome, symbolic echo.
This second link must be conceptual only. NO physical interaction. NO narrative. NO causality. NO one object operating the other. NO explanation mechanics.
The viewer must make a cognitive leap.

STEP 5 — DO NOT LOWER ABSTRACTION
If a strong shape pairing is found, DO NOT replace the object with a more literal or explanatory variant.
Never downgrade abstraction to "help" understanding. Do NOT alter object choice to make the MESSAGE clearer.
The MESSAGE must emerge from the objects — the objects must NOT change to match the MESSAGE.
Assume the viewer is capable of intuitive inference.

STEP 6 — MESSAGE DERIVATION
Derive a short, bold advertising MESSAGE from the combination of: shape similarity, cross-domain contrast, second conceptual link.
The MESSAGE must: be concise; feel headline-like; be persuasive or striking; not describe the drawing; not explain the metaphor; not use generic marketing phrases.
It may be bold or slightly wild. That is acceptable.

STEP 7 — OUTPUT EXACTLY 3 PAIRS
Rules: Exactly 3 pairs. No repeated primary objects across pairs. No swapping duplicates.
Secondary objects (a_sub, b_sub): Must be an external, separate, classic contextual object associated with the primary — never a physical part, surface feature, opening, handle, frame, base, sole, peel, cap, face, edge, horn, or other built-in component of the primary. Prefer: visually separate, naturally adjacent, classically associated (e.g. can→straw, dog→bone, bee→flower). Reject as invalid: sole, pip faces, horn opening, glassy edge, or any component/structural part of the primary. No "hand" unless concept absolutely requires gesture.

SIMILARITY SCORING
silhouette_similarity (0-100) must reflect: interchangeability of outer contour, orientation match, aspect ratio match, mass distribution similarity.
Conceptual strength MUST NOT inflate silhouette_similarity. If the selected object is a fragment/variant, score THAT exact form — not the canonical whole.

MODE NOTE (DO NOT OUTPUT MODE)
>= 85 -> REPLACEMENT (downstream logic). < 85 -> SIDE_BY_SIDE. You only output silhouette_similarity.

ABSOLUTE BANS
No functional similarity pairs. No narrative interaction. No object controlling another. No explanatory substitution. No repeated primary objects. No altering objects to make message easier. No generic marketing language.

END. Return JSON only.
{{"advertising_goal":"...","pairs":[{{"a_primary":"...","a_sub":"...","b_primary":"...","b_sub":"...","silhouette_similarity":0}},{{...}},{{...}}]}}"""

# When product name is empty: ask model to invent a creative brand-style name (not copied from description). Same method, extended schema.
GOAL_PAIR_O3_PROMPT_WHEN_NO_PRODUCT = """No product name provided.

If the user did not provide a product name, invent a short (1–2 words), original, memorable, advertising-style brand name that does not simply repeat or lightly rephrase words from the product description, unless unavoidable. It should feel like a distinctive new market brand, be easy to pronounce, and work naturally inside a headline. Avoid existing well-known brand names.

Description:
{product_description}

Return ONLY valid JSON. You MUST include "product_name" with your invented name. Schema:
{{ "advertising_goal": "<string>", "product_name": "<string>", "pairs": [ {{ "a_primary": "<string>", "a_sub": "<string>", "b_primary": "<string>", "b_sub": "<string>", "silhouette_similarity": <number 0-100> }} ] }}

(The field "advertising_goal" is the MESSAGE. Follow the same method: anchor shape from description, shape search, cross-domain, second link, message derivation, exactly 3 pairs. a_sub and b_sub must be external contextual objects, never physical parts of the primary. Output the same JSON structure with "product_name" added.)
"""


def _build_goal_pair_prompt(product_name: str, product_description: str) -> str:
    """Build GOAL_PAIR prompt. When product_name is empty, use variant that asks for invented product_name in JSON."""
    pn = (product_name or "").strip()
    desc = product_description or "description"
    if pn:
        return GOAL_PAIR_O3_PROMPT_TEMPLATE.format(product_name=pn, product_description=desc)
    logger.info("PRODUCT_NAME_FALLBACK_MODE=creative_brand_name")
    return GOAL_PAIR_O3_PROMPT_WHEN_NO_PRODUCT.format(product_description=desc)


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
