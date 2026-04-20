"""
Compatibility bridge for Builder1 preview job wiring used by app.py.

Exceptions and GOAL_PAIR_* constants below are local copies matching
engine.side_by_side_v1. Goal-pair callables are still imported from there.
"""

from engine.side_by_side_v1 import (
    create_goal_pair_background,
    poll_goal_pair_response,
    cancel_goal_pair_response,
)

# Exceptions for STEP0_BUNDLE timeout/errors (so app can return 504/500)
class Step0BundleTimeoutError(Exception):
    """STEP0_BUNDLE OpenAI call timed out."""
    pass


class Step0BundleOpenAIError(Exception):
    """STEP0_BUNDLE OpenAI call failed (non-timeout)."""
    pass


# Phase 2D: background mode max wait for GOAL_PAIR polling (then fallback)
GOAL_PAIR_BG_MAX_WAIT_SECONDS = 180  # 180s total; only trigger timeout fallback if status stays queued/in_progress beyond this
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
