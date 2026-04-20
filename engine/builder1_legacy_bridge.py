"""
Compatibility re-exports for Builder1 preview job wiring used by app.py.

Decouples app from `from engine.side_by_side_v1 import ...` while preserving
behavior. Implementations remain in `engine.side_by_side_v1` until inlined.
"""

from engine.side_by_side_v1 import (
    Step0BundleTimeoutError,
    Step0BundleOpenAIError,
    create_goal_pair_background,
    poll_goal_pair_response,
    cancel_goal_pair_response,
    GOAL_PAIR_BG_MAX_WAIT_SECONDS,
    GOAL_PAIR_BG_POLL_INTERVAL_SECONDS,
    GOAL_PAIR_MIN_SIMILARITY_ACCEPT,
    GOAL_PAIR_RETRY_INSTRUCTION,
)

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
