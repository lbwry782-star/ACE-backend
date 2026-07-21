"""
Shared OpenAI reasoning-model configuration for ACE text / planning Responses API calls.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_REASONING_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_MODE = "standard"
DEFAULT_REASONING_EFFORT = "high"

VALID_REASONING_MODES = frozenset({"standard", "pro"})
VALID_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)

_config_logged = False


def normalize_legacy_text_model(model: str) -> str:
    """Map deprecated o3-pro alias to the current reasoning model."""
    normalized = (model or "").strip()
    if normalized == "o3-pro":
        return DEFAULT_OPENAI_REASONING_MODEL
    return normalized or DEFAULT_OPENAI_REASONING_MODEL


def resolve_openai_reasoning_model() -> str:
    raw = (os.environ.get("OPENAI_TEXT_MODEL") or DEFAULT_OPENAI_REASONING_MODEL).strip()
    return normalize_legacy_text_model(raw)


def resolve_reasoning_mode() -> str:
    raw = (os.environ.get("OPENAI_REASONING_MODE") or DEFAULT_REASONING_MODE).strip().lower()
    if raw not in VALID_REASONING_MODES:
        logger.warning(
            "OPENAI_REASONING_MODE_INVALID mode=%s fallback=%s",
            raw,
            DEFAULT_REASONING_MODE,
        )
        return DEFAULT_REASONING_MODE
    return raw


def resolve_default_reasoning_effort() -> str:
    raw = (os.environ.get("OPENAI_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT).strip().lower()
    if raw not in VALID_REASONING_EFFORTS:
        logger.warning(
            "OPENAI_REASONING_EFFORT_INVALID effort=%s fallback=%s",
            raw,
            DEFAULT_REASONING_EFFORT,
        )
        return DEFAULT_REASONING_EFFORT
    return raw


def build_reasoning_payload(
    *,
    effort: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_mode = (mode or resolve_reasoning_mode()).strip().lower()
    resolved_effort = (effort or resolve_default_reasoning_effort()).strip().lower()
    if resolved_mode not in VALID_REASONING_MODES:
        resolved_mode = DEFAULT_REASONING_MODE
    if resolved_effort not in VALID_REASONING_EFFORTS:
        resolved_effort = DEFAULT_REASONING_EFFORT
    return {"mode": resolved_mode, "effort": resolved_effort}


def model_supports_reasoning_config(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith(("o1", "o3", "gpt-5"))


def model_uses_responses_api(model: str) -> bool:
    return model_supports_reasoning_config(model)


def log_openai_reasoning_config(*, model: Optional[str] = None) -> None:
    global _config_logged
    if _config_logged:
        return
    _config_logged = True
    resolved_model = model or resolve_openai_reasoning_model()
    resolved_mode = resolve_reasoning_mode()
    resolved_effort = resolve_default_reasoning_effort()
    logger.info("OpenAI reasoning model: %s", resolved_model)
    logger.info("Reasoning mode: %s", resolved_mode)
    logger.info("Reasoning effort: %s", resolved_effort)
