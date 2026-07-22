"""
Builder2 reasoning model configuration — isolated from Builder1 and global OpenAI vars.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from engine.openai_reasoning import (
    DEFAULT_OPENAI_REASONING_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_MODE,
    VALID_REASONING_EFFORTS,
    VALID_REASONING_MODES,
    build_reasoning_payload,
    normalize_legacy_text_model,
)

logger = logging.getLogger(__name__)

DEFAULT_BUILDER2_REASONING_MODEL = DEFAULT_OPENAI_REASONING_MODEL
DEFAULT_BUILDER2_REASONING_MODE = DEFAULT_REASONING_MODE
DEFAULT_BUILDER2_REASONING_EFFORT = DEFAULT_REASONING_EFFORT


def resolve_builder2_reasoning_model() -> str:
    raw = (os.environ.get("BUILDER2_REASONING_MODEL") or "").strip()
    if not raw:
        return DEFAULT_BUILDER2_REASONING_MODEL
    return normalize_legacy_text_model(raw)


def resolve_builder2_reasoning_mode() -> str:
    raw = (os.environ.get("BUILDER2_REASONING_MODE") or "").strip().lower()
    if not raw:
        return DEFAULT_BUILDER2_REASONING_MODE
    if raw not in VALID_REASONING_MODES:
        logger.warning(
            "BUILDER2_REASONING_MODE_INVALID mode=%s fallback=%s",
            raw,
            DEFAULT_BUILDER2_REASONING_MODE,
        )
        return DEFAULT_BUILDER2_REASONING_MODE
    return raw


def resolve_builder2_reasoning_effort() -> str:
    raw = (os.environ.get("BUILDER2_REASONING_EFFORT") or "").strip().lower()
    if not raw:
        return DEFAULT_BUILDER2_REASONING_EFFORT
    if raw not in VALID_REASONING_EFFORTS:
        logger.warning(
            "BUILDER2_REASONING_EFFORT_INVALID effort=%s fallback=%s",
            raw,
            DEFAULT_BUILDER2_REASONING_EFFORT,
        )
        return DEFAULT_BUILDER2_REASONING_EFFORT
    return raw


def build_builder2_reasoning_payload() -> Dict[str, Any]:
    return build_reasoning_payload(
        effort=resolve_builder2_reasoning_effort(),
        mode=resolve_builder2_reasoning_mode(),
    )


def log_builder2_model_selected(
    *,
    role: str,
    call_type: str = "normal",
    attempt: Optional[int] = None,
) -> None:
    model = resolve_builder2_reasoning_model()
    mode = resolve_builder2_reasoning_mode()
    effort = resolve_builder2_reasoning_effort()
    parts = [
        "BUILDER2_MODEL_SELECTED",
        f"role={role}",
        f"model={model}",
        f"reasoningMode={mode}",
        f"reasoningEffort={effort}",
        f"callType={call_type}",
    ]
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    logger.info(" ".join(parts))
