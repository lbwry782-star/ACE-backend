"""
Builder2 Google Lyria configuration — isolated from Builder1 and global Gemini env vars.

Environment (Worker service):
  BUILDER2_LYRIA_ENABLED   — default false; when false, Builder2 media behaves as before Lyria.
  BUILDER2_LYRIA_API_KEY   — Google AI API key for Lyria (environment-only, never persisted).
  BUILDER2_LYRIA_MODEL     — model id, default lyria-3-pro-preview (Preview; override via env).

Optional:
  BUILDER2_LYRIA_ARTIFACT_DIR — durable local directory for per-job MP3 artifacts (Worker disk).

Builder2-only. No fallback to OpenAI or other music providers.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_BUILDER2_LYRIA_ENABLED = "BUILDER2_LYRIA_ENABLED"
ENV_BUILDER2_LYRIA_API_KEY = "BUILDER2_LYRIA_API_KEY"
ENV_BUILDER2_LYRIA_MODEL = "BUILDER2_LYRIA_MODEL"
ENV_BUILDER2_LYRIA_ARTIFACT_DIR = "BUILDER2_LYRIA_ARTIFACT_DIR"

DEFAULT_BUILDER2_LYRIA_ENABLED = False
DEFAULT_BUILDER2_LYRIA_MODEL = "lyria-3-pro-preview"

GEMINI_GENERATE_CONTENT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class Builder2LyriaConfigError(ValueError):
    """Invalid Builder2 Lyria configuration."""


def _parse_bool(raw: str, *, default: bool) -> bool:
    text = (raw or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise Builder2LyriaConfigError(f"builder2_lyria_invalid_bool:{raw}")


def resolve_builder2_lyria_enabled() -> bool:
    raw = os.environ.get(ENV_BUILDER2_LYRIA_ENABLED)
    if raw is None or not str(raw).strip():
        return DEFAULT_BUILDER2_LYRIA_ENABLED
    return _parse_bool(str(raw), default=DEFAULT_BUILDER2_LYRIA_ENABLED)


def resolve_builder2_lyria_model() -> str:
    raw = (os.environ.get(ENV_BUILDER2_LYRIA_MODEL) or "").strip()
    if not raw:
        return DEFAULT_BUILDER2_LYRIA_MODEL
    return raw


def resolve_builder2_lyria_api_key(*, required: bool = False) -> str:
    key = (os.environ.get(ENV_BUILDER2_LYRIA_API_KEY) or "").strip()
    if required and not key:
        raise Builder2LyriaConfigError("builder2_lyria_missing_api_key")
    return key


def resolve_builder2_lyria_generate_content_url(*, model: str | None = None) -> str:
    resolved_model = (model or resolve_builder2_lyria_model()).strip()
    if not resolved_model:
        raise Builder2LyriaConfigError("builder2_lyria_missing_model")
    return f"{GEMINI_GENERATE_CONTENT_BASE}/{resolved_model}:generateContent"


def resolve_builder2_lyria_artifact_dir() -> str:
    explicit = (os.environ.get(ENV_BUILDER2_LYRIA_ARTIFACT_DIR) or "").strip()
    if explicit:
        return explicit
    import tempfile
    from pathlib import Path

    return str(Path(tempfile.gettempdir()) / "ace_builder2_lyria")


def resolve_builder2_lyria_job_artifact_path(job_id: str):
    from pathlib import Path

    jid = (job_id or "").strip()
    if not jid:
        raise Builder2LyriaConfigError("builder2_lyria_missing_job_id")
    base = Path(resolve_builder2_lyria_artifact_dir())
    return base / jid / "soundtrack.mp3"
