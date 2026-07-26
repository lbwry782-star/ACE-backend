"""
Canonical public-base URL resolution for ACE video/media delivery.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PUBLIC_BASE_URL_FAILURE = "builder2_media_resume_not_configured:publicBaseUrl"


@dataclass(frozen=True)
class PublicBaseUrlResolution:
    configured: bool
    source: str
    value: str

    def to_safe_metadata(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "source": self.source or None,
        }


def normalize_public_base_url(raw: Any) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return normalized


def _candidate_value(job_data: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    if not isinstance(job_data, dict):
        return None
    value = job_data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_public_base_url(
    *,
    job_data: Optional[Dict[str, Any]] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> PublicBaseUrlResolution:
    precedence: tuple[tuple[str, Optional[str]], ...] = (
        ("job_public_base_url", _candidate_value(job_data, "public_base_url")),
        ("job_publicBaseUrl", _candidate_value(job_data, "publicBaseUrl")),
        ("ACE_PUBLIC_BASE_URL", (os.environ.get("ACE_PUBLIC_BASE_URL") or "").strip() or None),
        ("PUBLIC_BASE_URL", (os.environ.get("PUBLIC_BASE_URL") or "").strip() or None),
    )
    for source, raw in precedence:
        if raw is None:
            continue
        normalized = normalize_public_base_url(raw)
        if normalized:
            logger.info("PUBLIC_BASE_URL_RESOLVED source=%s configured=true", source)
            return PublicBaseUrlResolution(configured=True, source=source, value=normalized)
        logger.info("PUBLIC_BASE_URL_INVALID source=%s configured=false", source)
        return PublicBaseUrlResolution(configured=False, source=source, value="")

    logger.info("PUBLIC_BASE_URL_RESOLVED source=none configured=false")
    return PublicBaseUrlResolution(configured=False, source="", value="")


def require_public_base_url(
    *,
    job_data: Optional[Dict[str, Any]] = None,
    tournament_state: Optional[Dict[str, Any]] = None,
) -> PublicBaseUrlResolution:
    resolution = resolve_public_base_url(job_data=job_data, tournament_state=tournament_state)
    if not resolution.configured:
        from engine.builder2_tournament_contracts import Builder2TournamentError

        raise Builder2TournamentError(PUBLIC_BASE_URL_FAILURE)
    return resolution


def resolve_ace_public_base_url_from_env() -> PublicBaseUrlResolution:
    return resolve_public_base_url()
