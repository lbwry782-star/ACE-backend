"""
Canonical Runway REST API URL construction — single /v1 prefix, no double joins.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

DEFAULT_RUNWAY_ORIGIN = "https://api.dev.runwayml.com"
RUNWAY_API_PREFIX = "/v1"
_RUNWAY_VERSION_PREFIX_RE = re.compile(r"/v1/?$")


class RunwayUrlConfigurationError(ValueError):
    """Invalid Runway API URL configuration."""


@dataclass(frozen=True)
class RunwayUrlResolution:
    absoluteUrl: str
    origin: str
    apiPrefix: str
    endpointPath: str
    normalizedPath: str
    configuredBaseHadVersionPrefix: bool
    configuredBaseSource: str

    def to_safe_metadata(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "apiPrefix": self.apiPrefix,
            "endpointPath": self.endpointPath,
            "normalizedPath": self.normalizedPath,
            "configuredBaseHadVersionPrefix": self.configuredBaseHadVersionPrefix,
            "configuredBaseSource": self.configuredBaseSource,
            "runwayVersionPrefixCount": self.normalizedPath.count("/v1"),
        }


def _configured_base_candidates() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for name in ("RUNWAY_API_BASE", "RUNWAY_API_BASE_URL"):
        raw = (os.environ.get(name) or "").strip()
        if raw:
            pairs.append((name, raw))
    return tuple(pairs)


def resolve_configured_runway_base(*, configured_base: str | None = None) -> tuple[str, str]:
    if configured_base is not None:
        return "explicit", configured_base.strip()
    candidates = _configured_base_candidates()
    if candidates:
        return candidates[0]
    return "default", DEFAULT_RUNWAY_ORIGIN


def normalize_runway_origin(configured_base: str) -> tuple[str, bool]:
    raw = (configured_base or "").strip()
    if not raw:
        raw = DEFAULT_RUNWAY_ORIGIN
    had_version_prefix = False
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw.lstrip('/')}"
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    path = parsed.path or ""
    path = path.rstrip("/")
    if _RUNWAY_VERSION_PREFIX_RE.search(path):
        had_version_prefix = True
        path = _RUNWAY_VERSION_PREFIX_RE.sub("", path)
    path = re.sub(r"/+", "/", path)
    if path and not path.startswith("/"):
        path = f"/{path}"
    origin = urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/") or "", "", "", "")).rstrip("/")
    return origin, had_version_prefix


def _normalize_endpoint_path(endpoint: str) -> str:
    token = (endpoint or "").strip()
    if not token:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    token = token.lstrip("/")
    token = re.sub(r"/+", "/", token)
    if token.startswith("v1/"):
        token = token[3:]
    if token.startswith("v1"):
        token = token[2:].lstrip("/")
    if not token:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    return f"/{token}"


def build_runway_api_url(
    endpoint: str,
    *,
    configured_base: str | None = None,
) -> RunwayUrlResolution:
    source, base_raw = resolve_configured_runway_base(configured_base=configured_base)
    origin, had_version_prefix = normalize_runway_origin(base_raw)
    endpoint_path = _normalize_endpoint_path(endpoint)
    normalized_path = f"{RUNWAY_API_PREFIX}{endpoint_path}"
    absolute = f"{origin}{normalized_path}"
    return RunwayUrlResolution(
        absoluteUrl=absolute,
        origin=origin,
        apiPrefix=RUNWAY_API_PREFIX,
        endpointPath=endpoint_path,
        normalizedPath=normalized_path,
        configuredBaseHadVersionPrefix=had_version_prefix,
        configuredBaseSource=source,
    )


def validate_runway_api_url(resolution: RunwayUrlResolution) -> None:
    parsed = urlparse(resolution.absoluteUrl)
    if parsed.scheme != "https":
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    if not parsed.netloc:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    if parsed.query or parsed.fragment:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    if resolution.normalizedPath.count("/v1") != 1:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    if "/v1/v1/" in resolution.absoluteUrl:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    auth = parsed.netloc.split("@")
    if len(auth) > 1:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")


def build_runway_image_to_video_url(*, configured_base: str | None = None) -> RunwayUrlResolution:
    resolution = build_runway_api_url("image_to_video", configured_base=configured_base)
    validate_runway_api_url(resolution)
    if resolution.normalizedPath != "/v1/image_to_video":
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    return resolution


def build_runway_text_to_video_url(*, configured_base: str | None = None) -> RunwayUrlResolution:
    resolution = build_runway_api_url("text_to_video", configured_base=configured_base)
    validate_runway_api_url(resolution)
    if resolution.normalizedPath != "/v1/text_to_video":
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    return resolution


def build_runway_task_poll_url(task_id: str, *, configured_base: str | None = None) -> RunwayUrlResolution:
    task = (task_id or "").strip()
    if not task:
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    resolution = build_runway_api_url(f"tasks/{task}", configured_base=configured_base)
    validate_runway_api_url(resolution)
    if not resolution.normalizedPath.startswith("/v1/tasks/"):
        raise RunwayUrlConfigurationError("builder2_runway_invalid_endpoint_url")
    return resolution
